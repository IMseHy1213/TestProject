"""v1.1 추적(tracing): 질문 하나가 어디서 얼마나 걸렸는지 한 줄기(trace)로 기록.

    trace (질문 하나)
     └─ span agent.run (총괄)                      ← 손님 질문, 방식, 결과
         ├─ span llm.tool_step (openai)            ← 모델, 토큰, 비용, 실패 이유
         ├─ span tool.search_listings              ← 성공 여부, 결과 수
         └─ span a2a.send → (다른 서버) a2a.serve → agent.run (검색 Agent) → tool.search_listings …

- OpenTelemetry 와 같은 개념(trace id 32자리 / span id 16자리 / 부모 / 속성 / 이벤트 / 상태), W3C traceparent 헤더로
  서버 사이에도 같은 trace 를 이어 감 (A2A 요청에 실어 보냄).
- 기록은 손님 응답을 늦추지 않도록 큐에 넣고 백그라운드에서 묶어서 저장 (SQLite, 선택: OTLP 로 Jaeger·Grafana 등에 전송).
- 개인정보: 질문은 전화번호·지번을 가리고 100자까지만, 방문자 id 는 바꾼 값만 기록.
"""

from __future__ import annotations

import contextvars
import os
import re
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

_current: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar("rea_span", default=None)
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
_PHONE = re.compile(r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}")
_LOT = re.compile(r"([가-힣]{1,6}(?:리|동))\s*(?:산\s*)?\d+(?:-\d+)?")


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str
    name: str
    service: str
    start_ns: int
    end_ns: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"          # ok / error
    status_msg: str = ""

    @property
    def duration_ms(self) -> float:
        return round(((self.end_ns or time.time_ns()) - self.start_ns) / 1e6, 2)

    def set(self, **attrs: Any) -> "Span":
        for k, v in attrs.items():
            if v is not None:
                self.attrs[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        return self

    def event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "ts_ns": time.time_ns(),
                            "attrs": {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                                      for k, v in attrs.items() if v is not None}})

    def error(self, message: str) -> None:
        self.status, self.status_msg = "error", str(message)[:300]


def redact(text: str, limit: int = 100) -> str:
    """기록용 질문: 전화번호·지번을 가리고 짧게."""
    t = _PHONE.sub("***", text or "")
    t = _LOT.sub(r"\1 ***", t)
    return t[:limit] + ("…" if len(t) > limit else "")


class Tracer:
    def __init__(self, service: str = "app"):
        self.service = service
        self.exporters: list[Any] = []
        self.ring: deque[Span] = deque(maxlen=int(os.getenv("REA_TRACE_RING", "5000")))
        self._lock = threading.Lock()
        self.enabled = os.getenv("REA_TRACING", "1") != "0"

    # ------------------------------------------------------------------ 시작/끝
    @contextmanager
    def span(self, name: str, traceparent: Optional[str] = None, **attrs: Any) -> Iterator[Span]:
        """with tracer.span('tool.search_listings', tool=...) as s: ...  (예외가 나면 상태 error 로 기록 후 다시 던짐)"""
        if not self.enabled:
            dummy = Span("0" * 32, "0" * 16, "", name, self.service, time.time_ns())
            yield dummy
            return
        parent = _current.get()
        remote = parse_traceparent(traceparent) if traceparent else None
        if remote:
            trace_id, parent_id = remote
        elif parent is not None:
            trace_id, parent_id = parent.trace_id, parent.span_id
        else:
            trace_id, parent_id = secrets.token_hex(16), ""
        s = Span(trace_id, secrets.token_hex(8), parent_id, name, self.service, time.time_ns())
        s.set(**attrs)
        token = _current.set(s)
        try:
            yield s
        except BaseException as e:
            s.error(f"{type(e).__name__}: {e}")
            raise
        finally:
            _current.reset(token)
            s.end_ns = time.time_ns()
            self._finish(s)

    def _finish(self, s: Span) -> None:
        with self._lock:
            self.ring.append(s)
        for ex in list(self.exporters):
            try:
                ex.export(s)
            except Exception:  # noqa: BLE001 - 기록이 실패해도 상담은 계속
                pass

    # ------------------------------------------------------------------ 조회 (메모리)
    def trace(self, trace_id: str) -> list[Span]:
        with self._lock:
            return [s for s in self.ring if s.trace_id == trace_id]

    def last_trace_id(self, root_name: str = "agent.run") -> Optional[str]:
        with self._lock:
            for s in reversed(self.ring):
                if s.name == root_name and not s.parent_id:
                    return s.trace_id
        return None

    def flush(self, timeout: float = 5.0) -> None:
        for ex in list(self.exporters):
            f = getattr(ex, "flush", None)
            if f:
                f(timeout)

    def shutdown(self) -> None:
        for ex in list(self.exporters):
            close = getattr(ex, "close", None)
            if close:
                close()
        self.exporters = []


TRACER = Tracer()


def span(name: str, **attrs: Any):
    return TRACER.span(name, **attrs)


def current() -> Optional[Span]:
    return _current.get()


def event(name: str, **attrs: Any) -> None:
    s = _current.get()
    if s is not None:
        s.event(name, **attrs)


def current_ids() -> tuple[str, str]:
    s = _current.get()
    return (s.trace_id, s.span_id) if s else ("", "")


def traceparent() -> Optional[str]:
    """지금 span 을 다른 서버에 이어 줄 W3C traceparent 헤더 값."""
    s = _current.get()
    return f"00-{s.trace_id}-{s.span_id}-01" if s and TRACER.enabled else None


def parse_traceparent(value: Optional[str]) -> Optional[tuple[str, str]]:
    m = _TRACEPARENT.match((value or "").strip().lower())
    if not m or m.group(1) == "0" * 32 or m.group(2) == "0" * 16:
        return None
    return m.group(1), m.group(2)


def copy_context():
    """다른 스레드에서도 같은 trace 를 이어 가도록 지금 문맥을 복사 (pool.submit(ctx.run, fn, ...))."""
    return contextvars.copy_context()
