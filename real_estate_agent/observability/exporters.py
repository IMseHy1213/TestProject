"""span 내보내기: 손님 응답을 늦추지 않도록 큐에 넣고 백그라운드 스레드가 묶어서 저장.

- SQLiteSpanExporter: 같은 DB 파일의 obs_spans 표 (관리자 '관측' 화면·python main.py trace 가 읽음).
  분산 상담팀(A2A 서버)이 같은 DB 파일을 쓰면 여러 서버의 span 이 한 trace 로 모임. 7일·5만 개 넘으면 오래된 것부터 삭제.
- OTLPHttpExporter: REA_OTLP_ENDPOINT (예: http://127.0.0.1:4318) 가 있으면 OpenTelemetry 표준(OTLP/HTTP JSON)으로 전송
  → Jaeger · Grafana Tempo · Honeycomb 등에서 그대로 볼 수 있음.
- 큐가 가득 차면(저장이 밀리면) 새 span 은 버리고 개수만 셈 (상담이 기록 때문에 멈추지 않게).
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .tracing import Span

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS obs_spans (
    span_id   TEXT PRIMARY KEY,
    trace_id  TEXT NOT NULL,
    parent_id TEXT,
    name      TEXT NOT NULL,
    service   TEXT,
    start_ns  INTEGER NOT NULL,
    dur_ms    REAL,
    status    TEXT,
    status_msg TEXT,
    attrs     TEXT,
    events    TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_spans_trace ON obs_spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_obs_spans_start ON obs_spans(start_ns);
CREATE INDEX IF NOT EXISTS idx_obs_spans_root ON obs_spans(name, parent_id, start_ns);
"""
RETENTION_DAYS = 7
MAX_SPANS = 50000


class _BatchExporter:
    def __init__(self, max_queue: int = 10000, interval: float = 0.5, batch: int = 200):
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.interval, self.batch = interval, batch
        self.dropped = 0
        self.exported = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()

    def export(self, s: Span) -> None:
        try:
            self.q.put_nowait(s)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        stop = False
        while not stop:
            items, sentinels = [], 0
            try:
                items.append(self.q.get(timeout=self.interval))
                while len(items) < self.batch:
                    items.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if None in items:  # close(): 남은 것을 저장하고 끝냄
                sentinels = items.count(None)
                items = [s for s in items if s is not None]
                stop = True
            for _ in range(sentinels):
                self.q.task_done()
            if items:
                try:
                    self.write(items)
                    self.exported += len(items)
                except Exception as e:  # noqa: BLE001
                    log.warning("span 저장 실패 (%d개 버림): %s", len(items), e)
                finally:
                    for _ in items:
                        self.q.task_done()

    def write(self, spans: list[Span]) -> None:
        raise NotImplementedError

    def flush(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        self.flush(2)
        self._stop.set()
        try:
            self.q.put_nowait(None)  # 기다리는 스레드를 바로 깨움
        except queue.Full:
            pass
        self._thread.join(2)


class SQLiteSpanExporter(_BatchExporter):
    def __init__(self, db_path: Path, **kw: Any):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._writes = 0
        super().__init__(**kw)
        self.prune()

    def write(self, spans: list[Span]) -> None:
        rows = [(s.span_id, s.trace_id, s.parent_id, s.name, s.service, s.start_ns, s.duration_ms, s.status,
                 s.status_msg, json.dumps(s.attrs, ensure_ascii=False, default=str),
                 json.dumps(s.events, ensure_ascii=False, default=str)) for s in spans]
        self.conn.executemany("INSERT OR REPLACE INTO obs_spans VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        self._writes += 1
        if self._writes % 50 == 0:
            self.prune()

    def prune(self) -> None:
        cutoff = time.time_ns() - RETENTION_DAYS * 86400 * 10**9
        self.conn.execute("DELETE FROM obs_spans WHERE start_ns < ?", (cutoff,))
        n = self.conn.execute("SELECT COUNT(*) FROM obs_spans").fetchone()[0]
        if n > MAX_SPANS:
            self.conn.execute("DELETE FROM obs_spans WHERE span_id IN (SELECT span_id FROM obs_spans "
                              "ORDER BY start_ns LIMIT ?)", (n - MAX_SPANS,))
        self.conn.commit()

    def close(self) -> None:
        super().close()
        try:
            self.conn.close()
        except sqlite3.Error:
            pass


def _otlp_value(v: Any) -> dict:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def otlp_json(spans: list[Span]) -> dict:
    """OTLP/HTTP JSON (opentelemetry-proto ExportTraceServiceRequest)."""
    by_service: dict[str, list[Span]] = {}
    for s in spans:
        by_service.setdefault(s.service, []).append(s)
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": f"real-estate-agent.{svc}"}}]},
        "scopeSpans": [{"scope": {"name": "real_estate_agent", "version": "1.1"}, "spans": [{
            "traceId": s.trace_id, "spanId": s.span_id, **({"parentSpanId": s.parent_id} if s.parent_id else {}),
            "name": s.name, "kind": 1, "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns),
            "attributes": [{"key": k, "value": _otlp_value(v)} for k, v in s.attrs.items()],
            "events": [{"name": e["name"], "timeUnixNano": str(e["ts_ns"]),
                        "attributes": [{"key": k, "value": _otlp_value(v)} for k, v in e["attrs"].items()]}
                       for e in s.events],
            "status": {"code": 2, "message": s.status_msg} if s.status == "error" else {"code": 1},
        } for s in items]}],
    } for svc, items in by_service.items()]}


class OTLPHttpExporter(_BatchExporter):
    def __init__(self, endpoint: str, headers: Optional[dict[str, str]] = None, **kw: Any):
        import httpx

        self.url = endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/v1/traces") else "/v1/traces")
        self.http = httpx.Client(timeout=5, headers={"Content-Type": "application/json", **(headers or {})})
        self.failures = 0
        super().__init__(**kw)

    def write(self, spans: list[Span]) -> None:
        r = self.http.post(self.url, content=json.dumps(otlp_json(spans), ensure_ascii=False).encode())
        if r.status_code >= 300:
            self.failures += 1
            raise RuntimeError(f"OTLP 수집기 응답 HTTP {r.status_code}")
