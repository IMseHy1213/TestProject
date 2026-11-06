"""v1.1 지표(metrics): 횟수(counter)와 걸린 시간 분포(histogram)를 메모리에 모아 Prometheus 형식으로 공개 (/metrics).

예) rea_agent_requests_total{mode="single",policy="openai",stopped="answer"} 12
    rea_llm_tokens_total{provider="openai",type="input"} 4810
    rea_tool_latency_seconds_bucket{tool="search_listings",le="0.25"} 9
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Iterable, Optional

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def _key(labels: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _fmt_labels(key: tuple, extra: Optional[tuple] = None) -> str:
    items = list(key) + ([extra] if extra else [])
    if not items:
        return ""
    esc = lambda v: v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in items) + "}"


def _num(v: float) -> str:
    if v == math.inf:
        return "+Inf"
    return str(int(v)) if float(v).is_integer() else repr(round(v, 6))


class Counter:
    kind = "counter"

    def __init__(self, name: str, help_: str):
        self.name, self.help = name, help_
        self.values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, n: float = 1, **labels: Any) -> None:
        k = _key(labels)
        with self._lock:
            self.values[k] = self.values.get(k, 0.0) + n

    def get(self, **labels: Any) -> float:
        return self.values.get(_key(labels), 0.0)

    def total(self, **match: Any) -> float:
        want = {(k, str(v)) for k, v in match.items()}
        with self._lock:
            return sum(v for k, v in self.values.items() if want <= set(k))

    def lines(self) -> Iterable[str]:
        with self._lock:
            for k, v in sorted(self.values.items()):
                yield f"{self.name}{_fmt_labels(k)} {_num(v)}"


class Histogram:
    kind = "histogram"

    def __init__(self, name: str, help_: str, buckets: tuple = LATENCY_BUCKETS):
        self.name, self.help, self.buckets = name, help_, buckets
        self.values: dict[tuple, list] = {}   # key -> [bucket counts..., sum, count]
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: Any) -> None:
        k = _key(labels)
        with self._lock:
            row = self.values.setdefault(k, [0] * len(self.buckets) + [0.0, 0])
            for i, b in enumerate(self.buckets):
                if value <= b:
                    row[i] += 1
            row[-2] += value
            row[-1] += 1

    def quantile(self, q: float, **match: Any) -> Optional[float]:
        """버킷으로 어림한 분위수 (대시보드용)."""
        want = {(k, str(v)) for k, v in match.items()}
        with self._lock:
            rows = [r for k, r in self.values.items() if want <= set(k)]
        if not rows:
            return None
        counts = [sum(r[i] for r in rows) for i in range(len(self.buckets))]
        total = sum(r[-1] for r in rows)
        if not total:
            return None
        target = q * total
        for b, c in zip(self.buckets, counts):
            if c >= target:
                return b
        return math.inf

    def lines(self) -> Iterable[str]:
        with self._lock:
            for k, row in sorted(self.values.items()):
                for b, c in zip(self.buckets, row):
                    yield f"{self.name}_bucket{_fmt_labels(k, ('le', _num(b)))} {c}"
                yield f"{self.name}_bucket{_fmt_labels(k, ('le', '+Inf'))} {row[-1]}"
                yield f"{self.name}_sum{_fmt_labels(k)} {_num(row[-2])}"
                yield f"{self.name}_count{_fmt_labels(k)} {row[-1]}"


class Registry:
    def __init__(self):
        self.metrics: dict[str, Any] = {}
        self.gauges: dict[str, tuple[str, Callable[[], float]]] = {}

    def counter(self, name: str, help_: str) -> Counter:
        return self.metrics.setdefault(name, Counter(name, help_))

    def histogram(self, name: str, help_: str, buckets: tuple = LATENCY_BUCKETS) -> Histogram:
        return self.metrics.setdefault(name, Histogram(name, help_, buckets))

    def gauge(self, name: str, help_: str, fn: Callable[[], float]) -> None:
        """값을 읽을 때 계산하는 지표 (매물 수, 진행 중인 상담 수 …)."""
        self.gauges[name] = (help_, fn)

    def prometheus(self) -> str:
        out = []
        for m in self.metrics.values():
            out += [f"# HELP {m.name} {m.help}", f"# TYPE {m.name} {m.kind}", *m.lines()]
        for name, (help_, fn) in self.gauges.items():
            try:
                v = float(fn())
            except Exception:  # noqa: BLE001
                continue
            out += [f"# HELP {name} {help_}", f"# TYPE {name} gauge", f"{name} {_num(v)}"]
        return "\n".join(out) + "\n"

    def reset(self) -> None:
        for m in self.metrics.values():
            with m._lock:
                m.values.clear()


METRICS = Registry()
AGENT_REQUESTS = METRICS.counter("rea_agent_requests_total", "AI 상담 질문 수 (방식·답을 만든 쪽·끝난 이유별)")
AGENT_LATENCY = METRICS.histogram("rea_agent_latency_seconds", "질문 하나를 처리한 시간")
LLM_CALLS = METRICS.counter("rea_llm_calls_total", "AI(GPT/Claude) 호출 수 (성공/실패 이유별)")
LLM_LATENCY = METRICS.histogram("rea_llm_latency_seconds", "AI 호출 하나의 응답 시간")
LLM_TOKENS = METRICS.counter("rea_llm_tokens_total", "AI 가 읽고(input) 쓴(output) 토큰 수")
LLM_COST = METRICS.counter("rea_llm_cost_usd_total", "AI 사용 추정 비용 (달러, REA_LLM_PRICES 설정 시)")
TOOL_CALLS = METRICS.counter("rea_tool_calls_total", "도구 호출 수 (성공 여부별)")
TOOL_LATENCY = METRICS.histogram("rea_tool_latency_seconds", "도구 하나의 실행 시간")
RELIABILITY = METRICS.counter("rea_reliability_events_total", "안전장치 작동 수 (v1.0: 답변 재작성·규칙 대체·시간 예산 …)")
A2A_CALLS = METRICS.counter("rea_a2a_requests_total", "A2A 원격 Agent 호출 수 (결과별)")
A2A_LATENCY = METRICS.histogram("rea_a2a_latency_seconds", "A2A 원격 호출 왕복 시간")
HTTP_REQUESTS = METRICS.counter("rea_http_requests_total", "웹·API 요청 수 (화면·상태 코드별)")
HTTP_LATENCY = METRICS.histogram("rea_http_latency_seconds", "웹·API 요청 처리 시간")
