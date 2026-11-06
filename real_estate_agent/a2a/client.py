"""A2A 클라이언트: 다른 Agent(다른 프로세스·다른 컴퓨터)에게 일을 보내고 Task 로 결과를 받음.

신뢰성 장치
- 시간 제한(timeout): 상대가 멈춰도 상담 전체가 멈추지 않게
- 재시도: 연결 실패·서버 오류(5xx)는 잠깐 쉬었다가 한 번 더 (같은 messageId 로 보내므로 상대가 두 번 실행하지 않음)
- 서킷 브레이커: 한 번 닿지 않은 Agent 는 잠깐(30초) 바로 '닿지 않음' 처리 -> 손님마다 시간 초과를 기다리지 않음
- 요청 오류(JSON-RPC 오류)는 다시 보내도 같으므로 재시도하지 않음
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterator, Optional

from . import protocol as p

BREAKER_SECONDS = 30.0


class A2AClient:
    _open_until: dict[str, float] = {}   # 주소 -> 이 시각까지 연결 시도 안 함 (모든 클라이언트가 함께 씀)
    _mu = threading.Lock()

    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 30.0, retries: int = 1,
                 http=None, breaker_seconds: float = BREAKER_SECONDS):
        """http: httpx.Client 호환 객체 (테스트에서는 FastAPI TestClient)."""
        import httpx

        self.url = url if url.endswith("/") else url + "/"
        self.token, self.timeout, self.retries, self.breaker_seconds = token, timeout, retries, breaker_seconds
        self.http = http or httpx.Client(timeout=timeout)
        self.last_ms = 0.0

    # ------------------------------------------------------------------ 연결 상태
    def available(self) -> bool:
        with self._mu:
            return self._open_until.get(self.url, 0) <= time.monotonic()

    def _trip(self) -> None:
        with self._mu:
            self._open_until[self.url] = time.monotonic() + self.breaker_seconds

    def _reset(self) -> None:
        with self._mu:
            self._open_until.pop(self.url, None)

    @classmethod
    def circuit_open(cls, url: str) -> bool:
        """이 주소가 최근 연결 실패로 '쉬는 중'인지 (상태 점검용, 연결은 하지 않음)."""
        url = url if url.endswith("/") else url + "/"
        with cls._mu:
            return cls._open_until.get(url, 0) > time.monotonic()

    @classmethod
    def reset_all(cls) -> None:
        with cls._mu:
            cls._open_until.clear()

    def _headers(self) -> dict[str, str]:
        from ..observability.tracing import traceparent

        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        tp = traceparent()
        if tp:
            h["traceparent"] = tp  # W3C Trace Context: 상대 서버가 같은 trace 로 이어서 기록
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # ------------------------------------------------------------------ 요청
    def card(self) -> dict[str, Any]:
        """Agent Card (상대가 누구고 무엇을 할 수 있는지)."""
        import httpx

        last: Optional[Exception] = None
        for path in (p.CARD_PATH, p.OLD_CARD_PATH):
            try:
                r = self.http.get(self.url.rstrip("/") + path)
            except httpx.HTTPError as e:
                last = e
                break
            if r.status_code == 200:
                return r.json()
        raise p.A2AUnavailable(f"Agent Card 를 받지 못했습니다: {self.url} ({last or 'not found'})")

    def call(self, method: str, params: dict[str, Any]) -> Any:
        """v1.1: 원격 호출 하나 = span 하나 (traceparent 헤더로 상대 서버의 span 이 이 아래에 이어짐) + 지표."""
        from ..observability.metrics import A2A_CALLS, A2A_LATENCY
        from ..observability.tracing import TRACER

        started = time.perf_counter()
        outcome = "error"
        with TRACER.span("a2a.client", method=method, url=self.url) as sp:
            try:
                result = self._call(method, params)
                outcome = "ok"
                if isinstance(result, dict):
                    sp.set(task_id=result.get("id"), state=(result.get("status") or {}).get("state"))
                return result
            except p.A2AUnavailable as e:
                outcome = "unavailable"
                sp.error(str(e))
                raise
            except p.A2AError as e:
                outcome = "rejected"
                sp.error(e.message)
                raise
            finally:
                A2A_CALLS.inc(url=self.url, method=method, outcome=outcome)
                A2A_LATENCY.observe(time.perf_counter() - started, url=self.url)

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        import httpx

        if not self.available():
            raise p.A2AUnavailable("최근 연결 실패로 잠시 쉬는 중 (서킷 브레이커)")
        payload = {"jsonrpc": "2.0", "id": p.new_id(), "method": method, "params": params}
        last: Any = None
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(0.2 * 2 ** (attempt - 1))
            try:
                r = self.http.post(self.url, content=json.dumps(payload, ensure_ascii=False).encode(),
                                   headers=self._headers())
            except httpx.TimeoutException:
                last = f"시간 초과 ({self.timeout:.0f}초)"
                break  # 오래 걸리는 상대에게 또 보내면 손님이 두 배로 기다림
            except httpx.ConnectError as e:
                # v1.1 관측으로 찾은 개선: 연결 거부 = 서버가 꺼져 있음 → 0.2초 쉬고 다시 보내 봐야 소용없음
                last = f"{type(e).__name__}: {e}"
                break
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 401:
                raise p.A2AError(401, "인증 실패 (REA_A2A_TOKEN 을 확인하세요)")
            if r.status_code >= 500:
                last = f"서버 오류 HTTP {r.status_code}"
                continue
            try:
                body = r.json()
            except ValueError:
                last = "JSON 이 아닌 응답"
                continue
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
            self._reset()
            if "error" in body:
                e = body["error"]
                raise p.A2AError(e.get("code", p.INTERNAL_ERROR), e.get("message", ""), e.get("data"))
            return body.get("result")
        self._trip()
        raise p.A2AUnavailable(str(last))

    def send(self, text: str, data: Optional[dict] = None, context_id: Optional[str] = None,
             metadata: Optional[dict] = None) -> dict[str, Any]:
        """message/send -> 끝난 Task."""
        parts = [p.text_part(text)] + ([p.data_part(data)] if data else [])
        result = self.call("message/send", {"message": p.message("user", parts, context_id, metadata=metadata)})
        if not isinstance(result, dict):
            raise p.A2AError(p.INTERNAL_ERROR, "Task 가 아닌 응답")
        return result

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.call("tasks/get", {"id": task_id})

    def stream(self, text: str, data: Optional[dict] = None, context_id: Optional[str] = None) -> Iterator[dict]:
        """message/stream -> Task, status-update, artifact-update … 를 차례로."""
        import httpx

        parts = [p.text_part(text)] + ([p.data_part(data)] if data else [])
        payload = {"jsonrpc": "2.0", "id": p.new_id(), "method": "message/stream",
                   "params": {"message": p.message("user", parts, context_id)}}
        try:
            with self.http.stream("POST", self.url, content=json.dumps(payload, ensure_ascii=False).encode(),
                                  headers=self._headers()) as r:
                if r.status_code == 401:
                    raise p.A2AError(401, "인증 실패")
                if r.headers.get("content-type", "").startswith("application/json"):
                    body = json.loads(r.read())
                    e = body.get("error") or {}
                    raise p.A2AError(e.get("code", p.INTERNAL_ERROR), e.get("message", "오류"))
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        body = json.loads(line[6:])
                        if "error" in body:
                            raise p.A2AError(body["error"]["code"], body["error"]["message"])
                        yield body["result"]
        except httpx.HTTPError as e:
            self._trip()
            raise p.A2AUnavailable(f"{type(e).__name__}: {e}") from e
