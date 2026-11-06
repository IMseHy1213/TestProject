"""v0.8 분산 상담팀: 총괄 Agent 는 웹 서버에, 검색·분석 Agent 는 각자의 서버(프로세스)에서 A2A 로 협업.

    [웹 서버: 총괄 Agent] ──A2A message/send──▶ [검색 Agent 서버 :8101]
            │  (대화 상태를 메시지에 실어 보내고, 바뀐 상태를 돌려받음)
            └────────────A2A message/send──▶ [분석 Agent 서버 :8102]

- v0.7 멀티 Agent 와 같은 총괄 Agent·같은 결과 모양. 다른 점은 '맡기기'가 함수 호출이 아니라 네트워크 요청이라는 것.
- 원격 Agent 가 꺼져 있거나 응답이 없으면 이 서버 안의 같은 Agent 가 대신 처리 (상담이 끊기지 않음, trace 에 '전환' 표시).
- 손님의 장기 기억은 웹 서버에서만 읽고 씀 (원격 Agent 에는 이번 대화에 필요한 상태만 보냄).
- trace 에 원격 호출마다 왕복 시간 = 원격 처리 시간 + 통신 시간을 기록.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, fields
from typing import Any, Optional

from ..tool_agent.loop import AgentResult, Step
from ..tool_agent.multi import ANALYST, SEARCHER, SUPERVISOR, MultiAgent
from ..tool_agent.tools import ToolError
from . import protocol as p
from .client import A2AClient
from .state import id_maps, private_state, public_state, copy_into

SEARCH_URL = os.getenv("REA_A2A_SEARCH_URL", "http://127.0.0.1:8101/")
ANALYSIS_URL = os.getenv("REA_A2A_ANALYSIS_URL", "http://127.0.0.1:8102/")
REMOTE_TIMEOUT = float(os.getenv("REA_A2A_TIMEOUT", "60"))
_STEP_FIELDS = {f.name for f in fields(Step)}


def context_id_for(conversation_id: str) -> str:
    """원격 Agent 에게는 대화 id 대신 그것을 바꾼 값을 보냄 (대화 id 는 브라우저 세션 비밀)."""
    return hashlib.sha256(("a2a:" + conversation_id).encode()).hexdigest()[:24]


class DistributedMultiAgent(MultiAgent):
    mode = "distributed"

    def __init__(self, agent, search_url: Optional[str] = None, analysis_url: Optional[str] = None,
                 token: Optional[str] = None, clients: Optional[dict[str, A2AClient]] = None,
                 context_id: str = "", **kwargs: Any):
        super().__init__(agent, **kwargs)
        token = token if token is not None else os.getenv("REA_A2A_TOKEN") or None
        self.clients = clients or {
            SEARCHER: A2AClient(search_url or SEARCH_URL, token, timeout=REMOTE_TIMEOUT),
            ANALYST: A2AClient(analysis_url or ANALYSIS_URL, token, timeout=REMOTE_TIMEOUT),
        }
        self.memory, self.owner = kwargs.get("memory"), kwargs.get("owner", "")
        self.context_id = context_id or p.new_id()
        self.remote_searches: list[dict] = []
        self.remote_used = False
        self.hops: list[dict] = []  # 원격 호출 기록 (비교·발표용)

    # ------------------------------------------------------------------ 장기 기억은 여기(웹 서버)에서만
    def run(self, question: str, history: Optional[list[dict]] = None):
        self._before()
        result, messages = super().run(question, history)
        self._after()
        return result, messages

    def resume(self, history: list[dict]):
        self._before()
        result, messages = super().resume(history)
        self._after()
        return result, messages

    def _before(self) -> None:
        self.remote_searches, self.remote_used, self.hops = [], False, []
        if self.memory and self.owner:
            self.memory.apply_to(self.owner, self.ctx)

    def _after(self) -> None:
        if self.memory and self.owner and self.remote_used:
            self.memory.update_after(self.owner, self.ctx, self.remote_searches)

    # ------------------------------------------------------------------ 원격 맡기기
    def _note(self, kind: str, summary: str, ms: float = 0.0) -> None:
        step = Step(0, kind, "", summary=summary, ms=ms, agent=SUPERVISOR)
        self.trace.append(step)
        self._forward({"type": "step", "step": asdict(step)})

    def delegate(self, who: str, task: str, listing_ids: Optional[list[str]] = None) -> dict[str, Any]:
        task = self._target(task, listing_ids)
        client = self.clients.get(who)
        if client is None:
            return self._run_local(who, task)
        started = time.perf_counter()
        try:
            out = client.send(task, data={"state": public_state(self.agent, self.ctx), "history": self.history[-12:]},
                              context_id=self.context_id, metadata={"from": SUPERVISOR})
        except p.A2AUnavailable as e:
            self._note("fallback", f"원격 {who}({client.url}) 응답 없음 → 이 서버의 {who}가 대신 처리 ({e})")
            self.hops.append({"agent": who, "url": client.url, "ok": False, "error": str(e)})
            return self._run_local(who, task)
        except p.A2AError as e:
            raise ToolError(f"원격 {who} 가 요청을 거절했습니다: {e.message}") from e
        state = (out.get("status") or {}).get("state")
        if state != "completed":
            raise ToolError(f"원격 {who} 작업 실패 ({state}): {p.task_text(out)}")
        data = p.task_data(out)
        _, to_real = id_maps(self.agent)
        copy_into(self.ctx, private_state(self.agent, data.get("state") or {}))
        self.remote_used = True
        self.remote_searches += [s for s in data.get("run_searches") or [] if isinstance(s, dict) and s.get("조건")]
        steps = []
        for raw in data.get("steps") or []:
            s = Step(**{k: v for k, v in raw.items() if k in _STEP_FIELDS})
            s.agent = who
            steps.append(s)
            self.trace.append(s)
            self._forward({"type": "step", "step": asdict(s)})
        total = round((time.perf_counter() - started) * 1000, 1)
        remote_ms = float(data.get("total_ms") or 0)
        self.hops.append({"agent": who, "url": client.url, "ok": True, "task_id": out.get("id"), "ms": total,
                          "remote_ms": remote_ms, "network_ms": round(max(total - remote_ms, 0), 1)})
        self._note("remote", f"A2A {client.url} · task {str(out.get('id'))[:8]} {state} · 왕복 {total}ms "
                             f"(원격 처리 {remote_ms}ms + 통신 {max(total - remote_ms, 0):.1f}ms)", total)
        result = AgentResult(task, data.get("answer") or p.task_text(out), steps,
                             [to_real(i) for i in data.get("listing_ids") or []], data.get("policy", ""),
                             data.get("stopped", "answer"), remote_ms, str(out.get("id", "")),
                             {to_real(k): v for k, v in (data.get("badges") or {}).items()})
        observations = [(o[0], o[1]) for o in data.get("observations") or [] if isinstance(o, list) and len(o) == 2]
        return self._record(who, result, observations)
