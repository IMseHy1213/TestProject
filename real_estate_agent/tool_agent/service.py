"""AI 상담 서비스: 대화 열기 → 질문 처리(체크포인트·장기 기억·실시간 진행 전달) → 저장.

웹 화면(Flask)과 JSON API(FastAPI)가 같은 코드를 씁니다.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import threading
from datetime import datetime
from typing import Any, Callable, Iterator, Optional

from .loop import ToolAgent, step_dict
from .memory import ConversationStore, LongTermMemory, is_incomplete
from .tools import AgentContext

MAX_TURNS = 20  # 화면에 남길 질문/답변 수
MODES = ("single", "multi", "distributed")  # 단일 / v0.7 멀티 Agent / v0.8 분산(A2A) 멀티 Agent
_CID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class Busy(Exception):
    """같은 대화에서 앞 질문을 아직 처리 중."""


class AgentService:
    def __init__(self, agent, lock: Optional[threading.RLock] = None):
        self.agent = agent
        self.lock = lock or threading.RLock()
        self.store = ConversationStore(agent.storage.conn, self.lock)
        self.memory = LongTermMemory(agent.storage.conn, self.lock)
        self.store.prune()

    # ------------------------------------------------------------------ 대화
    def open(self, owner: str, cid: Optional[str] = None, reuse_recent: bool = True) -> dict:
        """대화 불러오기. 없으면(또는 남의 대화면) 이 방문자의 최근 대화, 그것도 없으면 새 대화."""
        loaded = self.store.load(cid)
        if loaded is not None and loaded["owner"] == owner:
            return loaded
        conv = self.store.latest_for(owner) if reuse_recent else None
        if conv is None:
            # 아직 저장 전인 새 대화면 그 id 를 그대로 (남의 대화 id 면 새 id)
            new_cid = cid if cid and loaded is None and _CID.match(cid) else secrets.token_urlsafe(16)
            conv = {"cid": new_cid, "owner": owner, "history": [], "ctx": AgentContext(),
                    "turns": [], "status": "idle", "pending": None, "updated": None}
        return conv

    def needs_resume(self, conv: dict) -> bool:
        """답을 만들다 끊긴 질문이 있는지 (서버가 꺼졌거나 오류로 중단)."""
        return bool(conv.get("pending")) and (conv["status"] == "interrupted" or ConversationStore.is_stale(conv))

    def ask(self, owner: str, cid: Optional[str], question: str,
            on_event: Optional[Callable[[dict], None]] = None, mode: str = "single") -> tuple[str, dict]:
        """질문 하나 처리 -> (대화 id, 이번 질문/답변 기록). mode: 'single' / 'multi'."""
        conv = self._begin(owner, cid, question)
        runner = self._runner(conv, question, on_event, mode)
        history = conv["history"]
        if runner.__class__.__name__ == "ToolAgent" and any(
                c.name.startswith("delegate_") for m in history for c in (m.get("tool_calls") or [])):
            from .multi import text_history

            history = text_history(history)  # 멀티 → 단일로 바꾼 경우: 총괄 Agent 의 도구 기록은 빼고 대화만
        try:
            result, history = runner.run(question, history)
        except Exception:
            self._interrupted(conv, question, runner)
            raise
        return conv["cid"], self._finish(conv, result, history, getattr(runner, "mode", "single"))

    def resume(self, owner: str, cid: str, on_event: Optional[Callable[[dict], None]] = None,
               mode: str = "single") -> tuple[str, dict]:
        """끊긴 질문을 마지막 체크포인트부터 이어서 처리."""
        conv = self.open(owner, cid, reuse_recent=False)
        question = conv.get("pending")
        if not question or not is_incomplete(conv["history"]):
            raise ValueError("이어서 처리할 질문이 없습니다.")
        conv = self._begin(owner, conv["cid"], question, resuming=True)
        if any(c.name.startswith("delegate_") for m in conv["history"] for c in (m.get("tool_calls") or [])):
            mode = "multi"  # 끊긴 질문은 시작한 방식 그대로 이어서
        runner = self._runner(conv, question, on_event, mode)
        try:
            result, history = runner.resume(conv["history"])
        except Exception:
            self._interrupted(conv, question, runner)
            raise
        return conv["cid"], self._finish(conv, result, history, getattr(runner, "mode", "single"))

    def reset(self, cid: Optional[str]) -> None:
        """새 상담 시작 (대화만 지움, 장기 기억은 유지)."""
        self.store.delete(cid)

    def forget(self, owner: str) -> None:
        """기억 지우기: 이 방문자의 장기 기억과 대화를 모두 삭제."""
        self.memory.forget(owner)
        self.store.delete_owner(owner)

    def memory_view(self, owner: str) -> dict:
        """화면 표시용 장기 기억 요약."""
        from .tools import display_name

        mem = self.memory.load(owner)
        listing = {l.listing_id: l for l in self.agent.listings()}
        name = lambda i: display_name(listing[i]) if i in listing else "(지금은 없는 매물)"  # noqa: E731
        return {
            "visits": mem.get("visits", 0),
            "searches": [s["조건"] for s in reversed(mem["searches"][-3:])],
            "liked": [name(i) for i in list(mem["liked"])[-5:]],
            "rejected": [name(i) for i in list(mem["rejected"])[-5:]],
            "empty": not (mem["searches"] or mem["liked"] or mem["rejected"]),
        }

    # ------------------------------------------------------------------ 스트리밍
    def stream(self, owner: str, cid: Optional[str], question: Optional[str] = None,
               resume: bool = False, mode: str = "single") -> Iterator[str]:
        """질문 처리 진행 상황을 Server-Sent Events 문자열로 차례차례 내보냄.
        (Agent 는 별도 스레드에서 돌고, 손님 화면 연결이 끊겨도 끝까지 처리·저장됨)"""
        events: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                if resume:
                    new_cid, turn = self.resume(owner, cid or "", on_event=events.put, mode=mode)
                else:
                    new_cid, turn = self.ask(owner, cid, question or "", on_event=events.put, mode=mode)
                events.put({"type": "saved", "cid": new_cid, "answer": turn["answer"]})
            except Busy as e:
                events.put({"type": "error", "message": str(e), "busy": True})
            except Exception as e:  # noqa: BLE001 - 화면에 오류를 알려 주고 끝냄
                events.put({"type": "error", "message": f"처리 중 오류: {type(e).__name__}"})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield sse(item)

    # ------------------------------------------------------------------ 내부
    def _begin(self, owner: str, cid: Optional[str], question: str, resuming: bool = False) -> dict:
        with self.lock:  # 같은 대화에 질문 두 개가 동시에 들어오지 않게
            conv = self.open(owner, cid, reuse_recent=False)
            if conv["status"] == "running" and not ConversationStore.is_stale(conv) and not resuming:
                raise Busy("앞 질문의 답을 만드는 중입니다. 잠시 후 다시 물어봐 주세요.")
            self.store.save(conv["cid"], owner, conv["history"], conv["ctx"], conv["turns"], "running", question)
        return conv

    def _runner(self, conv: dict, question: str, on_event, mode: str = "single"):
        def checkpoint(messages: list[dict], ctx: AgentContext) -> None:
            self.store.save(conv["cid"], conv["owner"], messages, ctx, conv["turns"], "running", question)

        if mode == "distributed":  # 검색·분석 Agent 가 다른 서버에 있음 (꺼져 있으면 여기서 대신 처리)
            from ..a2a.distributed import DistributedMultiAgent, context_id_for

            return DistributedMultiAgent(self.agent, ctx=conv["ctx"], lock=self.lock, memory=self.memory,
                                         owner=conv["owner"], checkpoint=checkpoint, on_event=on_event,
                                         context_id=context_id_for(conv["cid"]))
        if mode == "multi":
            from .multi import MultiAgent

            return MultiAgent(self.agent, ctx=conv["ctx"], lock=self.lock, memory=self.memory, owner=conv["owner"],
                              checkpoint=checkpoint, on_event=on_event)
        return ToolAgent(self.agent, ctx=conv["ctx"], lock=self.lock, memory=self.memory, owner=conv["owner"],
                         checkpoint=checkpoint, on_event=on_event)

    def _interrupted(self, conv: dict, question: str, runner) -> None:
        latest = self.store.load(conv["cid"])
        history = latest["history"] if latest else conv["history"]
        self.store.save(conv["cid"], conv["owner"], history, runner.ctx, conv["turns"], "interrupted", question)

    def _finish(self, conv: dict, result, history: list[dict], mode: str = "single") -> dict:
        turn = {"question": result.question, "answer": result.answer,
                "steps": [step_dict(s) for s in result.steps], "listing_ids": result.listing_ids,
                "badges": result.badges, "policy": result.policy, "total_ms": result.total_ms,
                "tool_calls": result.tool_calls, "tool_errors": result.tool_errors,
                "mode": mode,
                "at": datetime.now().isoformat(timespec="seconds")}
        conv["turns"] = (conv["turns"] + [turn])[-MAX_TURNS:]
        self.store.save(conv["cid"], conv["owner"], history, conv["ctx"], conv["turns"], "idle", None)
        return turn


def sse(data: dict[str, Any]) -> str:
    """Server-Sent Events 한 건."""
    return f"event: {data.get('type', 'message')}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
