"""v0.5 Framework: 같은 AI 상담 Agent 를 LangGraph(에이전트 프레임워크)로 다시 만든 판.

    v0.2 직접 구현:   for 단계 in 1..6:  판단 → 도구 실행 → (답이면 끝)          ← 반복문을 손으로 씀
    v0.5 LangGraph:   START → [판단·실행] ─(계속)→ [판단·실행] … ─(답)→ END
                                        └─(단계 다 씀)→ [정리] → END        ← 흐름을 '그래프'로 선언

공정한 비교를 위해 도구·판단 규칙(정책)·안전장치(v1.0)·추적(v1.1)은 똑같이 쓰고, '흐름 제어'만 프레임워크에 맡깁니다.
프레임워크가 덤으로 주는 것: 대화 상태 저장(checkpointer, 대화마다 thread_id), 저장된 상태 다시 보기(get_state_history),
그래프 그림(mermaid), 단계별 스트리밍.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, TypedDict

from ..tool_agent.loop import ToolAgent, _Run
from ..tool_agent.memory import dump_messages, load_messages


class GraphState(TypedDict):
    messages: list[dict]   # 대화 기록 (JSON 으로 저장 가능한 모양)
    n: int                 # 몇 번째 판단인지
    status: str            # continue / done


def available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


class LangGraphAgent(ToolAgent):
    """ToolAgent 와 같은 사용법 (run / resume / 결과 모양). 흐름만 LangGraph 그래프로."""

    mode_label = "langgraph"

    def __init__(self, *args: Any, thread_id: Optional[str] = None, checkpointer=None, **kwargs: Any):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        super().__init__(*args, **kwargs)
        self.mode_label = "langgraph"
        self.thread_id = thread_id or uuid.uuid4().hex
        self._active: Optional[_Run] = None
        g = StateGraph(GraphState)
        g.add_node("decide_and_act", self._node_step)   # 판단 1번 + 고른 도구 실행 (직접 구현과 같은 _step)
        g.add_node("wrap_up", self._node_wrap_up)        # 단계 수를 다 쓰면 정리해서 답
        g.add_edge(START, "decide_and_act")
        g.add_conditional_edges("decide_and_act", self._route,
                                {"continue": "decide_and_act", "wrap_up": "wrap_up", "end": END})
        g.add_edge("wrap_up", END)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = g.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ 그래프 노드
    def _node_step(self, state: GraphState) -> dict:
        run = self._active
        run.messages[:] = load_messages(state["messages"])
        status = self._step(run, state["n"])
        return {"messages": dump_messages(run.messages), "n": state["n"] + 1, "status": status}

    def _node_wrap_up(self, state: GraphState) -> dict:
        run = self._active
        run.messages[:] = load_messages(state["messages"])
        self._wrap_up(run)
        return {"messages": dump_messages(run.messages), "status": "done"}

    def _route(self, state: GraphState) -> str:
        if state["status"] == "done":
            return "end"
        return "wrap_up" if state["n"] > self.max_steps else "continue"

    # ------------------------------------------------------------------ ToolAgent 의 반복문 대신
    def _run_steps(self, run: _Run) -> None:
        self._active = run
        try:
            self.graph.invoke({"messages": dump_messages(run.messages), "n": 1, "status": "continue"},
                              {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 4 * self.max_steps + 10})
        finally:
            self._active = None

    # ------------------------------------------------------------------ 프레임워크가 주는 것
    def saved_states(self) -> list[dict]:
        """이 대화에서 LangGraph 가 저장한 상태들 (최근 것부터) - 어느 단계에서 무엇을 했는지 되짚기."""
        out = []
        for snap in self.graph.get_state_history({"configurable": {"thread_id": self.thread_id}}):
            out.append({"n": snap.values.get("n"), "status": snap.values.get("status"),
                        "next": list(snap.next), "messages": len(snap.values.get("messages") or [])})
        return out

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
