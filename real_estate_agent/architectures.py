"""v0.4 Architecture Study: 같은 도구로 '에이전트 구조'만 바꿔 비교하기 위한 구현들.

    ① Workflow (고정 파이프라인)   질문 → 검색 → (비교/건축 확인) → 정해진 형식의 답.  판단하지 않고, 대화 상태도 안 씀
    ② ReAct (우리 기본 = ToolAgent) 판단 → 도구 → 결과 보고 다시 판단 … → 답.         결과에 따라 다음 행동이 바뀜
    ③ Plan-and-Execute             처음에 계획(도구 순서)을 한 번 세우고 그대로 실행 → 답. 중간에 계획을 고치지 않음
    ④ Multi-Agent (v0.7)           총괄이 검색·분석 Agent 에게 나눠 맡김
    ⑤ LangGraph (v0.5)             ②와 같은 판단을 프레임워크 그래프로

모두 같은 도구·같은 실행기(ToolAgent._execute)·같은 안전장치를 쓰고 '흐름 제어'만 다릅니다.
판단 횟수(decisions) = AI 모드라면 AI 를 부르는 횟수 → 구조별 비용·지연의 차이.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .llm import LLMUnavailable, ToolTurn
from .tool_agent.loop import (RulePolicy, Step, ToolAgent, _Run, _asks_buildable, _call, _loads, _named_listing,
                              _purpose_in, _sort_in, _wanted_count, _wants_compare, rule_answer)


def _observations(run: _Run) -> list[tuple[str, dict]]:
    start = max(i for i, m in enumerate(run.messages) if m["role"] == "user")
    return [(m["name"], _loads(m["content"])) for m in run.messages[start:] if m["role"] == "tool"]


def _found(run: _Run) -> list[dict]:
    for name, obs in reversed(_observations(run)):
        if name == "search_listings" and obs.get("ok"):
            return obs["result"].get("매물", [])
    return []


def _finish(agent: ToolAgent, run: _Run, n: int, answer: str, policy: str, label: str) -> None:
    run.answer, run.policy, run.stopped = answer, policy, "answer"
    agent._add(run.steps, Step(n, "answer", policy, summary=label))
    run.messages.append({"role": "assistant", "content": answer})


class WorkflowAgent(ToolAgent):
    """① 고정 파이프라인. 판단(AI) 없이 정해진 순서로만 도구를 부름. '그 중에', '다른 지역', '정말 없어?' 를 모름."""

    mode_label = "workflow"
    uses_policy = False

    def _run_steps(self, run: _Run) -> None:
        q = run.question
        args: dict[str, Any] = {"query": q, "limit": _wanted_count(q) or (3 if _wants_compare(q) else 5)}
        if _sort_in(q):
            args["sort"] = _sort_in(q)
        run.policy = "workflow"
        self._execute(run, 1, _call("search_listings", args))
        found, n = _found(run), 2
        if _wants_compare(q) and len(found) >= 2:
            k = min(len(found), _wanted_count(q) or 2, 5)
            self._execute(run, n, _call("compare_listings", {"listing_ids": [x["id"] for x in found[:k]]}))
            n += 1
        purpose = _purpose_in(q)
        if purpose and _asks_buildable(q) and found:
            self._execute(run, n, _call("check_land_use", {"listing_id": found[0]["id"], "purpose": purpose}))
            n += 1
        _finish(self, run, n, rule_answer(q, _observations(run)), "workflow", "정해진 형식으로 답")


PLANNER_PROMPT = """토지 매물 상담의 '계획 세우기' 담당입니다. 손님 질문에 답하려면 어떤 도구를 어떤 순서로 쓸지 한 번에 정하세요.
도구: search_listings(query, regions, purpose, budget_max_manwon, sort, limit, use_previous_conditions, exclude_regions, other_regions, recheck),
      compare_listings(listing_ids), check_land_use(listing_id, purpose), get_listing(listing_id).
검색 결과의 매물 id 는 아직 모르므로 "$search[:2]" (앞 2개) 또는 "$search[0]" (첫 번째) 처럼 적으세요.
JSON 으로만 답하세요: {"plan": [{"tool": "...", "args": {...}}, ...]}"""


class PlanExecuteAgent(ToolAgent):
    """③ 계획 후 실행. 계획 1번(판단) → 그대로 실행 → 답 1번(판단). 결과가 예상과 달라도 계획을 고치지 않음."""

    mode_label = "plan_execute"

    def _plan(self, run: _Run) -> list[dict]:
        run.decisions += 1
        if run.use_llm:
            try:
                raw = self.llm_policy.llm.complete_json(PLANNER_PROMPT, run.question, 800)
                plan = [s for s in (raw or {}).get("plan", [])
                        if isinstance(s, dict) and s.get("tool") in self.registry.tools and isinstance(s.get("args"), dict)]
                if plan:
                    run.policy = getattr(self.llm_policy.llm, "last_used", "llm")
                    return plan[:5]
            except LLMUnavailable:
                pass
        # 규칙 계획: 첫 행동(검색, 이어서 묻기 포함)은 ReAct 와 같은 규칙으로 정하고, 나머지는 질문만 보고 미리 정함
        q = run.question
        first = RulePolicy(self.ctx, self.agent).next(run.messages, run.tools)
        plan = [{"tool": c.name, "args": c.arguments} for c in first.tool_calls]
        if plan and plan[0]["tool"] != "search_listings":
            return plan  # 매물번호를 말한 건축 확인 등
        if _wants_compare(q):
            plan.append({"tool": "compare_listings", "args": {"listing_ids": f"$search[:{_wanted_count(q) or 2}]"}})
        purpose = _purpose_in(q)
        if purpose and _asks_buildable(q):
            plan.append({"tool": "check_land_use",
                         "args": {"listing_id": _named_listing(self.agent, q) or "$search[0]", "purpose": purpose}})
        return plan

    @staticmethod
    def _fill(args: dict, found: list[dict]) -> dict:
        ids = [x["id"] for x in found]
        out = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("$search["):
                inner = v[len("$search["):-1]
                if inner.startswith(":"):
                    out[k] = ids[: int(inner[1:] or 2)]
                else:
                    out[k] = ids[int(inner)] if len(ids) > int(inner) else ""
            else:
                out[k] = v
        return out

    def _run_steps(self, run: _Run) -> None:
        t0 = time.perf_counter()
        plan = self._plan(run)
        self._add(run.steps, Step(0, "plan", run.policy, summary="계획: " + " → ".join(s["tool"] for s in plan) if plan
                                  else "계획: 도구 없이 답", ms=round((time.perf_counter() - t0) * 1000, 1)))
        for n, s in enumerate(plan, start=1):  # 그대로 실행 (중간 결과로 계획을 바꾸지 않음)
            self._execute(run, n, ToolTurn(tool_calls=_call(s["tool"], self._fill(s["args"], _found(run))).tool_calls,
                                           provider=run.policy))
        run.decisions += 1
        answer, policy = "", "rule"
        if run.use_llm:
            try:
                turn = self._llm_call(self.llm_policy.llm.tool_step, self.deadline,
                                      self.llm_policy.prompt + run.extra + "\n이제 도구 없이 지금까지 결과로 답하세요.",
                                      list(run.messages), [])
                if not turn.tool_calls and turn.text.strip():
                    answer, policy = turn.text.strip(), turn.provider or "llm"
            except LLMUnavailable:
                pass
        _finish(self, run, len(plan) + 1, answer or rule_answer(run.question, _observations(run)), policy, "계획대로 실행한 결과로 답")


ARCHITECTURES = {
    "workflow": ("① Workflow (고정 파이프라인)", WorkflowAgent),
    "single": ("② ReAct (기본)", ToolAgent),
    "plan_execute": ("③ Plan-and-Execute", PlanExecuteAgent),
    "multi": ("④ Multi-Agent (총괄→검색·분석)", None),
    "langgraph": ("⑤ ReAct on LangGraph (v0.5)", None),
}
