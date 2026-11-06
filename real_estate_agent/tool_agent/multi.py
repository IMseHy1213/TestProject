"""v0.7 Multi-Agent: 총괄(supervisor) Agent 가 전문 Agent 에게 일을 나눠 맡김 ('agent-as-tool').

    손님 질문
       │
    [총괄 Agent]  도구 = delegate_search, delegate_analysis   (매물 도구는 직접 쓰지 않음)
       ├─▶ [검색 Agent]  search_listings · list_regions · recall_memory   ─ 매물 찾기·이어서 묻기·지난 상담
       └─▶ [분석 Agent]  get_listing · compare_listings · check_land_use · record_feedback ─ 비교·건축 판단
       │
    최종 답변 (총괄이 두 Agent 의 보고를 모아 정리)

- 세 Agent 는 같은 대화 상태(AgentContext: 직전 검색 조건, 보여준 매물, 관심/제외)를 함께 씀 (공유 칠판).
  그래서 '그 중에 넓은 건?', '다른 지역에도 있어?' 같은 이어서 묻기가 그대로 됨.
- 단계 기록(trace)에 어느 Agent 가 무엇을 했는지 표시 (Step.agent).
- GPT/Claude 가 있으면 총괄도 LLM 이 판단하고, 없으면 규칙으로 나눔 (검색 → 필요하면 분석).
- 결과 모양은 단일 Agent(ToolAgent)와 같음 (AgentResult) -> 웹 화면·API·저장을 그대로 씀.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from .loop import (SYSTEM_PROMPT, AgentResult, NOT_FOUND_MESSAGE, Step, ToolAgent, _asks_buildable, _call,
                   _loads, _purpose_in, _wants_compare, rule_answer)
from .tools import AgentContext, Tool, ToolError, brief
from ..llm import ToolTurn

SUPERVISOR = "총괄 Agent"
SEARCHER = "검색 Agent"
ANALYST = "분석 Agent"
SEARCH_TOOLS = {"search_listings", "list_regions", "recall_memory"}
ANALYSIS_TOOLS = {"get_listing", "compare_listings", "check_land_use", "record_feedback"}

SUPERVISOR_PROMPT = """당신은 충북 음성의 토지 전문 공인중개사무소 상담의 '총괄 Agent'입니다.
매물 도구를 직접 쓰지 않고, 전문 Agent 에게 일을 맡긴 뒤 보고를 모아 손님에게 답합니다.

- delegate_search(task): 검색 Agent. 매물 찾기, 이어서 묻기('그 중에', '다른 지역에도', '정말 없어?'), 지역 목록, 지난 상담 기억.
- delegate_analysis(task, listing_ids): 분석 Agent. 매물 상세, 비교, 건축 가능 여부, 마음에 든다/싫다 기록.
  listing_ids 에는 검색 Agent 가 돌려준 id 를 그대로 넣으세요.

순서
1. 매물이 필요하면 먼저 delegate_search. task 에는 손님 말의 뜻을 그대로 쓰세요 ('그 중에', '말고' 같은 표현을 지우지 마세요).
2. 비교·건축 판단·상세 설명이 필요할 때만 delegate_analysis 를 부르세요. 단순 검색이면 부르지 마세요.
3. 보고에 없는 매물·가격·면적을 지어내지 마세요. 찾은 매물이 없으면 "해당 조건에 맞는 매물이 없습니다."
최종 답변: 한국어 1~3문장 요약. 매물은 매물번호(없으면 표시이름)로 가리키고 핵심 수치만 괄호로.
id(F-… 같은 내부 번호)·표·마크다운은 쓰지 마세요. 건축 가능 여부는 토지이용계획 확인이 필요하다고 덧붙이세요.
"""

SEARCH_PROMPT = SYSTEM_PROMPT + """
[역할] 당신은 '검색 Agent'입니다. search_listings · list_regions · recall_memory 만 씁니다.
비교·건축 판단은 분석 Agent 가 하므로 하지 말고, 찾은 결과(개수·조건·매물번호)를 1~2문장으로 보고하세요."""

ANALYSIS_PROMPT = SYSTEM_PROMPT + """
[역할] 당신은 '분석 Agent'입니다. get_listing · compare_listings · check_land_use · record_feedback 만 씁니다.
대상 매물은 맡은 일에 적힌 id 입니다 (없으면 직전에 보여준 매물). 확인한 결과를 1~3문장으로 보고하세요."""


class MultiAgent:
    """ToolAgent 와 같은 run / resume / ctx 를 가진 멀티 Agent."""

    mode = "multi"

    def __init__(self, agent, llm=None, ctx: Optional[AgentContext] = None, lock=None, max_steps: int = 6,
                 log_dir=None, memory=None, owner: str = "",
                 checkpoint: Optional[Callable[[list, AgentContext], None]] = None,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.agent = agent
        self.ctx_obj = ctx or AgentContext()
        self.on_event = on_event
        self.trace: list[Step] = []
        self.done: list[tuple[str, dict]] = []      # 이번 질문에서 전문 Agent 들이 받은 도구 결과 (규칙 답변용)
        self.reports: dict[str, AgentResult] = {}   # 전문 Agent 별 마지막 결과
        self.history: list[dict] = []               # 전문 Agent 에게 넘길 지난 대화 (글자만)
        forward = self._forward
        common = dict(llm=llm, ctx=self.ctx_obj, log_dir="", trace=self.trace)
        self.searcher = ToolAgent(agent, lock=lock, memory=memory, owner=owner, system_prompt=SEARCH_PROMPT,
                                  only=SEARCH_TOOLS, name=SEARCHER, on_event=forward, max_steps=4, **common)
        self.analyst = ToolAgent(agent, lock=lock, system_prompt=ANALYSIS_PROMPT, only=ANALYSIS_TOOLS, name=ANALYST,
                                 on_event=forward, max_steps=4, **common)
        self.searcher.answer_label = self.analyst.answer_label = "총괄 Agent 에게 보고"
        self.searcher.final_answerer = self.analyst.final_answerer = False
        # 총괄의 도구(맡기기)는 잠금 없이 실행: 전문 Agent 가 매물 도구를 쓸 때만 잠금 (LLM 을 기다리는 동안 다른 손님을 막지 않게)
        self.supervisor = ToolAgent(agent, lock=None, tool_source=_DelegateTools(self), system_prompt=SUPERVISOR_PROMPT,
                                    name=SUPERVISOR, checkpoint=checkpoint, on_event=on_event,
                                    rule_policy=RuleSupervisor(self), max_steps=max_steps, **common)
        self.supervisor.mode_label = self.mode  # v1.1 지표: multi / distributed
        # v1.0: 총괄이 쓴 답은 전문 Agent 들이 받은 도구 결과로 검증하고, 실패하면 그 결과로 다시 작성
        self.supervisor.evidence = lambda: list(self.done)
        self.supervisor.fallback_answer = lambda q, done: (rule_answer(q, self.done) if self.done
                                                           else NOT_FOUND_MESSAGE)
        self.log_dir = log_dir if log_dir is not None else getattr(agent.settings, "log_dir", None)

    @property
    def ctx(self) -> AgentContext:
        return self.ctx_obj

    # ------------------------------------------------------------------ 실행
    def run(self, question: str, history: Optional[list[dict]] = None) -> tuple[AgentResult, list[dict]]:
        self._begin(history)
        if any(not c.name.startswith("delegate_") for m in history or [] for c in (m.get("tool_calls") or [])):
            history = text_history(history or [])  # 단일 → 멀티로 바꾼 경우: 다른 도구 기록은 빼고 대화만
        result, messages = self.supervisor.run(question, history)
        return self._finish(result), messages

    def resume(self, history: list[dict]) -> tuple[AgentResult, list[dict]]:
        self._begin(history)
        result, messages = self.supervisor.resume(history)
        return self._finish(result), messages

    def _begin(self, history: Optional[list[dict]]) -> None:
        self.trace.clear()
        self.done.clear()
        self.reports.clear()
        self.history = text_history(history or [])

    def _forward(self, event: dict) -> None:
        """전문 Agent 의 진행 상황을 화면으로 (각자의 '끝' 알림은 빼고 - 끝은 총괄 Agent 가 알림)."""
        if self.on_event and event.get("type") != "done":
            self.on_event(event)

    def delegate(self, who: str, task: str, listing_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """전문 Agent 하나에게 일을 맡기고 보고를 받음 (총괄 Agent 의 도구)."""
        return self._run_local(who, self._target(task, listing_ids))

    def _target(self, task: str, listing_ids: Optional[list[str]]) -> str:
        """총괄이 고른 대상 매물을 확인해 대화 상태에 표시하고, 맡길 일에 덧붙임."""
        if listing_ids:
            real = [self.supervisor.registry.resolve(i) for i in listing_ids]
            if not all(real):
                bad = [i for i, r in zip(listing_ids, real) if not r]
                raise ToolError(f"없는 매물 id: {', '.join(bad)}. 검색 Agent 가 돌려준 id 를 쓰세요.")
            self.ctx.last_shown = list(dict.fromkeys(real))  # 분석 대상 = 총괄이 고른 매물
            task = f"{task}\n(대상 매물 id: {', '.join(listing_ids)})"
        return task

    def _run_local(self, who: str, task: str) -> dict[str, Any]:
        worker = self.searcher if who == SEARCHER else self.analyst
        result, messages = worker.run(task, self.history)
        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        obs = [(m["name"], _loads(m["content"])) for m in messages[start:] if m["role"] == "tool"]
        return self._record(who, result, obs)

    def _record(self, who: str, result: AgentResult, obs: list[tuple[str, dict]]) -> dict[str, Any]:
        """전문 Agent 의 결과를 기록하고 총괄 Agent 에게 줄 보고로 정리 (v0.8 원격 Agent 도 같은 모양)."""
        self.done.extend(obs)
        self.reports[who] = result
        listings = [self.agent.find_listing(i) for i in result.listing_ids]
        search = next((o for n, o in reversed(obs) if n == "search_listings" and o.get("ok")), {})
        return {"agent": who, "보고": result.answer,
                "매물": [_short(brief(l)) for l in listings if l is not None],
                "조건에_맞는_매물_수": search.get("result", {}).get("조건에_맞는_매물_수"),
                "앞_조건_이어서": bool(search.get("result", {}).get("앞_조건_이어서")),
                "도구_호출": result.tool_calls}

    # ------------------------------------------------------------------ 결과 합치기
    def _finish(self, sup: AgentResult) -> AgentResult:
        search, analysis = self.reports.get(SEARCHER), self.reports.get(ANALYST)
        if sup.answer.strip() == NOT_FOUND_MESSAGE:
            ids: list[str] = []
        elif analysis and any(n == "compare_listings" and o.get("ok") for n, o in self.done):
            ids = analysis.listing_ids
        else:
            ids = list(dict.fromkeys((search.listing_ids if search else []) + (analysis.listing_ids if analysis else [])))
        badges: dict[str, list[str]] = {}
        for r in (search, analysis):
            for lid, values in (r.badges if r else {}).items():
                badges.setdefault(lid, [])
                badges[lid] += [v for v in values if v not in badges[lid]]
        steps, notes = [], set()
        for s in self.trace:  # 같은 '전환' 안내(AI 를 쓸 수 없어 규칙으로 …)는 한 번만
            if s.kind == "fallback" and s.n == 0:
                if s.summary in notes:
                    continue
                notes.add(s.summary)
            steps.append(s)
        result = AgentResult(sup.question, sup.answer, steps, ids[:6], sup.policy, sup.stopped,
                             sup.total_ms, sup.run_id, badges, trace_id=sup.trace_id,
                             span_id=sup.span_id)
        if self.log_dir:
            self.supervisor.log_dir = self.log_dir
            try:
                self.supervisor._log(result)
            finally:
                self.supervisor.log_dir = ""
        return result


class _DelegateTools:
    """총괄 Agent 의 도구 = 전문 Agent 에게 맡기기 (ToolRegistry 에 tool_source 로 넣음)."""

    def __init__(self, multi: MultiAgent):
        self.multi = multi

    def tools(self) -> list[Tool]:
        m = self.multi
        return [
            Tool("delegate_search", "검색 Agent 에게 매물 찾기를 맡김 (이어서 묻기·지역 목록·지난 상담 기억 포함). "
                 "결과: 보고, 찾은 매물(id·매물번호·가격·면적), 조건에 맞는 매물 수",
                 {"type": "object", "properties": {"task": {"type": "string", "description": "맡길 일 (손님 말의 뜻 그대로)"}},
                  "required": ["task"]},
                 lambda task: m.delegate(SEARCHER, task)),
            Tool("delegate_analysis", "분석 Agent 에게 매물 상세·비교·건축 가능 여부 확인·관심/제외 기록을 맡김",
                 {"type": "object", "properties": {
                     "task": {"type": "string", "description": "맡길 일 (예: '두 매물 비교', '창고 지을 수 있는지')"},
                     "listing_ids": {"type": "array", "items": {"type": "string"},
                                     "description": "대상 매물 id (검색 Agent 가 돌려준 id)"}},
                  "required": ["task"]},
                 lambda task, listing_ids=None: m.delegate(ANALYST, task, listing_ids)),
        ]


class RuleSupervisor:
    """LLM 없이 일 나누기: 검색 Agent → (비교·건축 판단이 필요하면) 분석 Agent → 결과를 모아 답."""

    name = "rule"

    def __init__(self, multi: MultiAgent):
        self.multi = multi

    def next(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        done = [(m["name"], _loads(m["content"])) for m in messages[start:] if m["role"] == "tool"]
        names = [n for n, _ in done]
        if "delegate_search" not in names:
            return _call("delegate_search", {"task": question})
        report = next(o for n, o in done if n == "delegate_search").get("result", {})
        found = report.get("매물", [])
        needs = ((_wants_compare(question) and len(found) >= 2)
                 or (_purpose_in(question) and _asks_buildable(question) and found))
        if needs and not report.get("앞_조건_이어서") and "delegate_analysis" not in names:
            return _call("delegate_analysis", {"task": question})  # 대상 = 검색 Agent 가 보여준 매물 (공유 상태)
        answer = rule_answer(question, self.multi.done) if self.multi.done else (report.get("보고") or "")
        return ToolTurn(text=answer, provider="rule")


def _short(b: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "매물번호", "표시이름", "주소", "가격", "면적_평", "지목", "용도지역")
    return {k: b[k] for k in keys if k in b}


def text_history(history: list[dict]) -> list[dict]:
    """전문 Agent 에게는 지난 대화의 글자(질문·답변)만 넘김 (도구 호출 기록은 총괄 Agent 것이라 빼고)."""
    return [{"role": m["role"], "content": m["content"]} for m in history
            if m["role"] == "user" or (m["role"] == "assistant" and not m.get("tool_calls") and m.get("content"))]


# ---------------------------------------------------------------------- 단일 vs 멀티 비교
COMPARE_SCENARIO = [
    "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘",
    "그 중에 넓은 건?",
    "공장 지을 수 있는 땅 중 넓은 거 3개 보여줘",
    "음성군 말고 다른 지역에도 있어?",
    "정말 다른 지역에는 없어?",
    "양평에 전원주택 지을 땅 3억 이하로 찾아줘",
    "여주 공장 부지 2개 비교해줘",
]


def compare_agents(agent, questions: Optional[list[str]] = None, llm=None) -> dict[str, Any]:
    """같은 대화를 단일 Agent 와 멀티 Agent 로 돌려 비교: 답변 여부, 단계·도구 호출 수, 판단 횟수, 시간, 결과 일치."""
    questions = questions or COMPARE_SCENARIO
    out: dict[str, Any] = {"questions": questions, "single": [], "multi": []}
    ToolAgent(agent, llm=llm, log_dir="").run("음성군 땅")  # 준비 운동 (첫 실행의 불러오기 시간 제외)
    for mode in ("single", "multi"):
        ctx = AgentContext()
        runner = (ToolAgent(agent, llm=llm, ctx=ctx, log_dir="") if mode == "single"
                  else MultiAgent(agent, llm=llm, ctx=ctx, log_dir=""))
        history: list[dict] = []
        for q in questions:
            t0 = time.perf_counter()
            r, history = runner.run(q, history)
            decisions = len({(s.agent, s.n) for s in r.steps if s.kind in {"tool", "answer", "guard"}})  # 정책 호출
            out[mode].append({"question": q, "answer": r.answer, "listing_ids": r.listing_ids,
                              "ok": bool(r.answer) and r.stopped == "answer",
                              "steps": len(r.steps),
                              "tool_calls": sum(1 for s in r.steps if s.kind == "tool" and s.agent != SUPERVISOR),
                              "delegations": sum(1 for s in r.steps if s.kind == "tool" and s.agent == SUPERVISOR),
                              "tool_errors": r.tool_errors,
                              "decisions": decisions, "agents": list(dict.fromkeys(s.agent for s in r.steps if s.agent)),
                              "ms": round((time.perf_counter() - t0) * 1000, 1)})
    same = [a["listing_ids"] == b["listing_ids"] for a, b in zip(out["single"], out["multi"])]
    out["summary"] = {mode: {"answered": sum(r["ok"] for r in rows), "tool_calls": sum(r["tool_calls"] for r in rows),
                             "delegations": sum(r["delegations"] for r in rows),
                             "tool_errors": sum(r["tool_errors"] for r in rows),
                             "decisions": sum(r["decisions"] for r in rows),
                             "ms": round(sum(r["ms"] for r in rows), 1)}
                      for mode, rows in (("single", out["single"]), ("multi", out["multi"]))}
    out["summary"]["same_listings"] = f"{sum(same)}/{len(same)}"
    return out


def compare_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# 단일 Agent vs 멀티 Agent 비교", "",
             "| 항목 | 단일 Agent | 멀티 Agent |", "|---|---|---|"]
    for key, label in (("answered", "답변한 질문"), ("tool_calls", "매물 도구 호출"),
                       ("delegations", "Agent 사이 맡기기"), ("tool_errors", "도구 오류"),
                       ("decisions", "판단(정책 호출) 횟수"), ("ms", "총 시간(ms)")):
        lines.append(f"| {label} | {s['single'][key]} | {s['multi'][key]} |")
    lines += [f"| 같은 매물을 보여준 질문 | {s['same_listings']} | |", "", "## 질문별", ""]
    for a, b in zip(report["single"], report["multi"]):
        lines += [f"### {a['question']}", f"- 단일: {a['answer']}  (도구 {a['tool_calls']}회, {a['ms']}ms)",
                  f"- 멀티: {b['answer']}  (도구 {b['tool_calls']}회 + 맡기기 {b['delegations']}회, "
                  f"{' → '.join(b['agents'])}, {b['ms']}ms)", ""]
    lines += ["## 해석", "",
              "- 멀티 Agent 는 총괄이 일을 나누고 보고를 모으는 만큼 판단(=LLM 호출) 횟수가 늘어납니다 (비용·지연 증가).",
              "- 대신 Agent 마다 도구가 3~4개로 줄어 지시문이 짧고, 역할별로 따로 고치거나 시험하기 쉽습니다.",
              "- 같은 대화 상태를 함께 쓰므로 이어서 묻기('그 중에', '다른 지역', '정말 없어?') 결과는 단일 Agent 와 같아야 합니다."]
    return "\n".join(lines)


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
