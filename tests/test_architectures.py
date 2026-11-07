"""v0.4 Architecture Study (Workflow·ReAct·Plan-and-Execute·Multi) + v0.5 Framework (LangGraph) + 태그 스크립트."""

import importlib.util
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.architectures import PlanExecuteAgent, WorkflowAgent
from real_estate_agent.config import Settings
from real_estate_agent.tool_agent import AgentContext, ToolAgent

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arch")
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp / "a.db", log_dir=tmp / "logs", llm_providers=["rule"]))
    a.sync()
    return a


def tools_of(r):
    return [s.tool for s in r.steps if s.kind == "tool"]


def test_workflow_is_fixed_and_forgets_context(agent):
    w = WorkflowAgent(agent, ctx=AgentContext(), log_dir="")
    r, h = w.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    assert tools_of(r)[:2] == ["search_listings", "compare_listings"] and r.decisions == 0
    assert not [s for s in r.steps if s.kind == "fallback"]  # 판단하지 않으니 'AI 없음' 안내도 없음
    r2, _ = w.run("그 중에 넓은 건?", h)
    # 앞 대화를 이어 받지 못함 (구조의 한계): 보여준 2건 중에서 고르지 않고 전체 매물에서 넓은 순 5건
    assert len(r2.listing_ids) == 5 and not set(r2.listing_ids) <= set(r.listing_ids)


def test_plan_execute_plans_once(agent):
    p = PlanExecuteAgent(agent, ctx=AgentContext(), log_dir="")
    r, _ = p.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    plan = [s for s in r.steps if s.kind == "plan"]
    assert plan and "search_listings → compare_listings → check_land_use" in plan[0].summary
    assert r.decisions == 2 and tools_of(r) == ["search_listings", "compare_listings", "check_land_use"]
    react, _ = ToolAgent(agent, ctx=AgentContext(), log_dir="").run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    assert r.listing_ids == react.listing_ids and react.decisions == 4  # 같은 답, 판단은 계획 쪽이 적음


def test_plan_execute_does_not_adapt(agent):
    """결과가 0건인데 계획대로 비교 → 헛도구 호출. ReAct 는 결과를 보고 건너뜀."""
    q = "음성군 3000평 넘는 땅 2개 비교해줘"
    p, _ = PlanExecuteAgent(agent, ctx=AgentContext(), log_dir="").run(q)
    r, _ = ToolAgent(agent, ctx=AgentContext(), log_dir="").run(q)
    assert any(s.ok is False for s in p.steps if s.kind == "tool")
    assert not any(s.ok is False for s in r.steps if s.kind == "tool") and tools_of(r) == ["search_listings"]


def test_plan_execute_llm_planner(agent):
    class FakeLLM:
        available = ready = True
        last_used = "openai"

        def complete_json(self, system, user, max_tokens=800):
            return {"plan": [{"tool": "search_listings", "args": {"query": user, "limit": 2, "sort": "price_asc"}},
                             {"tool": "compare_listings", "args": {"listing_ids": "$search[:2]"}},
                             {"tool": "no_such_tool", "args": {}}]}

        def tool_step(self, system, messages, tools, max_tokens=1500):
            from real_estate_agent.llm import ToolTurn
            return ToolTurn(text="두 매물을 비교했습니다.", provider="openai")

    p = PlanExecuteAgent(agent, llm=FakeLLM(), ctx=AgentContext(), log_dir="")
    r, _ = p.run("양평 땅 싼 거 2개 비교")
    assert tools_of(r) == ["search_listings", "compare_listings"]  # 없는 도구는 계획에서 빠짐
    assert r.answer == "두 매물을 비교했습니다." and r.decisions == 2


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph 미설치")
def test_langgraph_matches_hand_written(agent):
    from real_estate_agent.framework.langgraph_agent import LangGraphAgent

    lg, rt = LangGraphAgent(agent, ctx=AgentContext(), log_dir=""), ToolAgent(agent, ctx=AgentContext(), log_dir="")
    h1 = h2 = []
    for q in ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "그 중에 넓은 건?", "음성군 말고 다른 지역에도 있어?"]:
        a, h1 = lg.run(q, h1)
        b, h2 = rt.run(q, h2)
        assert (a.answer, a.listing_ids, tools_of(a)) == (b.answer, b.listing_ids, tools_of(b))
    states = lg.saved_states()
    assert states and states[0]["status"] == "done"  # 프레임워크 checkpointer 에 단계별 상태가 남음
    assert "decide_and_act" in lg.mermaid() and lg.mode_label == "langgraph"


def test_study_report(agent):
    from real_estate_agent.architecture_study import framework_markdown, run_study, study_markdown

    report = run_study(agent)
    b = report["bench"]["configs"]
    assert b["workflow"]["case_pass"] < b["single"]["case_pass"]  # 고정 파이프라인은 이어서 묻기에 약함
    assert b["single"]["case_pass"] == b["multi"]["case_pass"]
    # 계획을 미리 고정하면 '먼저 세어 보고 → 그 지역 검색' 처럼 중간 결과에 따라 정해지는 일을 못 함
    assert b["plan_execute"]["case_pass"] < b["single"]["case_pass"] and "R03" in b["plan_execute"]["failed_cases"]
    assert b["multi"]["decisions_per_turn"] > b["single"]["decisions_per_turn"]
    assert report["adaptivity"]["plan_execute"]["tool_errors"] > report["adaptivity"]["single"]["tool_errors"] == 0
    md = study_markdown(report)
    assert "## 4. 결정" in md and "Plan-and-Execute" in md and "E-30" not in md
    if "langgraph" in b:
        assert "완전히 같음" in framework_markdown(report)


def test_tag_script_versions():
    spec = importlib.util.spec_from_file_location("tag_versions", ROOT / "scripts" / "tag_versions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tags = [t for t, _, _ in mod.VERSIONS]
    assert tags == ["v0.1", "v0.2", "v0.3", "v0.4", "v0.5", "v0.6", "v0.7", "v0.8", "v0.9", "v1.0", "v1.1", "v1.2",
                    "v1.3"]
    assert all(m.startswith(t) for t, _, m in mod.VERSIONS)
    assert mod.plan(set(tags)) == []  # 이미 다 있으면 할 일 없음 (덮어쓰지 않음)
