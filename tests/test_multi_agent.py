"""v0.7 Multi-Agent: 총괄 Agent → 검색 Agent · 분석 Agent (agent-as-tool), 공유 상태, 단일 vs 멀티 비교."""

import json
import re
import shutil
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import ToolTurn
from real_estate_agent.tool_agent import AgentContext, ToolAgent
from real_estate_agent.tool_agent.multi import (ANALYSIS_TOOLS, ANALYST, SEARCH_TOOLS, SEARCHER, SUPERVISOR,
                                                MultiAgent, compare_agents, compare_markdown, text_history)
from real_estate_agent.tool_agent.service import AgentService

from test_tool_agent import ScriptedLLM, call, router

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def agent(tmp_path):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / "m.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


def tools_of(result, who):
    return [s.tool for s in result.steps if s.kind == "tool" and s.agent == who]


def test_each_agent_only_gets_its_tools(agent):
    m = MultiAgent(agent)
    assert set(m.searcher.registry.tools) == SEARCH_TOOLS
    assert set(m.analyst.registry.tools) == ANALYSIS_TOOLS
    assert set(m.supervisor.registry.tools) == {"delegate_search", "delegate_analysis"}


def test_rule_mode_delegates_search_then_analysis(agent):
    m = MultiAgent(agent, ctx=AgentContext())
    r, h = m.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    assert tools_of(r, SUPERVISOR) == ["delegate_search", "delegate_analysis"]
    assert tools_of(r, SEARCHER) == ["search_listings"]
    assert tools_of(r, ANALYST) == ["compare_listings", "check_land_use"]
    assert len(r.listing_ids) == 2 and "최저가" in r.answer and r.stopped == "answer"
    assert any("최저가" in b for b in r.badges.values())
    # 공유 상태: '그 중에 넓은 건?' 은 검색 Agent 만으로, 보여준 2건 안에서
    r2, _ = m.run("그 중에 넓은 건?", h)
    assert tools_of(r2, SUPERVISOR) == ["delegate_search"]
    assert len(r2.listing_ids) == 1 and r2.listing_ids[0] in r.listing_ids
    widest = max(r.listing_ids, key=lambda i: agent.find_listing(i).area_m2)
    assert r2.listing_ids[0] == widest


def test_multi_matches_single_agent(agent):
    report = compare_agents(agent)
    s = report["summary"]
    assert s["single"]["answered"] == s["multi"]["answered"] == len(report["questions"])
    assert s["same_listings"] == f"{len(report['questions'])}/{len(report['questions'])}"
    for a, b in zip(report["single"], report["multi"]):
        assert a["answer"] == b["answer"]
        assert b["agents"][0] == SEARCHER or SUPERVISOR in b["agents"]
    assert s["multi"]["delegations"] >= len(report["questions"]) and s["multi"]["tool_errors"] == 0
    md = compare_markdown(report)
    assert "| 답변한 질문 |" in md and "Agent 사이 맡기기" in md


def test_llm_supervisor_delegates_with_ids(agent):
    """GPT/Claude 총괄: 검색 Agent 에게 맡기고 → 그 결과 id 로 분석 Agent 에게 비교를 맡긴 뒤 정리."""
    state = {}

    def analysis(messages):
        report = json.loads(messages[-1]["content"])["result"]
        state["ids"] = [x["id"] for x in report["매물"]][:2]
        return call("delegate_analysis", task="두 매물 비교", listing_ids=state["ids"])

    def compare(messages):
        assert "대상 매물 id" in messages[-1]["content"]
        return call("compare_listings", listing_ids=state["ids"])

    llm = ScriptedLLM([
        call("delegate_search", task="음성군 창고 부지 싼 순 2개"),               # 총괄
        call("search_listings", query="음성군 창고", sort="price_asc", limit=2),  # 검색 Agent
        ToolTurn(text="음성군 창고 부지 2건을 찾았습니다."),                       # 검색 Agent 보고
        analysis,                                                                 # 총괄
        compare,                                                                  # 분석 Agent
        ToolTurn(text="두 매물을 비교했습니다."),                                   # 분석 Agent 보고
        ToolTurn(text="음성군 창고 부지 2건을 비교했습니다."),                       # 총괄 최종 답
    ])
    m = MultiAgent(agent, llm=router(llm), ctx=AgentContext())
    r, h = m.run("음성군에 창고 지을 땅 싼 거 2개 비교해줘")
    assert r.answer == "음성군 창고 부지 2건을 비교했습니다." and r.policy == "openai"
    assert [s.agent for s in r.steps if s.kind == "tool"] == [SEARCHER, SUPERVISOR, ANALYST, SUPERVISOR]
    assert set(r.listing_ids) == {agent.find_listing(i).listing_id for i in r.listing_ids} and len(r.listing_ids) == 2
    # 전문 Agent 에게는 총괄의 도구 기록이 아닌 대화 글자만 넘어감
    assert not any(m_.get("tool_calls") for m_ in text_history(h))


def test_bad_listing_id_is_reported_back(agent):
    m = MultiAgent(agent, ctx=AgentContext())
    with pytest.raises(Exception) as e:
        m.delegate(ANALYST, "비교", ["없는-id"])
    assert "없는-id" in str(e.value)


def test_service_mode_and_switching(agent):
    service = AgentService(agent)
    cid, turn = service.ask("v1", None, "여주 공장 부지 2개 비교해줘", mode="multi")
    assert turn["mode"] == "multi" and {s["agent"] for s in turn["steps"]} >= {SUPERVISOR, SEARCHER, ANALYST}
    _, turn2 = service.ask("v1", cid, "그 중에 넓은 건?", mode="single")  # 도중에 방식을 바꿔도 이어서 묻기 유지
    assert turn2["mode"] == "single" and len(turn2["listing_ids"]) == 1
    assert turn2["listing_ids"][0] in turn["listing_ids"]
    _, turn3 = service.ask("v1", cid, "그 중에 싼 건?", mode="multi")
    assert turn3["mode"] == "multi" and turn3["listing_ids"][0] in turn["listing_ids"]


def test_web_mode_toggle_and_trace(tmp_path, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                              llm_providers=["rule"]))
    app.config["TESTING"] = True
    c = app.test_client()
    html = c.get("/agent").get_data(as_text=True)
    assert 'name="mode" value="multi"' in html
    csrf = re.search(r'name="csrf" value="([^"]+)"', html).group(1)
    r = c.post("/agent/stream", data={"q": "양평 땅 2개 비교해줘", "csrf": csrf, "mode": "multi"})
    events = [json.loads(line[6:]) for line in r.get_data(as_text=True).splitlines() if line.startswith("data: ")]
    agents = [e["step"]["agent"] for e in events if e["type"] == "step"]
    assert SEARCHER in agents and ANALYST in agents and SUPERVISOR in agents
    assert [e["type"] for e in events].count("done") == 1  # 끝 알림은 총괄 Agent 한 번만
    html = c.get("/agent").get_data(as_text=True)
    assert "(멀티 Agent)" in html and 'class="agent-tag agent-search"' in html
    assert re.search(r'value="multi" checked', html)  # 고른 방식 기억


def test_api_multi_mode(tmp_path):
    from fastapi.testclient import TestClient

    from real_estate_agent.api import create_api

    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / "api.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    api = TestClient(create_api(agent=a))
    body = api.post("/api/v1/agent/ask", json={"question": "여주 공장 부지 2개 비교해줘", "mode": "multi"}).json()
    assert len(body["listings"]) == 2 and {s["agent"] for s in body["steps"]} >= {SUPERVISOR, SEARCHER, ANALYST}
    assert api.post("/api/v1/agent/ask", json={"question": "땅", "mode": "team"}).status_code == 422
