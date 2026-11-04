"""'음성군 말고 다른 지역' / '다른 지역에도 있어?' / '진천군에도 있어?' 정확도 테스트."""

import json
import shutil
from pathlib import Path

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import ToolTurn
from real_estate_agent.requirement_parser import RequirementParser
from real_estate_agent.tool_agent import AgentContext, ToolAgent

from test_tool_agent import ScriptedLLM, call, router

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def agent(tmp_path):
    """샘플(양평·여주·음성) + 진천군 폴더에 공장 부지 2건 (사용자 PC 와 같은 상황)."""
    data = tmp_path / "매물"
    shutil.copytree(SAMPLE, data)
    (data / "진천군").mkdir()
    wb = Workbook()
    ws = wb.active
    ws.append(["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "도로", "특이사항"])
    ws.append(["J-1", "충청북도 진천군 덕산읍 용몽리 10", "공장용지", "계획관리지역", 1200, "8억", "8m 도로 접함", "공장 부지, 진천IC 5분"])
    ws.append(["J-2", "충청북도 진천군 이월면 송림리 20", "공장용지", "계획관리지역", 900, "6억", "6m 도로 접함", "공장 부지"])
    wb.save(data / "진천군" / "진천_공장.xlsx")
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "r.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


def region(agent, lid):
    return agent.find_listing(lid).address.split()[1]


def test_parser_reads_region_exclusions(agent):
    p = RequirementParser(None, agent.known_regions(), [l.address for l in agent.listings()])
    r = p.parse("음성군 말고 다른 지역에 공장 지을 땅")
    assert r.regions == [] and r.excluded_regions == ["음성군"] and r.purpose == "공장" and r.excluded_keywords == []
    assert p.parse("음성 빼고 창고 부지").excluded_regions == ["음성군"]
    assert p.parse("양평 외에 전원주택 땅").excluded_regions == ["양평군"]
    assert p.parse("여주 아닌 곳 공장").excluded_regions == ["여주시"]
    r = p.parse("음성군 땅 임야 말고")  # 지목 제외는 그대로
    assert r.regions == ["음성군"] and r.excluded_keywords == ["임야"] and r.excluded_regions == []


def test_users_conversation(agent):
    """공장 넓은 거 3개 → 그 중에 넓은 건? → 다른 지역에 있어? → 음성군 말고 다른 지역에도 있어?"""
    t = ToolAgent(agent, ctx=AgentContext())
    first, h = t.run("공장 지을 수 있는 땅 중 넓은 거 3개 보여줘")
    shown = {region(agent, i) for i in first.listing_ids}
    _, h = t.run("그 중에 넓은 건?", h)
    third, h = t.run("공장 지을 땅으로 다른 지역에 있어?", h)
    assert third.answer != "해당 조건에 맞는 매물이 없습니다."
    assert third.listing_ids and not ({region(agent, i) for i in third.listing_ids} & shown)  # 보여준 지역은 빼고
    fourth, _ = t.run("음성군 말고 다른 지역에도 있어?", h)
    regions = {region(agent, i) for i in fourth.listing_ids}
    assert fourth.answer.startswith("음성군 말고 다른 지역에서") and "음성군" not in regions
    assert "진천군" in regions  # 새로 넣은 진천군 공장 부지가 나옴
    for lid in third.listing_ids + fourth.listing_ids:  # 공장 조건은 계속 유지
        assert agent.find_listing(lid).jimok in {"공장용지", "잡종지", "대", "창고용지"} or agent.find_listing(lid).zoning


def test_named_exclusion_replaces_previous_one(agent):
    t = ToolAgent(agent, ctx=AgentContext())
    _, h = t.run("음성군 창고 지을 땅")
    _, h = t.run("다른 지역에도 있어?", h)          # 음성군 제외
    r, _ = t.run("양평군 말고 다른 곳은?", h)        # 이제는 양평군만 제외 (음성군은 다시 포함)
    regions = {region(agent, i) for i in r.listing_ids}
    assert "양평군" not in regions and "음성군" in regions


def test_short_region_follow_up_keeps_conditions(agent):
    t = ToolAgent(agent, ctx=AgentContext())
    _, h = t.run("공장 지을 땅 보여줘")
    r, _ = t.run("진천군에도 있어?", h)
    assert {region(agent, i) for i in r.listing_ids} == {"진천군"}
    assert r.answer.startswith("앞의 조건 그대로") and "공장 용도" in r.answer


def test_gpt_style_exclude_regions_argument(agent):
    """GPT/Claude 가 exclude_regions 로 부르는 경우 (regions 에 넣지 않음)."""
    def check(messages):
        res = json.loads(messages[-1]["content"])["result"]
        assert res["제외_지역"] == ["음성군"] and res["조건에_맞는_매물_수"] > 0
        return ToolTurn(text="진천군 등에 있습니다.")

    fake = ScriptedLLM([call("search_listings", query="공장 지을 땅", limit=3), ToolTurn(text="찾았습니다."),
                        call("search_listings", use_previous_conditions=True, exclude_regions=["음성군"]), check])
    t = ToolAgent(agent, llm=router(fake), ctx=AgentContext())
    _, h = t.run("공장 지을 땅")
    r, _ = t.run("음성군 말고 다른 지역에도 있어?", h)
    assert r.listing_ids and all(region(agent, i) != "음성군" for i in r.listing_ids)


@pytest.mark.parametrize("text, expected", [
    ("진천군이랑 음성군 말고 다른 지역으로 있어?", ["진천군", "음성군"]),
    ("진천, 음성 빼고 공장 땅", ["진천군", "음성군"]),
    ("진천군과 음성군은 제외하고", ["진천군", "음성군"]),
    ("진천·음성·여주 말고", ["진천군", "음성군", "여주시"]),
    ("진천군하고 음성군 외에 창고 부지", ["진천군", "음성군"]),
    ("진천군이나 음성군 아닌 곳", ["진천군", "음성군"]),
    ("공장 땅 음성군 말고", ["음성군"]),
])
def test_parser_excludes_every_listed_region(agent, text, expected):
    p = RequirementParser(None, agent.known_regions(), [l.address for l in agent.listings()])
    r = p.parse(text)
    assert r.excluded_regions == expected and r.regions == []


def test_users_conversation_with_two_excluded_regions(agent):
    """공장 넓은 거 3개 → 그 중에 넓은 건? → 다른 지역에 공장 지을 땅 있어? → 진천군이랑 음성군 말고 다른 지역으로 있어?"""
    t = ToolAgent(agent, ctx=AgentContext())
    h = []
    for q in ["공장 지을 수 있는 땅 중 넓은 거 3개 보여줘", "그 중에 넓은 건?", "다른 지역에 공장 지을 수 있는 땅 있어?"]:
        _, h = t.run(q, h)
    r, _ = t.run("진천군이랑 음성군 말고 다른 지역으로 있어?", h)
    regions = {region(agent, i) for i in r.listing_ids}
    assert r.listing_ids and not regions & {"진천군", "음성군"}
    assert r.answer.startswith("진천군, 음성군 말고 다른 지역에서")
