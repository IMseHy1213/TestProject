"""'어느 지역 매물이 제일 많아요?' → 수만 답함 (매물 카드 없음), '제일 많은/그 지역 매물 보여줘' → 그때 그 지역 매물."""

from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import ToolTurn
from real_estate_agent.tool_agent import AgentContext, ToolAgent
from real_estate_agent.tool_agent.loop import _asks_region_counts, _region_rank_ref
from real_estate_agent.tool_agent.multi import MultiAgent

from test_tool_agent import ScriptedLLM, call, router

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rc")
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp / "r.db", log_dir=tmp / "l", llm_providers=["rule"]))
    a.sync()
    return a


def regions_searched(r):
    return [s.args.get("regions") for s in r.steps if s.tool == "search_listings"]


def test_question_patterns():
    for q in ["어느 지역 매물이 제일 많아요?", "어디에 매물이 제일 많아?", "지역별로 몇 건이에요?", "음성군에 매물 몇 개 있어요?",
              "어느 지역 매물이 제일 적어요?"]:
        assert _asks_region_counts(q), q
    for q in ["어느 지역이 제일 싸요?", "어느 지역이 창고에 적합해요?", "음성군 땅 2개 보여줘", "제일 많은 지역 매물을 보고 싶어요"]:
        assert not _asks_region_counts(q), q
    assert _region_rank_ref("제일 많은 지역 매물") == 0 and _region_rank_ref("두 번째로 많은 지역") == 1
    assert _region_rank_ref("제일 적은 곳 매물") == -1 and _region_rank_ref("그 지역 매물") is None
    assert _region_rank_ref("그 지역 매물", pronoun_ok=True) == 0


@pytest.mark.parametrize("runner_cls", [ToolAgent, MultiAgent])
def test_counts_then_listings(agent, runner_cls):
    t = runner_cls(agent, ctx=AgentContext(), log_dir="")
    r1, h = t.run("어느 지역 매물이 제일 많아요?")
    assert r1.listing_ids == [] and not regions_searched(r1)  # 수만 답함
    assert r1.answer.startswith("보유 매물 20건 중 양평군이 11건으로 가장 많습니다") and "음성군 6건" in r1.answer
    r2, h = t.run("제일 많은 지역 매물을 보고 싶어요", h)
    assert regions_searched(r2) == [["양평군"]] and len(r2.listing_ids) == 5
    assert all("양평군" in agent.find_listing(i).address for i in r2.listing_ids)
    r3, h = t.run("두 번째로 많은 지역 매물도 보여줘", h)
    assert regions_searched(r3) == [["음성군"]]


def test_that_region_means_the_one_just_mentioned(agent):
    t = ToolAgent(agent, ctx=AgentContext(), log_dir="")
    r, h = t.run("어느 지역 매물이 제일 적어요?")
    assert "여주시가 3건으로 가장 적습니다" in r.answer and not r.listing_ids
    r, h = t.run("그 지역 매물 보여줘", h)
    assert regions_searched(r) == [["여주시"]]
    r, h = t.run("음성군에 매물 몇 개 있어요?", h)
    assert r.answer == "음성군 매물은 6건입니다 (전체 20건 중)." and not r.listing_ids
    r, h = t.run("거기 창고 지을 땅 보여줘", h)
    assert regions_searched(r) == [["음성군"]] and "창고" in r.answer


def test_without_asking_counts_first(agent):
    r, _ = ToolAgent(agent, ctx=AgentContext(), log_dir="").run("제일 많은 지역 매물 보여줘")
    assert [s.tool for s in r.steps if s.kind == "tool"] == ["list_regions", "search_listings"]
    assert regions_searched(r) == [["양평군"]]


def test_ai_answer_with_only_counts_shows_no_cards(agent):
    llm = ScriptedLLM([call("list_regions"), ToolTurn(text="양평군 매물이 11건으로 가장 많습니다.")])
    r, _ = ToolAgent(agent, llm=router(llm), ctx=AgentContext(), log_dir="").run("어느 지역 매물이 제일 많아요?")
    assert r.listing_ids == [] and r.policy == "openai"


def test_benchmark_region_cases(agent):
    from real_estate_agent.benchmark import load_dataset, run_benchmark

    data = load_dataset()
    data = {**data, "cases": [c for c in data["cases"] if c["id"].startswith("R")]}
    report = run_benchmark(agent, ["single", "multi"], dataset=data)
    for cfg in report["configs"].values():
        assert cfg["failed_cases"] == [] and cfg["case_pass"] == 100.0


@pytest.mark.parametrize("runner_cls", [ToolAgent, MultiAgent])
def test_follow_up_after_counts_stays_in_that_region(agent, runner_cls):
    """'어느 지역 매물이 제일 많아요?' → '양평군 11건' 다음의 '그 중에 넓은 건?' 은 양평군 매물 중에서."""
    t = runner_cls(agent, ctx=AgentContext(), log_dir="")
    _, h = t.run("어느 지역 매물이 제일 많아요?")
    r, h = t.run("그 중에 넓은 건?", h)
    yangpyeong = [l for l in agent.listings() if "양평군" in l.address and l.status not in {"계약완료", "보류"}]
    widest = max(yangpyeong, key=lambda l: l.area_pyeong)
    assert regions_searched(r) == [["양평군"]] and r.listing_ids == [widest.listing_id]
    assert r.answer.startswith("양평군 매물 11건 중 가장 넓은 매물은")
    r, h = t.run("그 중에 싼 건?", h)  # 계속 양평군 안에서
    cheapest = min(yangpyeong, key=lambda l: l.price_manwon)
    assert r.listing_ids == [cheapest.listing_id] and "양평군 매물 11건 중 가장 싼 매물" in r.answer
    r, h = t.run("공장 지을 땅 보여줘", h)  # 새로 검색하면 '그 중에' 는 그 검색 결과를 가리킴
    r2, _ = t.run("그 중에 넓은 건?", h)
    assert r2.listing_ids and r2.listing_ids[0] in r.listing_ids and "앞서 보여드린" in r2.answer


def test_follow_up_after_least_region(agent):
    t = ToolAgent(agent, ctx=AgentContext(), log_dir="")
    _, h = t.run("어느 지역 매물이 제일 적어요?")
    r, _ = t.run("거기서 제일 싼 거 2개", h)
    assert regions_searched(r) == [["여주시"]] and len(r.listing_ids) == 2
