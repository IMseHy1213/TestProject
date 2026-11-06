"""v1.3 Benchmark: 정답(oracle), 채점, 방식별 실행, 기준선 비교(회귀 검사), 동시 처리량."""

import json
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.benchmark import (BASELINE, DATASET, available_configs, baseline_of, check_turn, compare_baseline,
                                         load_dataset, markdown, oracle, run_benchmark, run_load)
from real_estate_agent.config import Settings
from real_estate_agent.tool_agent.loop import AgentResult, _wanted_count

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp / "b.db", log_dir=tmp / "logs", llm_providers=["rule"]))
    a.sync()
    return a


def test_dataset_is_well_formed():
    d = load_dataset()
    ids = [c["id"] for c in d["cases"]]
    assert len(ids) == len(set(ids)) >= 30 and DATASET.exists()
    assert all(t["q"] and isinstance(t.get("expect"), dict) for c in d["cases"] for t in c["turns"])


def test_oracle_from_raw_data(agent):
    """음성군 창고 가능 + 싼 순 2개: 원자료로 직접 계산한 정답."""
    assert [l.listing_id for l in oracle(agent, {"regions": ["음성군"], "purpose": "창고", "sort": "price_asc", "k": 2})] \
        == ["E-3003", "E-3001"]
    assert all(l.price_manwon <= 22000 for l in oracle(agent, {"max_price": 20000}))  # 예산 10% 여유
    assert oracle(agent, {"regions": ["제주"]}) == []
    assert all("음성군" not in l.address for l in oracle(agent, {"exclude_regions": ["음성군"]}))


def test_checker_catches_wrong_answers(agent):
    wrong = AgentResult("q", "가장 싼 매물은 Y-2002(9억 8,000만원)입니다.", [], ["Y-2002"], "rule", "answer", 1.0)
    fails, m = check_turn(agent, wrong, {"gold": {"regions": ["음성군"], "purpose": "창고", "sort": "price_asc", "k": 2},
                                         "count": 2}, [], [])
    assert any("정답 순서" in f for f in fails) and any("2건" in f for f in fails)
    assert any("근거" in f for f in fails)  # 도구 결과 없이 가격을 말함
    leak = AgentResult("q", "연락처는 010-1234-5678 입니다", [], [], "rule", "answer", 1.0)
    fails, _ = check_turn(agent, leak, {"contact_notice": True}, [], [])
    assert any("전화번호" in f for f in fails) and any("안내 문장" in f for f in fails)


def test_counts_in_words():
    assert _wanted_count("공장 할 땅 제일 넓은 걸로 하나") == 1 and _wanted_count("셋만 보여줘") == 3
    assert _wanted_count("하나도 없어?") is None and _wanted_count("두 개 비교") == 2


def test_named_listing_is_the_one_checked(agent):
    """벤치마크로 찾은 문제: '매물번호 + 지을 수 있어?' 에 다른 매물(검색 1순위)을 판단하던 것."""
    from real_estate_agent.tool_agent import AgentContext, ToolAgent
    from real_estate_agent.tool_agent.multi import MultiAgent

    for runner in (ToolAgent(agent, ctx=AgentContext(), log_dir=""), MultiAgent(agent, ctx=AgentContext(), log_dir="")):
        r, _ = runner.run("T-1004에 공장 지을 수 있어?")
        checks = [s for s in r.steps if s.tool == "check_land_use"]
        assert checks and checks[0].args["listing_id"] == "T-1004" and "어려" in r.answer


def test_benchmark_meets_baseline(agent):
    """회귀 검사: 기준선에서 통과하던 문제가 하나라도 실패하면 이 테스트가 실패."""
    report = run_benchmark(agent, ["single", "multi"])
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert compare_baseline(report, baseline) == []
    single = report["configs"]["single"]
    assert single["precision"] == 100.0 and single["case_pass"] >= 95
    assert single["failed_cases"] == ["H05"]  # 알려진 약점 (규칙 기반에서 '자세히' 요청) - 고치면 기준선 갱신
    md = markdown(report)
    assert "문제 통과율" in md and "H05" in md


def test_all_modes_give_same_quality(agent):
    report = run_benchmark(agent, available_configs())
    passes = {c: v["passed_cases"] for c, v in report["configs"].items()}
    assert len(set(map(tuple, passes.values()))) == 1  # 상담원 1명·상담팀·분산·MCP 결과가 같음


def test_regression_is_detected(agent):
    report = run_benchmark(agent, ["single"], dataset={**load_dataset(), "cases": load_dataset()["cases"][:3]})
    fake = baseline_of(report)
    fake["configs"]["single"]["passed_cases"] = ["S01", "S02", "S03", "ZZZ"]
    report["configs"]["single"]["failed_cases"] = ["S02"]
    assert compare_baseline(report, fake) and "S02" in compare_baseline(report, fake)[0]


def test_load(agent):
    rows = run_load(agent, levels=(1, 2), requests_per_level=4)
    assert [r["concurrency"] for r in rows] == [1, 2] and all(r["errors"] == 0 and r["throughput"] > 0 for r in rows)
    slow = run_load(agent, levels=(4,), requests_per_level=4, ai_delay=0.05, limit=1)
    assert slow[0]["limit"] == 1 and slow[0]["p95_ms"] > slow[0]["p50_ms"] * 0.9  # 자리 1개면 줄을 섬
