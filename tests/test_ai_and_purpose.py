"""생성형 AI 취향 학습 + '창고를 지을 수 있는 토지' 같은 목적 조건 테스트."""

import json
import re
import shutil
from datetime import date
from pathlib import Path

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.ai_learning import apply_plan, ask_ai, sanitize_plan
from real_estate_agent.config import Settings
from real_estate_agent.land_use import check_use
from real_estate_agent.llm.base import LLMClient, LLMUnavailable
from real_estate_agent.llm.router import LLMRouter
from real_estate_agent.models import CustomerRequirement, LandListing
from real_estate_agent.preference import PreferenceModel
from real_estate_agent.recommender import RecommendationSession, RecommendOptions, rank
from real_estate_agent.requirement_parser import RequirementParser

TODAY = date(2026, 10, 12)
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


def L(id, addr, m2, price, **kw):
    return LandListing(listing_id=id, address=addr, area_m2=m2, price_manwon=price, **kw)


# ---------------- 목적: 창고를 지을 수 있는 토지 ----------------
STOCK = [
    L("W1", "충청북도 음성군 감곡면 오향리 1", 2000, 90000, jimok="잡종지", zoning="계획관리지역", road_access="8m 도로"),
    L("W2", "충청북도 음성군 대소면 오류리 2", 1500, 500000, jimok="전", zoning="계획관리지역", road_access="6m 도로"),
    L("W3", "충청북도 음성군 원남면 하당리 3", 1800, 8000, jimok="답", zoning="농림지역", road_access="농로"),
    L("W4", "충청북도 음성군 금왕읍 무극리 4", 1000, 30000, jimok="임야", zoning="보전관리지역", road_access="4m"),
    L("W5", "충청북도 음성군 맹동면 쌍정리 5", 900, 40000, jimok="대", zoning="", road_access="4m",
      description="물류창고 건축허가 득"),
    L("W6", "경상북도 문경시 산북면 대하리 6", 3000, 20000, jimok="공장용지", zoning="", road_access="6m"),
    L("W7", "경상남도 밀양시 산내면 가인리 7", 2500, 15000, jimok="전", zoning="생산녹지지역", road_access="4m"),
    L("W8", "충청북도 음성군 감곡면 문촌리 8", 2500, 60000, jimok="대", zoning="자연녹지지역", road_access="맹지"),
]


def ids(q, known=("음성군",)):
    req = RequirementParser(None, list(known)).parse(q)
    return {s.listing.listing_id for s in rank(STOCK, req, today=TODAY)[0]}, req


def test_warehouse_in_eumseong_regardless_of_price():
    got, req = ids("음성군에 위치한 창고를 지을 수 있는 토지")
    assert req.purpose == "창고" and req.regions == ["음성군"] and not req.budget_max
    # W2(50억)·W1(9억) 처럼 가격과 상관없이 창고 가능한 음성군 땅만. 농림(W3)·보전관리(W4)·맹지(W8) 제외
    assert got == {"W1", "W2", "W5"}


def test_warehouse_anywhere():
    got, req = ids("창고를 지을 수 있는 토지")
    assert req.purpose == "창고" and not req.regions
    assert got == {"W1", "W2", "W5", "W6"}  # 지역 상관없이 (문경 공장용지 포함, 밀양 생산녹지 전은 불가)


def test_check_use_reasons():
    assert check_use(STOCK[1], "창고").caution  # 전 -> 전용허가 필요 안내
    assert not check_use(STOCK[2], "창고").ok
    assert check_use(STOCK[4], "창고").reason == "매물 정보에 '창고' 용도 명시"
    no = L("N", "x군 y리 1", 900, 1000, jimok="대", description="창고 불가 지역")
    assert not check_use(no, "창고").ok
    assert check_use(STOCK[0], "농사") is None  # 규칙이 없는 목적은 판단하지 않음


# ---------------- 생성형 AI 학습 ----------------
class FakeGPT(LLMClient):
    """취향 분석 요청에만 정해진 JSON 을 답하고, 나머지(조건 추출·설명)는 실패시켜 규칙 기반으로 넘김."""
    name = "fake"
    model = "fake-gpt"

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def complete(self, system, user, max_tokens=2000):
        if "숨은 취향" in system:
            self.calls.append(user)
            return json.dumps(self.plan, ensure_ascii=False)
        raise LLMUnavailable("테스트: 규칙 기반 사용")


def router_with(fake):
    r = LLMRouter(["rule"])
    r.clients = [fake]
    return r


PLAN = {
    "summary": "가격보다 넓은 면적과 계곡 인접 땅을 중시하십니다.",
    "weights": {"area": "up", "price": "down", "nonsense": "up"},
    "prefer_features": ["계곡", "우주정거장"], "avoid_features": ["채광"],
    "prefer_jimok": ["대"], "avoid_jimok": ["임야", "아파트"],
    "add_keywords": ["계곡"], "add_excluded_keywords": ["묘지", "고압선"],
}


def test_sanitize_plan_drops_unknown_values():
    p = sanitize_plan(PLAN)
    assert p["weights"] == {"area": "up", "price": "down"}
    assert p["prefer_features"] == ["계곡"] and p["avoid_jimok"] == ["임야"]


def test_apply_plan_updates_preferences_safely():
    req = CustomerRequirement(regions=["양평군"], keywords=["채광"], raw_text="양평 채광 좋은 땅, 묘지 근처는 싫어요")
    sess = RecommendationSession(STOCK, req, RecommendOptions(pref=PreferenceModel(), min_candidates=1))
    before = sess.opt.pref.weights()
    changes = apply_plan(sess, sanitize_plan(PLAN))
    w = sess.opt.pref.weights()
    assert w["area"] > before["area"] and w["price"] < before["price"]
    assert sess.opt.pref.attrs["feature"]["계곡"] == [1.0, 0.0]
    assert "채광" not in sess.opt.pref.attrs["feature"]  # 손님이 직접 원한 특징은 비선호로 안 배움
    assert "계곡" in req.keywords
    assert req.excluded_keywords == ["묘지"]  # 손님이 말한 '묘지'만 제외, AI 추측 '고압선'은 무시
    assert sess.opt.pref.ai_summary.startswith("가격보다") and changes


def test_agent_reject_uses_ai_and_persists(tmp_path):
    fake = FakeGPT(PLAN)
    settings = Settings(data_dir=SAMPLE, db_path=tmp_path / "a.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    agent = RealEstateAgent(settings, llm=router_with(fake))
    agent.sync()
    sess = agent.recommend(agent.understand("양평 땅", "김손님"))
    agent.reject(sess, "너무 좁아요")
    assert fake.calls and "[거절]" in fake.calls[0] and "이유: 너무 좁아요" in fake.calls[0]
    assert "010-" not in fake.calls[0]  # 연락처는 AI 에 보내지 않음
    assert sess.ai_changes and sess.opt.pref.ai_summary
    assert agent.storage.load_preference("김손님").ai_summary == PLAN["summary"]  # 다음 상담에도 이어짐


def test_without_ai_nothing_breaks():
    assert ask_ai(LLMRouter(["rule"]), "양평 땅", [{"action": "reject", "reason": "", "listing": {}}]) is None


def test_web_shows_ai_summary(tmp_path):
    from real_estate_agent.web.app import create_app

    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    agent = RealEstateAgent(settings, llm=router_with(FakeGPT(PLAN)))
    c = create_app(settings, agent).test_client()

    def tok(path="/"):
        return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)

    c.post("/search", data={"q": "양평 땅", "csrf": tok()})
    html = c.post("/action", data={"act": "next", "text": "너무 좁아요", "csrf": tok("/result")},
                  follow_redirects=True).get_data(as_text=True)
    assert "AI가 파악한 취향" in html and PLAN["summary"] in html


# ---------------- '음성' -> 음성군, '여주' -> 여주시 ----------------
import pytest  # noqa: E402

from real_estate_agent.regions import canonical_region  # noqa: E402


@pytest.mark.parametrize("q,expected", [
    ("음성", ["음성군"]), ("음성 땅", ["음성군"]), ("음성에 창고", ["음성군"]), ("음성창고", ["음성군"]),
    ("음성쪽 전원주택", ["음성군"]), ("음성군 땅", ["음성군"]), ("여주", ["여주시"]), ("여주 땅", ["여주시"]),
    ("여주땅", ["여주시"]), ("여주에 창고 지을 땅", ["여주시"]), ("여주나 양평", ["여주시", "양평군"]),
])
def test_short_region_names(q, expected):
    assert RequirementParser(None, []).parse(q).regions == expected


def test_canonical_region():
    assert canonical_region("음성") == canonical_region("음성시") == canonical_region("음성 군") == "음성군"
    assert canonical_region("여주") == canonical_region("여주군") == "여주시"
    assert canonical_region("감곡면") == "감곡면" and canonical_region("경상북도") == "경북"


class FakeParserGPT(LLMClient):
    name = "fake"

    def complete(self, system, user, max_tokens=2000):
        if "검색 조건을 추출" in system:
            return '{"regions": ["음성", "여주시", "음성시"], "purpose": "창고"}'
        raise LLMUnavailable("x")


def test_gpt_short_region_is_normalized():
    req = RequirementParser(router_with(FakeParserGPT()), []).parse("음성이나 여주에 창고 지을 땅")
    assert req.regions == ["음성군", "여주시"]


def test_web_search_short_names(tmp_path):
    from real_estate_agent.web.app import create_app

    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    c = create_app(settings).test_client()

    def tok():
        return re.search(r'name="csrf" value="([^"]+)"', c.get("/").get_data(as_text=True)).group(1)

    for q, region in [("음성", "음성군"), ("여주", "여주시"), ("음성창고", "음성군")]:
        html = c.post("/search", data={"q": q, "csrf": tok()}, follow_redirects=True).get_data(as_text=True)
        assert f"지역={region}" in html, q
        assert re.search(r'<h2 id="addr">[^<]*' + region, html), q  # 1순위 매물이 그 지역
