from datetime import date
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.cleaner import clean, is_duplicate
from real_estate_agent.collectors import LocalFolderCollector
from real_estate_agent.collectors.base import RawRecord
from real_estate_agent.config import Settings
from real_estate_agent.llm.base import LLMClient, LLMUnavailable
from real_estate_agent.llm.router import LLMRouter
from real_estate_agent.models import CustomerRequirement, LandListing
from real_estate_agent.normalizer import normalize_zoning, parse_area_m2, parse_price_manwon
from real_estate_agent.recommender import RecommendationSession, RecommendOptions, rank
from real_estate_agent.requirement_parser import RequirementParser, parse_rule_based
from real_estate_agent.storage import Storage

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
TODAY = date(2026, 10, 12)


def L(id, addr, m2, price, **kw) -> LandListing:
    return LandListing(listing_id=id, address=addr, area_m2=m2, price_manwon=price, **kw)


# ---------------- 정규화 ----------------
@pytest.mark.parametrize("raw,expected", [
    ("3억 2천만원", 32000), ("3.2억", 32000), ("32,000", 32000), ("3억2000", 32000),
    ("2억 8000", 28000), ("5천만원", 5000), ("320000000", 32000), (28000, 28000), ("", None),
])
def test_parse_price(raw, expected):
    assert parse_price_manwon(raw) == expected


def test_parse_area():
    assert parse_area_m2("100평") == pytest.approx(330.5785)
    assert parse_area_m2("661㎡") == 661
    assert parse_area_m2(100, assume_pyeong=True) == pytest.approx(330.5785)


def test_normalize_zoning():
    assert normalize_zoning("계획관리") == "계획관리지역"
    assert normalize_zoning("2종일반주거") == "제2종일반주거지역"


# ---------------- 수집·정제 ----------------
def test_collect_sample_folder():
    c = LocalFolderCollector(SAMPLE)
    records = c.collect()
    assert len(c.files_read) == 4 and not c.errors
    sources = {r.source for r in records}
    assert {"local", "naver", "serve"} <= sources


def test_clean_filters_errors_and_duplicates():
    result = clean(LocalFolderCollector(SAMPLE).collect(), TODAY)
    ids = {l.listing_id for l in result.listings}
    assert result.error_count == 3  # 주소 누락, 면적 누락, 평당가 비현실
    assert result.duplicate_count == 3
    assert "T-1010" not in ids  # 가격 단위 오류
    # 목왕리 123-4 는 장부와 네이버에 모두 있음 -> 최신(네이버) 하나만 남고 출처 기록
    mokwang = [l for l in result.listings if "목왕리 123-4" in l.address]
    assert len(mokwang) == 1
    assert set(mokwang[0].extra["duplicate_sources"]) == {"local", "naver"}


def test_duplicate_with_different_notation():
    a = L("a", "경기도 양평군 양서면 목왕리 123-4", 661, 32000)
    b = L("b", "경기 양평군 양서면 목왕리 123-4번지", 661.5, 31500)
    c = L("c", "경기도 양평군 양서면 목왕리 123-5", 661, 32000)
    assert is_duplicate(a, b)
    assert not is_duplicate(a, c)


def test_validation_errors():
    recs = [
        RawRecord({"소재지": "경기도 양평군 a리 1", "면적": "100", "매매가": "1억"}, "local"),
        RawRecord({"소재지": "경기도 양평군 a리 2", "면적": "1000", "매매가": 10}, "local"),  # 평당 0.03만원
        RawRecord({"소재지": "", "면적": "1000", "매매가": "1억"}, "local"),
    ]
    r = clean(recs, TODAY)
    assert len(r.listings) == 1 and r.error_count == 2


# ---------------- 요구사항 이해 ----------------
def test_rule_parser():
    req = parse_rule_based("김철수 사장님, 양평이나 여주 쪽에 전원주택 지을 땅 200평 정도, 3억 이하, 맹지 제외, 남향",
                           ["양평군", "여주시", "서종면"])
    assert req.customer_name == "김철수"
    assert set(req.regions) == {"양평군", "여주시"}
    assert req.budget_max == 30000
    assert req.area_min_m2 == pytest.approx(200 * 3.305785 * 0.8)
    assert req.purpose == "전원주택" and req.need_road
    assert "남향" in req.keywords


def test_rule_parser_range_and_exclusion():
    req = parse_rule_based("논이나 밭 300~500평, 1억~2억, 임야는 제외")
    assert set(req.jimok) == {"답", "전"}
    assert (req.budget_min, req.budget_max) == (10000, 20000)
    assert req.area_max_m2 == pytest.approx(500 * 3.305785)
    assert "임야" in req.excluded_keywords


class FakeLLM(LLMClient):
    name = "fake"
    model = "fake-1"

    def __init__(self, response: str | None):
        self.response = response

    def complete(self, system, user, max_tokens=2000):
        if self.response is None:
            raise LLMUnavailable("down")
        return self.response


def _router(*clients) -> LLMRouter:
    r = LLMRouter(["rule"])
    r.clients = list(clients)
    return r


def test_llm_parser_and_sanitize():
    llm = _router(FakeLLM('```json\n{"regions": "양평군", "jimok": ["밭", "대"], "budget_max": 30000, '
                          '"zoning": ["계획관리"], "purpose": "집 지을 땅", "need_road": true}\n```'))
    req = RequirementParser(llm).parse("아무 문의")
    assert req.regions == ["양평군"]
    assert req.jimok == ["전", "대"]  # 별칭 정규화
    assert req.zoning == ["계획관리지역"]
    assert req.purpose == "전원주택"
    assert llm.last_used == "fake"


def test_llm_fallback_chain():
    llm = _router(FakeLLM(None), FakeLLM('{"budget_max": 12345}'))
    assert RequirementParser(llm).parse("x").budget_max == 12345
    # 전부 실패하면 규칙 기반
    llm = _router(FakeLLM(None))
    req = RequirementParser(llm).parse("2억 이하 임야")
    assert req.budget_max == 20000 and req.jimok == ["임야"] and llm.last_used == "rule"


# ---------------- 추천 ----------------
LISTINGS = [
    L("A", "경기도 양평군 서종면 문호리 1", 661, 28000, jimok="대", zoning="계획관리지역", road_access="4m 도로 접함"),
    L("B", "경기도 양평군 서종면 문호리 2", 661, 25000, jimok="전", zoning="계획관리지역", road_access="맹지"),
    L("C", "경기도 양평군 양서면 목왕리 3", 700, 34000, jimok="대", zoning="계획관리지역", road_access="6m 도로"),
    L("D", "경기도 여주시 대신면 천서리 4", 661, 20000, jimok="대", zoning="계획관리지역", road_access="4m 도로"),
    L("E", "경기도 양평군 양서면 목왕리 5", 650, 22000, jimok="전", zoning="생산관리지역", road_access="4m 도로",
      status="계약완료"),
    L("F", "경기도 양평군 단월면 보룡리 6", 600, 21000, jimok="전", zoning="계획관리지역", road_access="3m 도로"),
]


def test_rank_hard_filters_and_order():
    req = CustomerRequirement(regions=["양평군"], budget_max=30000, area_min_m2=500, purpose="전원주택", need_road=True)
    ranked, relaxed, stats = rank(LISTINGS, req, RecommendOptions(min_candidates=2), today=TODAY)
    ids = [s.listing.listing_id for s in ranked]
    assert ids[0] == "A"  # 대 + 계획관리 + 도로
    assert "B" not in ids  # 맹지
    assert "C" not in ids  # 예산 13% 초과
    assert "D" not in ids  # 지역
    assert "E" not in ids  # 계약완료
    assert not relaxed
    assert stats["맹지"] == 1 and stats["예산 초과"] == 1


def test_auto_relax_when_few_candidates():
    req = CustomerRequirement(regions=["여주시"], budget_max=30000, purpose="전원주택", need_road=True)
    ranked, relaxed, _ = rank(LISTINGS, req, today=TODAY)
    assert relaxed and len(ranked) >= 3
    assert ranked[0].listing.listing_id == "D"  # 지역 일치 매물이 여전히 1위


def test_session_next_and_feedback():
    req = CustomerRequirement(regions=["양평군"], budget_max=30000, need_road=True)
    sess = RecommendationSession(LISTINGS, req)
    first = sess.current().listing.listing_id
    second = sess.next("별로").listing.listing_id
    assert first != second and first in sess.rejected
    sess.update_requirement(CustomerRequirement(regions=["양평군"], budget_max=40000, need_road=True))
    assert first not in [s.listing.listing_id for s in sess.ranked]  # 거절한 매물은 다시 안 나옴


# ---------------- 에이전트 통합 ----------------
def test_agent_end_to_end(tmp_path):
    settings = Settings(data_dir=SAMPLE, db_path=tmp_path / "a.db", log_dir=tmp_path / "logs", llm_providers=["rule"])
    agent = RealEstateAgent(settings)
    report = agent.sync()
    assert len(report.new) == len(report.clean.listings) > 10
    assert (tmp_path / "logs" / "quality_report.txt").exists()

    req = agent.understand("양평 전원주택 부지 200평 정도 3억 이하 도로 접한 곳", "홍길동")
    sess = agent.recommend(req)
    assert sess.total > 0
    rejected = sess.current().listing.listing_id
    agent.reject(sess, "비쌈")
    agent.save_customer(sess)

    # 2회차 sync: 변화 없음
    assert not agent.sync().new
    alerts = agent.proactive_scan(threshold=0)
    assert alerts and all(a.customer == "홍길동" for a in alerts)
    assert rejected not in {a.scored.listing.listing_id for a in alerts}
    # 같은 매물은 다시 알리지 않음
    assert agent.proactive_scan(threshold=0) == []


def test_storage_detects_changes(tmp_path):
    st = Storage(tmp_path / "s.db")
    a = L("A", "경기도 양평군 x리 1", 500, 10000)
    assert len(st.upsert_listings([a])["new"]) == 1
    a.price_manwon = 9000
    diff = st.upsert_listings([a])
    assert len(diff["changed"]) == 1 and not diff["new"]
    assert st.upsert_listings([])["removed_ids"] == ["A"]
