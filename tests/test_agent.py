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


# ---------------- '2억원대' / '채광' / 토지 외 매물 ----------------
def test_parse_price_band_and_sunlight():
    p = RequirementParser(None, ["음성군", "금왕읍"])
    req = p.parse("음성군채광이 좋은 2억원대 아파트를 보여줘")
    assert req.regions == ["음성군"]
    assert (req.budget_min, req.budget_max) == (20000, 29999)
    assert "채광" in req.keywords
    assert req.notes and "아파트" in req.notes[0]
    # 보유 매물에 없는 군도 붙여쓰기로 인식
    assert parse_rule_based("괴산군햇빛 잘 드는 밭").regions == ["괴산군"]
    req = parse_rule_based("200평대 1억 5천만원대")
    assert (req.budget_min, req.budget_max) == (15000, 15999)
    assert req.area_min_m2 == pytest.approx(200 * 3.305785)
    assert req.area_max_m2 == pytest.approx(299 * 3.305785)


def test_sunlight_synonyms_match_listing():
    req = CustomerRequirement(keywords=["채광"])
    sunny = L("S", "충북 음성군 a리 1", 661, 20000, description="남동향 완경사, 햇빛 잘 듦")
    shady = L("N", "충북 음성군 a리 2", 661, 20000, description="북향, 오후 그늘")
    ranked, _, _ = rank([shady, sunny], req, today=TODAY)
    assert ranked[0].listing.listing_id == "S"
    assert ranked[0].factors["keyword"] == 1.0 and ranked[1].factors["keyword"] == 0.0


# ---------------- 피드백 학습 ----------------
from real_estate_agent.preference import PreferenceModel  # noqa: E402


def test_reason_raises_weight():
    m = PreferenceModel()
    before = m.weights()["price"]
    changes = m.learn("reject", {"price": 0.8}, [], reason="너무 비싸요")
    assert m.weights()["price"] > before
    assert "가격 중요도 ↑" in changes


def test_requested_attribute_is_not_disliked():
    m = PreferenceModel()
    m.learn("reject", {}, [], jimok="대", text="채광 우수", protected={"채광"})
    assert "채광" not in m.attrs["feature"]  # 손님이 원한 특징은 비선호로 배우지 않음
    assert m.attrs["jimok"]["대"] == [0.0, 0.5]  # 이유 모르는 거절은 약하게
    m.learn("reject", {}, [], jimok="대", text="", reason="너무 비싸요")
    assert m.attrs["jimok"]["대"] == [0.0, 0.5]  # 가격 때문이면 지목은 탓하지 않음


def test_learning_changes_next_rank():
    """싸지만 좁은 매물을 계속 거절하면 면적 중요도가 올라 넓은 매물이 먼저 나온다."""
    listings = [
        L("cheap1", "경기도 양평군 a리 1", 490, 9000, road_access="도로"),
        L("cheap2", "경기도 양평군 a리 2", 485, 9500, road_access="도로"),
        L("cheap3", "경기도 양평군 a리 3", 480, 9800, road_access="도로"),
        L("wide", "경기도 양평군 a리 4", 700, 19500, road_access="도로"),
    ]
    req = CustomerRequirement(budget_max=20000, area_min_m2=500)
    plain = RecommendationSession(listings, req, RecommendOptions(min_candidates=1))
    learn = RecommendationSession(listings, req, RecommendOptions(min_candidates=1, pref=PreferenceModel()))
    assert plain.current().listing.listing_id == learn.current().listing.listing_id == "cheap1"
    assert plain.next("너무 좁아요").listing.listing_id == "cheap2"  # 학습 없으면 원래 2순위
    assert learn.next("너무 좁아요").listing.listing_id == "wide"  # 학습하면 넓은 매물이 2순위로
    assert learn.cursor == 1 and learn.reordered
    assert any("면적" in c for c in learn.last_learning)


def test_learning_persists_for_customer(tmp_path):
    settings = Settings(data_dir=SAMPLE, db_path=tmp_path / "p.db", log_dir=tmp_path / "logs", llm_providers=["rule"])
    agent = RealEstateAgent(settings)
    agent.sync()
    req = agent.understand("음성군 채광 좋은 2억원대 땅", "최지우")
    sess = agent.recommend(req)
    agent.reject(sess, "너무 비싸요")
    agent.like(sess)
    saved = agent.storage.load_preference("최지우")
    assert saved.n_feedback == 2
    assert saved.weights()["price"] > 25
    assert agent.storage.load_preference("__office__").n_feedback == 2
    # 다음 상담에서 학습한 취향이 그대로 적용됨
    sess2 = agent.recommend(agent.understand("음성군 땅", "최지우"))
    assert sess2.opt.pref.n_feedback == 2


def test_reason_only_blames_mentioned_features():
    m = PreferenceModel()
    m.learn("reject", {}, [], text="정남향 평지, 채광 우수, 금왕IC 5분", reason="채광이 별로예요",
            protected={"채광"})
    assert m.attrs["feature"] == {}  # 남향(채광 동의어)·평지·IC 모두 비선호로 배우지 않음
    m.learn("reject", {}, [], text="계곡 인접, 평지", reason="계곡이 싫어요")
    assert m.attrs["feature"] == {"계곡": [0.0, 1.0]}


def test_input_template_roundtrip(tmp_path):
    import io as _io

    from openpyxl import load_workbook

    from real_estate_agent.template import build_template

    folder = tmp_path / "토지"
    folder.mkdir()
    blank = folder / "빈양식.xlsx"
    blank.write_bytes(build_template())
    assert LocalFolderCollector(folder).collect() == []  # 작성방법·목록 시트는 매물로 읽지 않음

    wb = load_workbook(_io.BytesIO(build_template()))
    ws = wb["매물입력"]
    ws.append(["E-9001", "충청북도 음성군 금왕읍 용계리 77-1", "대", "계획관리지역", 250, "2억 3천만원",
               "6m 포장도로 접함", "거래가능", "2026-10-12", "정남향, 햇빛 잘 듦", "010-1234-5678"])
    blank.unlink()
    wb.save(folder / "새매물.xlsx")
    result = clean(LocalFolderCollector(folder).collect(), TODAY)
    assert result.error_count == 0 and len(result.listings) == 1
    l = result.listings[0]
    assert l.area_m2 == pytest.approx(250 * 3.305785, rel=1e-3) and l.price_manwon == 23000
    req = RequirementParser(None, ["음성군"]).parse("음성군 채광이 좋은 2억원대 땅")
    ranked, _, _ = rank(result.listings, req, today=TODAY)
    assert ranked[0].listing.listing_id == "E-9001" and ranked[0].factors["keyword"] == 1.0
