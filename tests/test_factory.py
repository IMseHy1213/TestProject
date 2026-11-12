"""공장 매물: 매물/공장/매매|매매임대|임대/음성군|진천군/ 폴더를 읽고, 상담에서 찾을 수 있어야 함."""

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.cleaner import validate
from real_estate_agent.config import SAMPLE_DATA_DIR, Settings
from real_estate_agent.requirement_parser import RequirementParser
from real_estate_agent.tool_agent import AgentContext, ToolAgent
from real_estate_agent.tool_agent.multi import MultiAgent


def _sheet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """실제 사무실처럼: 토지는 지역 폴더, 공장은 공장/거래방식/지역 폴더 (장부 + 매물 하나 = 폴더 하나 섞어서)."""
    tmp = tmp_path_factory.mktemp("factory")
    data = tmp / "매물"
    land_head = ["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "도로", "특이사항"]
    _sheet(data / "토지" / "음성군" / "토지.xlsx", [land_head,
           ["L-1", "충청북도 음성군 대소면 오류리 1", "잡종지", "계획관리지역", 600, "3억", "6m 도로 접함", "창고 부지"],
           ["L-2", "충청북도 음성군 금왕읍 무극리 2", "공장용지", "계획관리지역", 900, "5억", "8m 도로 접함", "공장 부지"]])
    head = ["매물번호", "소재지", "지목", "용도지역", "대지면적(평)", "건물면적(평)", "매매가", "보증금", "월세", "특이사항"]
    _sheet(data / "공장" / "매매" / "음성군" / "장부.xlsx", [head,
           ["F-1", "충청북도 음성군 대소면 삼정리 10", "공장용지", "계획관리지역", 1000, 500, "15억", None, None, "호이스트 5톤"]])
    _sheet(data / "공장" / "매매" / "진천군" / "장부.xlsx", [head,
           ["F-2", "충청북도 진천군 덕산읍 용몽리 20", "공장용지", "일반공업지역", 2000, 800, "30억", None, None, "층고 10m"]])
    _sheet(data / "공장" / "매매임대" / "음성군" / "장부.xlsx", [head,
           ["F-3", "충청북도 음성군 맹동면 봉현리 30", "공장용지", "계획관리지역", 700, 300, "10억", "1억", "600만", "대형차 진입"]])
    _sheet(data / "공장" / "임대" / "음성군" / "장부.xlsx", [head,
           ["F-4", "충청북도 음성군 삼성면 대사리 40", "공장용지", "계획관리지역", 400, 150, None, "3천만", "250만", "호이스트"],
           ["F-5", "충청북도 음성군 감곡면 오향리 50", "공장용지", "계획관리지역", 500, 250, None, "5천만", "400만", "층고 8m"]])
    # 매물 하나 = 폴더 하나 (물건 정보지 양식)
    _sheet(data / "공장" / "임대" / "진천군" / "이월면 신계리 60(공장)" / "정보.xlsx", [
        ["매물번호", "F-6"], ["소재지", "충청북도 진천군 이월면 신계리 60"], ["대지면적(평)", 800], ["건물면적(평)", 400],
        ["보증금", "8천만"], ["월세", "500만"], ["지목", "공장용지"], ["매물특징", "● 호이스트 3톤\n● 층고 9m"]])
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp / "f.db", log_dir=tmp / "l", llm_providers=["rule"]))
    a.sync()
    return a


def by_no(agent):
    return {l.office_no: l for l in agent.listings()}


def test_folders_give_kind_and_deal(agent):
    ls = by_no(agent)
    assert set(ls) == {"L-1", "L-2", "F-1", "F-2", "F-3", "F-4", "F-5", "F-6"}
    assert {k for k, l in ls.items() if l.property_type == "공장"} == {"F-1", "F-2", "F-3", "F-4", "F-5", "F-6"}
    assert ls["L-1"].property_type == "토지" and ls["L-1"].deal_type == "매매"
    assert ls["F-1"].deal_type == "매매" and ls["F-3"].deal_type == "매매/임대" and ls["F-4"].deal_type == "임대"
    f6 = ls["F-6"]  # 폴더 하나 = 매물 하나 양식도
    assert f6.deal_type == "임대" and f6.deposit_manwon == 8000 and f6.monthly_rent_manwon == 500
    assert round(f6.building_area_pyeong) == 400 and round(f6.area_pyeong) == 800
    assert ls["F-4"].price_text == "임대 보증금 3,000만원 / 월 250만원"
    assert ls["F-3"].price_text == "매매 10억원 · 임대 보증금 1억원 / 월 600만원"
    assert validate(ls["F-4"])[0] == []  # 매매가 없는 임대 공장도 정상 매물


@pytest.mark.parametrize("q, types, deal, purpose", [
    ("음성군 공장 있어?", ["공장"], "", ""),
    ("공장 지을 땅 있어?", ["토지"], "", "공장"),
    ("공장 부지 3억 이하", ["토지"], "", "공장"),
    ("공장 임대 보증금 5천 월세 300 이하", ["공장"], "임대", ""),
    ("진천 공장 매매", ["공장"], "매매", ""),
    ("음성군에 창고 지을 땅 있어?", ["토지"], "", "창고"),
])
def test_parser_tells_factory_from_land_for_factory(q, types, deal, purpose):
    r = RequirementParser(None, ["음성군", "진천군"], []).parse(q)
    assert (r.property_types, r.deal, r.purpose) == (types, deal, purpose)


def test_parser_reads_deposit_and_rent_not_as_budget():
    r = RequirementParser(None, ["음성군"], []).parse("음성군 공장 임대 보증금 5천에 월세 300만원 이하")
    assert (r.deposit_max, r.rent_max, r.budget_max) == (5000, 300, None)


@pytest.mark.parametrize("runner_cls", [ToolAgent, MultiAgent])
def test_factory_consultation(agent, runner_cls):
    t = runner_cls(agent, ctx=AgentContext(), log_dir="")
    ls = by_no(agent)
    kinds = lambda ids: {ls_id.property_type for ls_id in (agent.find_listing(i) for i in ids)}

    r, h = t.run("음성군 공장 있어?")
    assert r.listing_ids and kinds(r.listing_ids) == {"공장"}
    assert {agent.find_listing(i).office_no for i in r.listing_ids} == {"F-1", "F-3", "F-4", "F-5"}
    r, h = t.run("다른 지역에는 없어?", h)
    assert {agent.find_listing(i).office_no for i in r.listing_ids} == {"F-2", "F-6"}  # 진천군 공장만

    r, _ = runner_cls(agent, ctx=AgentContext(), log_dir="").run("공장 임대 월세 400 이하")
    nos = {agent.find_listing(i).office_no for i in r.listing_ids}
    assert nos == {"F-4", "F-5"}  # 임대 가능 + 월세 400만원 이하 (매매 전용 F-1·F-2 와 월세 비싼 F-3·F-6 제외)
    assert "월" in r.answer and "보증금" in r.answer

    r, _ = runner_cls(agent, ctx=AgentContext(), log_dir="").run("공장 매매 20억 이하")
    assert {agent.find_listing(i).office_no for i in r.listing_ids} == {"F-1", "F-3"}

    r, _ = runner_cls(agent, ctx=AgentContext(), log_dir="").run("공장 지을 땅 있어?")
    assert r.listing_ids and kinds(r.listing_ids) == {"토지"}


def test_factory_area_matches_building_area(agent):
    t = ToolAgent(agent, ctx=AgentContext(), log_dir="")
    r, _ = t.run("건물 300평 정도 공장")  # 건물 240~360평: F-3(300평)·F-5(250평)
    nos = {agent.find_listing(i).office_no for i in r.listing_ids}
    assert {"F-3", "F-5"} <= nos and "F-2" not in nos


def test_region_counts_by_kind(agent):
    t = ToolAgent(agent, ctx=AgentContext(), log_dir="")
    r, _ = t.run("공장 매물은 어느 지역이 제일 많아요?")
    assert r.answer.startswith("보유 공장 매물 6건 중 음성군이 4건으로 가장 많습니다.")
    assert not r.listing_ids


def test_customer_card_hides_contact_and_shows_lease(agent):
    from real_estate_agent.tool_agent.tools import brief

    b = brief(by_no(agent)["F-4"])
    assert b["종류"] == "공장" and b["거래"] == "임대" and b["월세_만원"] == 250 and b["주소"].endswith("***")
    assert "가격_만원" not in b and "평당가_만원" not in b


def test_sample_data_has_practice_factories(tmp_path):
    a = RealEstateAgent(Settings(data_dir=SAMPLE_DATA_DIR, db_path=tmp_path / "s.db", log_dir=tmp_path / "l",
                                 llm_providers=["rule"]))
    a.sync()
    factories = [l for l in a.listings() if l.property_type == "공장"]
    assert len(factories) == 8 and {l.deal_type for l in factories} == {"매매", "매매/임대", "임대"}


def test_factory_over_mcp_and_a2a(tmp_path):
    """공장 조건(종류·임대·월세)이 MCP 도구와 분산 상담팀(A2A)에서도 똑같이 전달됨."""
    from fastapi.testclient import TestClient

    from real_estate_agent.a2a.distributed import DistributedMultiAgent
    from real_estate_agent.a2a.server import create_a2a_app
    from real_estate_agent.mcp_server import create_mcp_server
    from real_estate_agent.tool_agent.mcp_client import McpToolSource
    from real_estate_agent.tool_agent.multi import ANALYST, SEARCHER

    def make(name):
        a = RealEstateAgent(Settings(data_dir=SAMPLE_DATA_DIR, db_path=tmp_path / name, log_dir=tmp_path / "l",
                                     llm_providers=["rule"]))
        a.sync()
        return a

    web = make("w.db")
    questions = ["음성군 공장 임대 월세 300 이하", "다른 지역에는 없어?"]
    expected, h = [], None
    local = ToolAgent(web, ctx=AgentContext(), log_dir="")
    for q in questions:
        r, h = local.run(q, h)
        expected.append(r.listing_ids)
    assert expected[0] and all(web.find_listing(i).for_lease for i in expected[0])

    try:
        from mcp import Client  # noqa: F401
    except ImportError:
        pytest.skip("mcp 2.x 필요")
    with McpToolSource(create_mcp_server(web)) as src:
        t, h = ToolAgent(web, ctx=AgentContext(), tool_source=src), None
        got = []
        for q in questions:
            r, h = t.run(q, h)
            got.append(r.listing_ids)
    assert got == expected

    clients = {SEARCHER: TestClient(create_a2a_app("search", agent=make("s.db"), token="")),
               ANALYST: TestClient(create_a2a_app("analysis", agent=make("a.db"), token=""))}
    from real_estate_agent.a2a.client import A2AClient

    dist = DistributedMultiAgent(web, clients={k: A2AClient("http://testserver/", http=c) for k, c in clients.items()},
                                 ctx=AgentContext())
    local_multi, hd, hl = MultiAgent(web, ctx=AgentContext()), [], []
    for q in questions:
        rd, hd = dist.run(q, hd)
        rl, hl = local_multi.run(q, hl)
        assert rd.listing_ids == rl.listing_ids


def test_web_pages_show_factory_cards(tmp_path, monkeypatch):
    """손님 화면: 공장 표시·임대 조건·건물면적이 보이고 연락처·지번은 안 보임."""
    import re
    import shutil

    from real_estate_agent.web.app import create_app

    monkeypatch.delenv("REA_ADMIN_PASSWORD", raising=False)
    data = tmp_path / "매물"
    shutil.copytree(SAMPLE_DATA_DIR, data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                              llm_providers=["rule"]))
    app.config["TESTING"] = True
    c = app.test_client()

    def token(path):
        return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)

    html = c.post("/agent/ask", data={"q": "음성군 공장 임대 있어?", "csrf": token("/agent")},
                  follow_redirects=True).get_data(as_text=True)
    assert "공장(임대)" in html and "보증금" in html and "건물면적" in html
    assert "010-5101" not in html and "대사리 130" not in html
    html = c.post("/search", data={"q": "진천군 공장 매매", "csrf": token("/")},
                  follow_redirects=True).get_data(as_text=True)
    assert "공장(매매" in html and "건물면적" in html and "010-5101" not in html
