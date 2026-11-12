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


@pytest.mark.parametrize("runner_cls", [ToolAgent, MultiAgent])
def test_switch_from_sale_to_lease_factory(agent, runner_cls):
    """'음성군에 매매 공장' → '음성군 말고 다른 지역은?' → '매매 공장 말고 임대 공장은 없어?' 는 임대(매매/임대 포함)만."""
    t, h = runner_cls(agent, ctx=AgentContext(), log_dir=""), None
    no = lambda r: {agent.find_listing(i).office_no for i in r.listing_ids}
    r, h = t.run("음성군에 매매 공장 찾아줘", h)
    assert no(r) == {"F-1", "F-3"}
    r, h = t.run("음성군 말고 다른 지역은 없어?", h)
    assert no(r) == {"F-2"}
    r, h = t.run("매매 공장 말고 임대 공장은 없어?", h)
    assert no(r) == {"F-6"} and all(agent.find_listing(i).for_lease for i in r.listing_ids)
    assert r.answer.startswith("매매 말고 임대(매매/임대 포함) 매물로 다시 찾았습니다.")
    r, h = t.run("임대 말고 매매는?", h)
    assert no(r) == {"F-2"}
    r, h = t.run("공장 말고 땅은 없어?", h)  # 앞 조건(음성군 제외) 그대로 토지로: 이 장부의 토지는 음성군뿐
    search = [s for s in r.steps if s.tool == "search_listings"][0]
    assert search.args.get("property_type") == "토지" and not r.listing_ids
    r, h = t.run("음성군에는?", h)
    assert no(r) == {"L-1", "L-2"}


@pytest.mark.parametrize("q, types, deal", [
    ("매매 공장 말고 임대 공장은 없어?", ["공장"], "임대"),
    ("임대 말고 매매로 된 공장", ["공장"], "매매"),
    ("공장 말고 땅은?", ["토지"], ""),
    ("음성군 공장 매매", ["공장"], "매매"),
])
def test_parser_reads_excluded_deal_and_kind(q, types, deal):
    r = RequirementParser(None, ["음성군"], []).parse(q)
    assert (r.property_types, r.deal) == (types, deal) and not set(r.excluded_keywords) & {"공장", "매매", "임대"}


def _info_agent(tmp_path, sheets):
    """매물 하나 = 폴더 하나 (공장 설명서) 로 매물 폴더를 만들고 읽음."""
    data = tmp_path / "매물"
    for i, rows in enumerate(sheets):
        _sheet(data / "공장" / "임대" / "음성군" / f"대소면 오류리 {100 + i}(공장)" / "설명서.xlsx", rows)
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "i.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    a.sync()
    return sorted(a.listings(), key=lambda l: l.address)


def test_factory_sheet_jimok_zoning_and_won_amounts(tmp_path):
    """설명서의 '지목'·'지역/지구' 를 읽고, 원 단위로 적은 보증금·월세를 만원으로 (6500억원 X → 6,500만원)."""
    a_side = [["소재지", "충청북도 음성군 대소면 오류리 100"],
              ["지목", "대", "지역/지구", "계획관리지역, 성장관리계획구역"],   # 한 줄에 '항목 | 값' 두 쌍
              ["대지면적(평)", 500], ["건물면적(평)", 200], ["보증금", 65000000], ["월세", 6500000]]
    two_rows = [["소재지", "충청북도 음성군 대소면 오류리 101"],
                ["지목", "지역/지구"],                                      # 항목 줄 + 아래 값 줄
                ["전", "계획관리지역\n성장관리계획구역"],
                [None, None],
                ["대지면적(평)", 300], ["보증금", "3,000만원"], ["월세", "250만원"]]
    below = [["소재지", "충청북도 음성군 대소면 오류리 102"],
             ["지목", None, "지역·지구 등", None],                          # 값이 아래 칸에
             ["답", None, "자연녹지지역", None],
             ["대지면적(평)", 300], ["보증금", "5천만원"], ["월세", 3000000]]
    l0, l1, l2 = _info_agent(tmp_path, [a_side, two_rows, below])
    assert (l0.jimok, l0.zoning, l0.extra.get("_지역지구")) == ("대", "계획관리지역", "계획관리지역, 성장관리계획구역")
    assert (l0.deposit_manwon, l0.monthly_rent_manwon) == (6500, 650)
    assert l0.price_text == "임대 보증금 6,500만원 / 월 650만원"
    assert (l1.jimok, l1.zoning, l1.extra.get("_지역지구")) == ("전", "계획관리지역", "계획관리지역, 성장관리계획구역")
    assert (l1.deposit_manwon, l1.monthly_rent_manwon) == (3000, 250)
    assert (l2.jimok, l2.zoning, l2.monthly_rent_manwon) == ("답", "자연녹지지역", 300)

    from real_estate_agent.tool_agent.tools import brief

    b = brief(l0)
    assert b["지목"] == "대" and b["지역지구"] == "계획관리지역, 성장관리계획구역"


def test_lease_only_factories(agent):
    """'음성군에 임대공장' 은 임대 + 매매/임대, '임대만 하는 공장' 은 임대 전용만."""
    no = lambda r: {agent.find_listing(i).office_no for i in r.listing_ids}
    t, h = ToolAgent(agent, ctx=AgentContext(), log_dir=""), None
    r, h = t.run("음성군에 임대공장을 찾아줘.", h)
    assert no(r) == {"F-3", "F-4", "F-5"}
    r, h = t.run("오로지 임대만 하는 공장만 보고 싶어.", h)
    assert no(r) == {"F-4", "F-5"} and "임대 전용" in r.answer
    for q in ("임대만 하는 공장을 찾아줘.", "오로지 임대만 하는 공장만 보고 싶어.", "매매는 안 되는 임대 공장"):
        r, _ = MultiAgent(agent, ctx=AgentContext(), log_dir="").run(q)
        assert no(r) == {"F-4", "F-5", "F-6"}, q  # 매매/임대 F-3 은 빠짐
    r, _ = ToolAgent(agent, ctx=AgentContext(), log_dir="").run("매매만 하는 공장")
    assert no(r) == {"F-1", "F-2"}


@pytest.mark.parametrize("q, deal", [("임대만 하는 공장을 찾아줘", "임대만"), ("오로지 임대만 하는 공장만 보고 싶어", "임대만"),
                                     ("임대 전용 공장", "임대만"), ("음성군에 임대공장을 찾아줘", "임대"),
                                     ("매매만 하는 공장", "매매만"), ("음성군 매매 공장", "매매")])
def test_parser_lease_only(q, deal):
    assert RequirementParser(None, ["음성군"], []).parse(q).deal == deal


def test_combined_jimok_zoning_cell_and_ledger_columns(tmp_path):
    """'지목/용도 | 대 / 계획관리' 한 칸, 장부의 '지목/용도지역'·'지역/지구' 열도 읽음."""
    data = tmp_path / "매물"
    _sheet(data / "공장" / "임대" / "음성군" / "대소면 오류리 9(공장)" / "설명서.xlsx",
           [["공장 매물 설명서"], ["소재지", "충청북도 음성군 대소면 오류리 9"], ["지목/용도", "대 / 계획관리"],
            ["대지면적(평)", 500], ["보증금", 65000000], ["월세", 6500000]])
    _sheet(data / "공장" / "매매" / "진천군" / "공장장부.xlsx",
           [["매물번호", "소재지", "지목/용도지역", "대지면적(평)", "매매가"],
            ["J-1", "충청북도 진천군 덕산읍 용몽리 1", "공장용지/계획관리지역, 성장관리계획구역", 800, "12억"]])
    _sheet(data / "공장" / "매매" / "진천군" / "공장장부2.xlsx",
           [["매물번호", "소재지", "지목", "지역/지구", "대지면적(평)", "매매가"],
            ["J-2", "충청북도 진천군 이월면 신계리 2", "전", "계획관리지역", 600, "8억"]])
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "c.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    a.sync()
    got = {l.office_no or l.address.split()[-1]: (l.jimok, l.zoning, l.extra.get("_지역지구", "")) for l in a.listings()}
    assert got == {"9": ("대", "계획관리지역", ""), "J-1": ("공장용지", "계획관리지역", "계획관리지역, 성장관리계획구역"),
                   "J-2": ("전", "계획관리지역", "")}


def test_sheet_layout_hides_private_values(tmp_path):
    from real_estate_agent.sheet_layout import layout_lines

    p = tmp_path / "설명서.xlsx"
    _sheet(p, [["소재지", "충청북도 음성군 대소면 오류리 9", "소유자", "홍길동"], ["연락처", "010-1234-5678"],
               ["성명"], ["김철수"], ["지역/지구", "계획관리지역"], ["지목", "대"]])
    text = "\n".join(layout_lines(p))
    for secret in ("홍길동", "010", "오류리", "김철수"):
        assert secret not in text
    assert "지역/지구✓" in text and "계획관리지역" in text and "지목✓ | 대" in text


@pytest.mark.parametrize("q", ["임대 공장만 보여줘.", "임대 매물만 보여줘", "임대 공장만"])
def test_parser_lease_factory_only(q):
    assert RequirementParser(None, ["음성군"], []).parse(q).deal == "임대만"


def test_cli_sheet_layout_and_other_commands_run(tmp_path, capsys):
    """명령 하나에서 쓰는 import 가 다른 명령을 망가뜨리지 않는지 (sheet-layout, benchmark 경로)."""
    from real_estate_agent.cli import main

    data = tmp_path / "매물"
    _sheet(data / "공장" / "임대" / "음성군" / "장부.xlsx",
           [["매물번호", "소재지", "지목", "대지면적(평)", "보증금", "월세"], ["Z-1", "충청북도 음성군 대소면 오류리 9", "대", 300, "3천만", "200만"]])
    args = ["--no-llm", "--data-dir", str(data), "--db", str(tmp_path / "c.db")]
    assert main(args + ["sheet-layout"]) == 0
    assert "지목 또는 용도지역이 비어 있는 매물 1건" in capsys.readouterr().out
    assert main(args + ["organize-folders", "--dry-run"]) == 0
