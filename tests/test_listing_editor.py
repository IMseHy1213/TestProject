"""관리자 매물 추가·수정 테스트."""

import re
import shutil
from pathlib import Path

import pytest
from openpyxl import load_workbook

from real_estate_agent.config import Settings
from real_estate_agent.listing_editor import WEB_FILE, WEB_FOLDER
from real_estate_agent.web.app import create_app

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"

NEW = {"address": "충청북도 음성군 감곡면 오향리 777-1", "jimok": "대", "zoning": "계획관리지역",
       "area": "300", "area_unit": "pyeong", "price": "1억 5천만원", "road_access": "6m 포장도로 접함",
       "status": "거래가능", "registered_date": "2026-10-01", "description": "정남향 평지, 채광 우수",
       "owner_contact": "010-1234-5678"}


@pytest.fixture
def admin(tmp_path, monkeypatch):
    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs", llm_providers=["rule"])
    app = create_app(settings)
    app.config["TESTING"] = True
    c = app.test_client()
    c.post("/admin/login", data={"password": "office-pass-1", "csrf": token(c, "/admin")})
    return c, app.config["AGENT"], data, settings


def token(c, path="/admin/listings"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_listing_pages_need_login(tmp_path, monkeypatch):
    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    settings = Settings(data_dir=SAMPLE, db_path=tmp_path / "w.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    c = create_app(settings).test_client()
    assert c.get("/admin/listings").status_code == 302
    assert c.get("/admin/listings/new").status_code == 302
    t = token(c, "/")
    assert c.post("/admin/listings/new", data={**NEW, "csrf": t}).status_code == 302
    assert c.post("/admin/listings/state", data={"id": "x", "op": "hide", "csrf": t}).status_code == 403


def test_add_new_listing_goes_to_excel_and_search(admin):
    c, agent, data, _ = admin
    before = len(agent.listings())
    r = c.post("/admin/listings/new", data={**NEW, "csrf": token(c)}, follow_redirects=True)
    html = r.get_data(as_text=True)
    assert "새 매물을 추가했습니다" in html and "오향리 777-1" in html
    assert len(agent.listings()) == before + 1
    path = data / WEB_FOLDER / WEB_FILE
    ws = load_workbook(path).active
    assert ws.max_row == 2 and "오향리 777-1" in str(ws.cell(row=2, column=2).value)
    new = next(l for l in agent.listings() if "오향리 777-1" in l.address)
    assert new.price_manwon == 15000 and round(new.area_pyeong) == 300 and new.listing_id.startswith("W-")

    agent.sync()  # 프로그램을 다시 켜도(폴더를 다시 읽어도) 그대로 있음
    assert any("오향리 777-1" in l.address for l in agent.listings())

    # 같은 땅을 또 넣으면 막음
    r = c.post("/admin/listings/new", data={**NEW, "csrf": token(c)})
    assert r.status_code == 400 and "이미 같은 매물" in r.get_data(as_text=True)


def test_new_listing_validation(admin):
    c, agent, _, _ = admin
    bad = {**NEW, "address": "", "area": "abc", "price": ""}
    r = c.post("/admin/listings/new", data={**bad, "csrf": token(c)})
    html = r.get_data(as_text=True)
    assert r.status_code == 400 and "소재지를 입력" in html and "면적은 숫자" in html and "매매가를 입력" in html
    r = c.post("/admin/listings/new", data={**NEW, "price": "1", "csrf": token(c)})  # 평당 0.003만원
    assert r.status_code == 400 and "평당가" in r.get_data(as_text=True)
    r = c.post("/admin/listings/new", data={**NEW, "description": "=HYPERLINK(\"x\")", "csrf": token(c)})
    assert r.status_code == 302  # 수식처럼 보이는 글자는 엑셀에 글자로 저장
    ws = load_workbook(Path(agent.settings.data_dir) / WEB_FOLDER / WEB_FILE).active
    assert str(ws.cell(row=2, column=10).value).startswith("'=")


def test_edit_existing_listing_overrides_without_touching_excel(admin):
    c, agent, data, _ = admin
    target = sorted(agent.listings(), key=lambda l: l.listing_id)[0]
    lid, old_price = target.listing_id, target.price_manwon
    excel_before = {p: p.read_bytes() for p in data.rglob("*.xlsx")}

    page = c.get(f"/admin/listings/edit?id={lid}").get_data(as_text=True)
    assert target.address in page
    form = dict(re.findall(r'name="(\w+)" value="([^"]*)"', page))
    form.update(price=str(old_price - 1000), status="보류", description="가격 조정됨, 급매",
                jimok=target.jimok, area_unit="pyeong", csrf=token(c))
    r = c.post("/admin/listings/edit", data=form, follow_redirects=True)
    assert "수정했습니다" in r.get_data(as_text=True)
    l = agent.find_listing(lid)
    assert l.price_manwon == old_price - 1000 and l.status == "보류" and "급매" in l.description
    assert {p: p.read_bytes() for p in data.rglob("*.xlsx")} == excel_before  # 원본 엑셀은 그대로

    agent.sync()  # 폴더를 다시 읽어도 고친 내용 유지
    l = agent.find_listing(lid)
    assert l.price_manwon == old_price - 1000 and l.extra.get("관리자수정")
    page = c.get(f"/admin/listings/edit?id={lid}").get_data(as_text=True)
    assert "지금까지 고친 내용" in page and "매매가" in page

    c.post("/admin/listings/state", data={"id": lid, "op": "revert", "csrf": token(c)})
    assert agent.find_listing(lid).price_manwon == old_price
    agent.sync()
    assert agent.find_listing(lid).price_manwon == old_price and not agent.storage.listing_edits()


def test_hide_and_show_listing(admin):
    c, agent, _, _ = admin
    lid = agent.listings()[0].listing_id
    n = len(agent.listings())
    c.post("/admin/listings/state", data={"id": lid, "op": "hide", "csrf": token(c)})
    assert len(agent.listings()) == n - 1 and agent.find_listing(lid) is not None
    html = c.get("/admin/listings?show=hidden").get_data(as_text=True)
    assert lid in html and "숨김" in html
    agent.sync()
    assert all(l.listing_id != lid for l in agent.listings())  # 다시 읽어도 숨김 유지
    c.post("/admin/listings/state", data={"id": lid, "op": "show", "csrf": token(c)})
    assert any(l.listing_id == lid for l in agent.listings()) and len(agent.listings()) == n


def test_list_search_and_edit_link_on_result(admin):
    c, agent, _, _ = admin
    some = agent.listings()[0]
    word = some.address.split()[-2]
    html = c.get(f"/admin/listings?q={word}").get_data(as_text=True)
    assert some.listing_id in html and "새 매물 추가" in html
    r = c.post("/search", data={"q": "음성군 2억원대 땅", "csrf": token(c, "/")}, follow_redirects=True)
    assert "이 매물 수정" in r.get_data(as_text=True)


FACTORY_LEASE = {**NEW, "address": "충청북도 진천군 이월면 신계리 88", "jimok": "공장용지", "area": "600",
                 "price": "", "property_type": "공장", "deal_type": "임대", "building_area": "250",
                 "deposit": "5천만원", "rent": "350", "description": "층고 8m, 호이스트 3톤"}


def test_add_factory_lease_from_admin(admin):
    c, agent, data, _ = admin
    form = c.get("/admin/listings/new").get_data(as_text=True)
    assert 'name="property_type"' in form and 'name="deposit"' in form and 'name="building_area"' in form
    r = c.post("/admin/listings/new", data={**FACTORY_LEASE, "csrf": token(c)}, follow_redirects=True)
    assert "새 매물을 추가했습니다" in r.get_data(as_text=True)
    new = next(l for l in agent.listings() if "신계리 88" in l.address)
    assert (new.property_type, new.deal_type, new.deposit_manwon, new.monthly_rent_manwon) == ("공장", "임대", 5000, 350)
    assert round(new.building_area_pyeong) == 250 and new.price_manwon == 0
    ws = load_workbook(data / WEB_FOLDER / WEB_FILE).active  # 엑셀에도 공장 칸이 저장됨
    header = [x.value for x in ws[1]]
    row = dict(zip(header, [x.value for x in ws[ws.max_row]]))
    assert row["매물종류"] == "공장" and row["거래유형"] == "임대" and row["월세"] == 350 and not row["매매가"]

    from real_estate_agent.tool_agent import AgentContext, ToolAgent

    res, _ = ToolAgent(agent, ctx=AgentContext(), log_dir="").run("진천 공장 임대 월세 400 이하")
    assert new.listing_id in res.listing_ids


def test_factory_form_checks(admin):
    c, agent, _, _ = admin
    bad = {**FACTORY_LEASE, "deposit": "", "rent": ""}
    r = c.post("/admin/listings/new", data={**bad, "csrf": token(c)})
    assert r.status_code == 400 and "보증금이나 월세" in r.get_data(as_text=True)
    sale = {**FACTORY_LEASE, "deal_type": "매매", "price": "", "address": "충청북도 음성군 대소면 삼정리 9"}
    r = c.post("/admin/listings/new", data={**sale, "csrf": token(c)})
    assert r.status_code == 400 and "매매가를 입력하세요" in r.get_data(as_text=True)
    land = {**NEW, "address": "충청북도 음성군 감곡면 오향리 778", "deal_type": "임대", "deposit": "1억"}
    c.post("/admin/listings/new", data={**land, "csrf": token(c)})  # 토지는 임대를 골라도 매매로
    l = next(x for x in agent.listings() if "오향리 778" in x.address)
    assert (l.property_type, l.deal_type, l.deposit_manwon) == ("토지", "매매", None)


def test_old_web_excel_gets_factory_columns(admin):
    """예전 프로그램이 만든 웹입력 엑셀(공장 칸 없음)에 공장을 추가해도 칸이 늘어나 그대로 저장됨."""
    from openpyxl import Workbook

    c, agent, data, _ = admin
    path = data / WEB_FOLDER / WEB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "매물입력"
    ws.append(["매물번호", "소재지", "지목", "용도지역", "면적(㎡)", "매매가", "도로", "상태", "등록일", "특이사항", "연락처", "입력일시"])
    wb.save(path)
    c.post("/admin/listings/new", data={**FACTORY_LEASE, "csrf": token(c)})
    assert "월세" in [x.value for x in load_workbook(path).active[1]]
    assert any(l.property_type == "공장" and "신계리 88" in l.address for l in agent.listings())


def test_edit_factory_keeps_lease(admin):
    c, agent, _, _ = admin
    c.post("/admin/listings/new", data={**FACTORY_LEASE, "csrf": token(c)})
    lid = next(l for l in agent.listings() if "신계리 88" in l.address).listing_id
    form = c.get(f"/admin/listings/edit?id={lid}").get_data(as_text=True)
    assert 'value="350"' in form and "<option value=\"임대\" selected" in form
    c.post("/admin/listings/edit", data={**FACTORY_LEASE, "rent": "300", "id": lid, "csrf": token(c)})
    l = agent.find_listing(lid)
    assert l.monthly_rent_manwon == 300 and l.deal_type == "임대"
