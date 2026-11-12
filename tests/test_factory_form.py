"""부모님 사무실의 공장 설명서 양식 (python main.py sheet-layout 결과대로 칸 배치를 재현, 값은 가짜)."""

from openpyxl import Workbook


def row(ws, r, cells):
    for col, v in cells.items():
        ws[f"{col}{r}"] = v


def make_form(path, kind="공장", city="아산", address="충청남도 아산시 음봉면 산동리 607-4 외 2필지", price=135000,
         land=((1200, "대", "계획관리, 성장관리(산업)"), (300, "도로", "계획관리, 성장관리(산업)"), (500, "답", "계획관리, 성장관리(산업)")),
         buildings=(("공장, 창고", "일반철골", 800),), land_py=600, bld_py=250, deal="매매", total_after_land=False):
    wb = Workbook()
    ws = wb.active
    row(ws, 2, {"B": kind, "I": city})
    row(ws, 3, {"J": deal})
    row(ws, 6, {"H": "매매가", "I": price})
    row(ws, 7, {"B": "위치", "C": address})
    row(ws, 9, {"B": "토지면적", "C": land_py, "D": "평", "E": "건축면적", "F": bld_py, "G": "평", "J": 12345})
    row(ws, 10, {"B": "처마높이", "C": 9, "D": "m", "E": "전기", "F": 300, "G": "kw", "H": "호이스트", "I": "5톤 2대"})
    row(ws, 11, {"B": "폐수처리", "D": "日", "E": "상수도", "F": "지하수", "H": "사용승인일", "I": "2015-03-01"})
    row(ws, 12, {"B": "특이사항", "C": "● 대소IC 5분, 25톤 트럭 진입", "H": "●"})
    row(ws, 15, {"B": "구분", "C": "지번/필지", "E": "면적", "G": "지목/용도", "I": "용도지역/구조", "J": "공시지가(㎡당)/규격"})
    row(ws, 16, {"E": "㎡", "F": "평"})
    r = 17
    for i, (m2, jimok, zone) in enumerate(land):
        row(ws, r, {"B": "토지" if i == 0 else None, "C": f"{607 + i}-4", "E": m2, "F": round(m2 / 3.3058, 1),
                    "G": jimok, "I": zone, "J": 150000})
        r += 1
    if not total_after_land:
        r += 2  # 빈 줄
    row(ws, r, {"C": "합 계", "E": sum(x[0] for x in land), "F": round(sum(x[0] for x in land) / 3.3058, 1)})
    r += 1
    for i, (use, struct, m2) in enumerate(buildings):
        row(ws, r, {"B": "건물" if i == 0 else None, "C": i + 1, "E": m2, "F": round(m2 / 3.3058, 1), "G": use, "I": struct, "J": "H-300"})
        r += 1
    row(ws, r + 1, {"C": "합 계", "E": sum(b[2] for b in buildings)})
    row(ws, r + 3, {"B": "위치도", "G": "매물사진"})
    s2 = wb.create_sheet()
    row(s2, 4, {"B": "지번", "C": "토지주", "D": "지목", "E": "지역/지구", "F": "면적", "H": "공시지가(㎡당)"})
    row(s2, 5, {"F": "㎡", "G": "평"})
    for rr in range(6, 22):
        row(s2, rr, {"H": 0})
    row(s2, 22, {"B": "합계", "F": 0, "G": 0})
    s3 = wb.create_sheet()
    row(s3, 4, {"B": "동", "C": "용도", "E": "구조", "F": "면적", "H": "규격"})
    row(s3, 5, {"F": "㎡", "G": "평"})
    row(s3, 6, {"H": 0})
    row(s3, 26, {"B": "합계", "F": 0, "G": 0})
    wb.save(path)


import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings


@pytest.fixture(scope="module")
def listings(tmp_path_factory):
    t = tmp_path_factory.mktemp("form")
    base = t / "매물" / "공장" / "매매" / "음성군"
    d = base / "음봉면 산동리 607-4 외 2필지"
    d.mkdir(parents=True)
    make_form(d / "아산시 음봉면 산동리 607-4 외 2필지.xlsx")
    d = base / "서원면 창촌리 165-2 외 7필지"
    d.mkdir(parents=True)
    make_form(d / "b.xlsx", kind="생수공장", land=((3000, "공장용지", "농림지역, 생산관리지역"), (2000, "임야", "생산관리지역"),
                                                    (100, "도로", "농림지역, 생산관리지역"), (900, "임야", "농림지역")),
              buildings=(("생수생산업", "일반철골구조", 1500), ("사무동", "철근콘크리트구조", 200)), total_after_land=True)
    d = base / "삼죽면 기솔리 226 외 11필지"
    d.mkdir(parents=True)
    make_form(d / "c.xlsx", land=((2000, "공장용지", "계획관리"), (100, "도로", "계획관리"), (800, "대지", "생산관리")),
              buildings=(("공장, 창고", "일반철골", 900), ("식당, 기숙사", "", 100)), total_after_land=True)
    a = RealEstateAgent(Settings(data_dir=t / "매물", db_path=t / "x.db", log_dir=t / "l", llm_providers=["rule"]))
    a.sync()
    return a, {l.address.split()[-3]: l for l in a.listings()}


def test_jimok_and_zoning_from_parcel_table(listings):
    _, ls = listings
    assert (ls["607-4"].jimok, ls["607-4"].zoning) == ("대", "계획관리지역")
    assert ls["607-4"].extra["_지역지구"] == "계획관리지역, 성장관리계획구역(산업)"
    assert (ls["165-2"].jimok, ls["165-2"].zoning) == ("공장용지", "생산관리지역")  # 면적이 가장 큰 필지 / 면적 순
    assert ls["165-2"].extra["_지역지구"] == "생산관리지역, 농림지역"
    assert (ls["226"].jimok, ls["226"].zoning) == ("공장용지", "계획관리지역")  # '건물' 줄(공장, 창고)은 지목이 아님


def test_area_with_pyeong_unit_cell_and_no_parcel_count_text(listings):
    _, ls = listings
    l = ls["607-4"]
    assert round(l.area_pyeong) == 600 and round(l.building_area_pyeong) == 250  # '토지면적 | 600 | 평' 은 600평
    assert "필지 일괄" not in l.features and "필지 일괄" not in l.description
    assert l.price_text == "13억 5,000만원"


def test_cards_and_ai_see_jimok_and_zoning(listings):
    from real_estate_agent.tool_agent import AgentContext, ToolAgent
    from real_estate_agent.tool_agent.tools import brief

    a, ls = listings
    b = brief(ls["607-4"])
    assert b["지목"] == "대" and b["용도지역"] == "계획관리지역" and "성장관리계획구역" in b["지역지구"]
    assert "필지 일괄" not in (b.get("매물특징") or "")
    r, _ = ToolAgent(a, ctx=AgentContext(), log_dir="").run("음성군에 매매공장을 보여줘.")
    assert len(r.listing_ids) == 3


@pytest.mark.parametrize("folder, deal", [("매매임대", "매매/임대"), ("매매-임대", "매매/임대"), ("매매,임대", "매매/임대"),
                                          ("매매/임대", "매매/임대"), ("매매", "매매"), ("임대", "임대")])
def test_deal_folder_names(tmp_path, folder, deal):
    """'매매/임대' 로 바꾸면 윈도에서는 매매 폴더 안에 임대 폴더가 생김 -> 그래도 매매/임대."""
    from pathlib import Path

    from real_estate_agent.collectors.local_folder import kind_from_path

    root = Path(tmp_path)
    assert kind_from_path(root / "공장" / Path(folder) / "음성군" / "a.xlsx", root) == ("공장", deal)
