"""사무실이 준 매물장 양식 5종 (공장 매매·매매/임대·임대·부분임대, 토지)을 칸 위치 그대로 재현한 테스트.
값은 모두 가짜. (원본 양식 파일은 실제 매물이라 저장소에 넣지 않음)"""

from datetime import datetime

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings


def put(ws, cells):
    for ref, v in cells.items():
        ws[ref] = v


def factory_form(path, deal="매매", sheet="공장매매", price=None, deposit=None, rent=None, unit_price_label=True,
                 zone_header="용도지역/구조", land=(), buildings=(("1동", "공장", "일반철골구조", 600),), land_note=""):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    put(ws, {"B2": "일반 공장", "I2": "음성", "J3": deal})
    if deal == "매매":
        put(ws, {"H6": "매매가", "I6": price})
    elif deal == "매매/임대":  # 가격 오른쪽 J6 = 보증금, J7 = 월세 (이름표 없음)
        put(ws, {"G6": "가  격", "H6": price, "J6": deposit, "J7": rent})
    else:  # 임대 / 부분 임대
        put(ws, {"G6": "보증금", "I6": deposit, "G7": "월세", "I7": rent})
    put(ws, {"B7": "위치", "C7": "음성군 대소읍 가상리 1"})
    put(ws, {"B9": "대지면적", "C9": 1000.5, "D9": "평", "E9": "연면적", "F9": 260.2, "G9": "평"})
    if unit_price_label:
        put(ws, {"I9": "평단가"})  # 부분임대 양식: 이름표만 있고 값 칸(J9)은 빈칸
    else:
        put(ws, {"J9": 1100000})
    put(ws, {"B10": "처마높이", "C10": 9, "D10": "m", "E10": "전기", "F10": 100, "G10": "kw", "H10": "호이스트", "I10": "없음"})
    put(ws, {"B11": "폐수처리", "D11": "日", "E11": "상수도", "F11": "지하수", "H11": "사용승인일", "I11": datetime(2020, 1, 1)})
    put(ws, {"B12": "특이사항", "C12": "● 꽃동네IC-9분 거리", "H12": "● 산업단지 내 공장임", "C13": "● 2차선 도로와 접해 있음"})
    put(ws, {"B15": "구분", "C15": "지번/필지", "E15": "면적", "G15": "지목/용도", "I15": zone_header,
             "J15": "공시지가(㎡당)/규격", "E16": "㎡", "F16": "평"})
    r = 17
    if land_note:
        put(ws, {"B17": "토지", "C17": land_note})
    for i, (m2, jimok, zone) in enumerate(land):
        put(ws, {f"B{r}": "토지" if i == 0 else None, f"C{r}": str(100 + i), f"E{r}": m2, f"F{r}": m2 * 0.3025,
                 f"G{r}": jimok, f"I{r}": zone, f"J{r}": 100000})
        r += 1
    put(ws, {"C25": "합 계", "E25": sum(x[0] for x in land) or None})
    for i, (dong, use, struct, m2) in enumerate(buildings):
        put(ws, {f"B{26 + i}": "건물" if i == 0 else None, f"C{26 + i}": dong, f"E{26 + i}": m2, f"F{26 + i}": m2 * 0.3025,
                 f"G{26 + i}": use, f"I{26 + i}": struct})
    put(ws, {"C36": "합 계", "E36": sum(b[3] for b in buildings), "B38": "위성", "G38": "매물사진"})
    s2 = wb.create_sheet("토지 목록")  # 빈 양식 시트
    put(s2, {"B4": "지번", "C4": "지목", "D4": "지역/지구", "E4": "면적", "G4": "공시지가(㎡당)", "E5": "㎡", "F5": "평",
             "B27": "합계", "E27": 0, "F27": 0})
    for rr in range(6, 27):
        s2[f"F{rr}"] = 0
    s3 = wb.create_sheet("건축물 목록")
    put(s3, {"B4": "동", "C4": "용도", "E4": "구조", "F4": "면적", "H4": "규격", "F5": "㎡", "G5": "평", "B26": "합계", "F26": 0})
    wb.save(path)


def land_form(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "토지"
    put(ws, {"B2": "토지", "I2": "괴산", "L3": "매매", "G6": "가  격", "H6": 544512100, "B7": "위치",
             "C7": "괴산군 청천면 가상리 352 외 1필지", "B9": "대지면적", "C9": 4950.11, "D9": "평", "E9": "연면적", "F9": 0,
             "G9": "평", "H9": "진입로 지분\n합산 면적", "I9": 0, "J9": "평", "K9": "평단가", "L9": 110000,
             "B10": "건폐율\n용적률", "C10": "40%\n100%", "E10": "지목", "F10": "임야", "H10": "지역/지구",
             "I10": "계획관리지역\n성장관리계획구역(관광형)", "B11": "추천용도", "C11": "단독주택, 공장, 창고, 하우스시설 등",
             "B12": "특이사항", "C12": "● 연풍IC-41분 거리", "C13": "● 3m 포장도로와 접해 있음",
             "B15": "구분", "C15": "지번", "E15": "면적", "G15": "지목", "I15": "용도지역", "L15": "개별공시지가\n(㎡당)",
             "E16": "㎡", "F16": "평", "B17": "토지", "C17": "352", "E17": 11858, "F17": 3587.0, "G17": "임야",
             "I17": "계획관리, 성장관리(관광)", "C18": "353", "E18": 4506, "F18": 1363.0, "G18": "임야",
             "I18": "계획관리, 성장관리(관광)", "C30": "합 계", "E30": 16364, "F30": 4950.11, "B31": "건물",
             "C33": "합 계", "E33": 0, "B35": "위성", "G35": "매물사진"})
    s2 = wb.create_sheet("토지 목록")
    put(s2, {"B2": "토지목록", "B4": "순번", "C4": "지번", "D4": "면적", "F4": "지목", "G4": "토지주", "D5": "㎡", "E5": "평",
             "B16": "합계"})
    wb.save(path)


def land_list_file(path):
    """'토지목록 확인' 매물의 별도 토지목록 파일: 주소 | 지목 | 면적 | 비고, 지목은 '도(도로)' 처럼."""
    wb = Workbook()
    ws = wb.active
    put(ws, {"B4": "금왕읍 가상리 683-6 토지목록", "B5": "주소", "C5": "지목", "D5": "면적", "F5": "비고"})
    for i, (j, m2) in enumerate([("도(도로)", 120), ("대(대지)", 3000), ("대(대지)", 2500), ("임(임야)", 900),
                                 ("장(공장)", 1800), ("전(밭)", 400), ("잡(잡종지)", 700)]):
        put(ws, {f"B{6 + i}": f"가상리 {683 + i}", f"C{6 + i}": j, f"D{6 + i}": m2, f"E{6 + i}": round(m2 * 0.3025, 1)})
    put(ws, {"B13": "합계", "D13": 9420})
    wb.save(path)


@pytest.fixture(scope="module")
def office(tmp_path_factory):
    t = tmp_path_factory.mktemp("forms")
    data = t / "매물"

    def folder(*parts):
        d = data.joinpath(*parts)
        d.mkdir(parents=True)
        return d

    d = folder("공장", "매매", "음성군", "삼성면 가상리 13-4")
    factory_form(d / "삼성면 가상리 13-4.xlsx", price=1056000000, land=((1792, "공장용지", "계획관리, 성장관리(산업)"),))
    d = folder("공장", "매매", "음성군", "금왕읍 가상리 7-16 외 1필지")
    factory_form(d / "a.xlsx", price=950000000, unit_price_label=False, land=((6612, "공장용지", "도시, 일반공업"),))
    d = folder("공장", "매매임대", "음성군", "목행동 가상리 890")
    factory_form(d / "b.xlsx", deal="매매/임대", sheet="공장매매임대", price=482121475, deposit=20000000, rent=3000000,
                 unit_price_label=False, zone_header="구조/용도지역",
                 land=((208, "전", "계획관리, 성장관리"), (627, "공장용지", "계획관리, 성장관리"), (614, "공장용지", "계획관리, 성장관리")))
    d = folder("공장", "임대", "음성군", "대소읍 가상리 1031")
    factory_form(d / "c.xlsx", deal="임대", sheet="공장임대", deposit=60000000, rent=5750000, zone_header="구조/용도지역",
                 land=((3310.3, "공장용지", "도시, 일반공업, 지구단위"),))
    d = folder("공장", "임대", "음성군", "대소읍 가상리 75-1 외 8필지(부분임대)")  # '평단가' 이름표만 있고 값 없음
    factory_form(d / "d.xlsx", deal="부분 임대", sheet="공장임대", deposit=35000000, rent=3500000, zone_header="구조/용도지역",
                 land=((900, "잡종지", "계획관리, 성장관리(복합)"), (1200, "잡종지", "계획관리, 성장관리(복합)"),
                       (50, "도로", "계획관리, 성장관리(복합)"), (300, "전", "계획관리, 성장관리(복합)")))
    d = folder("공장", "매매", "음성군", "금왕읍 가상리 683-6번지 외 15필지")
    factory_form(d / "e.xlsx", price=18000000000, land_note="토지목록 확인")
    land_list_file(d / "토지목록.xlsx")
    d = folder("토지", "괴산군", "청천면 가상리 352 외 1필지")
    land_form(d / "f.xlsx")
    a = RealEstateAgent(Settings(data_dir=data, db_path=t / "o.db", log_dir=t / "l", llm_providers=["rule"],
                                 hide_unnumbered=False))
    report = a.sync()
    return a, {l.address.split("가상리 ")[-1].split()[0]: l for l in a.listings()}, report


def test_all_forms_read(office):
    _, ls, report = office
    assert len(ls) == 7 and report.clean.error_count == 0


@pytest.mark.parametrize("key, jimok, zoning, districts", [
    ("13-4", "공장용지", "계획관리지역", "계획관리지역, 성장관리계획구역(산업)"),
    ("7-16", "공장용지", "일반공업지역", "도시지역, 일반공업지역"),
    ("890", "공장용지", "계획관리지역", "계획관리지역, 성장관리계획구역"),
    ("1031", "공장용지", "일반공업지역", "도시지역, 일반공업지역, 지구단위계획구역"),
    ("75-1", "잡종지", "계획관리지역", "계획관리지역, 성장관리계획구역(복합)"),
    ("352", "임야", "계획관리지역", "계획관리지역, 성장관리계획구역(관광형)"),
])
def test_jimok_and_zoning(office, key, jimok, zoning, districts):
    l = office[1][key]
    assert (l.jimok, l.zoning, l.extra.get("_지역지구", l.zoning)) == (jimok, zoning, districts)


def test_land_list_file_with_short_jimok(office):
    l = office[1]["683-6번지"]  # 본문은 '토지목록 확인', 토지목록 파일의 '대(대지)', '도(도로)' …
    assert l.jimok == "대" and l.zoning == ""  # 이 양식에는 용도지역 칸이 없음


def test_prices_and_areas(office):
    ls = office[1]
    assert ls["13-4"].price_text == "10억 5,600만원"
    assert ls["890"].price_text == "매매 4억 8,212만원 · 임대 보증금 2,000만원 / 월 300만원"  # J6·J7 (이름표 없음)
    assert ls["1031"].price_text == "임대 보증금 6,000만원 / 월 575만원"
    assert ls["75-1"].price_text == "임대 보증금 3,500만원 / 월 350만원"
    assert round(ls["1031"].area_pyeong) == 1001 and round(ls["1031"].building_area_pyeong) == 260
    assert "필지 일괄" not in ls["75-1"].features


def test_sheet_layout_has_no_fake_labels(office, tmp_path):
    from real_estate_agent.sheet_layout import layout_lines

    l = office[1]["683-6번지"]
    from pathlib import Path

    text = "\n".join(layout_lines(Path(l.source_file).with_name("토지목록.xlsx")))
    assert "토지목록✓" not in text and "지목✓" in text
