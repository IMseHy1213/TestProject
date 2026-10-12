"""'매물 하나 = 폴더 하나' 구조 테스트.

매물/음성군/음성군 감곡면 문촌리 417-2 외 36필지(물류부지)/음성군 감곡면 문촌리 417-2 외 36필지.xlsx
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from real_estate_agent.agent import mask_address
from real_estate_agent.cleaner import clean
from real_estate_agent.collectors import LocalFolderCollector
from real_estate_agent.collectors.listing_folder import looks_like_listing_name, split_name
from real_estate_agent.recommender import rank
from real_estate_agent.requirement_parser import RequirementParser

TODAY = date(2026, 10, 12)
NAME = "음성군 감곡면 문촌리 417-2 외 36필지"


def write_xlsx(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def listing_dir(root: Path) -> Path:
    return root / "음성군" / f"{NAME}(물류부지)"


def parcel_list_rows() -> list[list]:
    """형태 1: 필지 목록표 + 합계 + 매매가 항목"""
    rows = [[f"{NAME} 물건 정보", None, None, None, None], [None] * 5,
            ["매매가", "35억", None, None, None], ["도로", "8m 도로 접함", None, None, None], [None] * 5,
            ["순번", "지번", "지목", "면적(㎡)", "용도지역"]]
    for i in range(37):
        jimok = "공장용지" if i < 25 else "전"
        rows.append([i + 1, f"문촌리 {417 + i}-2", jimok, 900 if i < 25 else 750, "계획관리지역"])
    rows.append(["합계", None, None, 25 * 900 + 12 * 750, None])
    return rows


def test_name_helpers():
    assert looks_like_listing_name(f"{NAME}(물류부지)")
    assert not looks_like_listing_name("음성군") and not looks_like_listing_name("토지매물장부_2026")
    assert split_name(f"{NAME}(물류부지)") == (NAME, ["물류부지"])
    assert mask_address(NAME) == "음성군 감곡면 문촌리 *** 외 36필지"


def test_parcel_list_folder_becomes_one_listing(tmp_path):
    root = tmp_path / "매물"
    write_xlsx(listing_dir(root) / f"{NAME}.xlsx", parcel_list_rows())
    result = clean(LocalFolderCollector(root).collect(), TODAY)
    assert result.error_count == 0 and len(result.listings) == 1  # 37필지가 아니라 매물 1건
    l = result.listings[0]
    assert l.address == NAME
    assert l.area_m2 == 25 * 900 + 12 * 750
    assert l.price_manwon == 350000
    assert l.jimok == "공장용지" and l.zoning == "계획관리지역"  # 면적이 가장 큰 지목
    assert "물류부지" in l.description and "37필지" in l.description
    assert l.road_access == "8m 도로 접함"


def test_info_sheet_layout(tmp_path):
    """형태 2: '항목 | 값' 정보지"""
    root = tmp_path / "매물"
    write_xlsx(listing_dir(root) / f"{NAME}.xlsx", [
        ["매물 정보지", None, None, None],
        ["소재지", "충청북도 음성군 감곡면 문촌리 417-2 외 36필지", None, None],
        ["지목", "공장용지", "용도지역", "계획관리지역"],
        ["면적", "31,500㎡", "평수", "9,529평"],
        ["매매가", "35억원", None, None],
        ["특이사항", "물류창고 부지, 감곡IC 3분, 대형차 진입 가능", None, None],
    ])
    l = clean(LocalFolderCollector(root).collect(), TODAY).listings[0]
    assert l.area_m2 == 31500 and l.price_manwon == 350000 and l.jimok == "공장용지"
    assert "감곡IC" in l.description and "물류부지" in l.description


def test_single_row_table_and_region_prefix(tmp_path):
    """형태 3: 한 줄짜리 표 / 폴더 이름에 군이 없으면 지역 폴더 이름을 붙임"""
    root = tmp_path / "매물"
    folder = root / "음성군" / "금왕읍 무극리 210-3(전원주택지)"
    write_xlsx(folder / "정보.xlsx", [["소재지", "지목", "면적(평)", "매매가", "특이사항"],
                                      ["금왕읍 무극리 210-3", "대", 250, "2억 3천", "정남향 채광 우수"]])
    l = clean(LocalFolderCollector(root).collect(), TODAY).listings[0]
    assert l.address == "음성군 금왕읍 무극리 210-3"
    assert l.price_manwon == 23000 and l.area_m2 == pytest.approx(250 * 3.305785, rel=1e-3)


def test_folders_and_ledgers_together_and_search(tmp_path):
    root = tmp_path / "매물"
    write_xlsx(listing_dir(root) / f"{NAME}.xlsx", parcel_list_rows())
    write_xlsx(root / "토지장부.xlsx", [["소재지", "지목", "면적(평)", "매매가"],
                                       ["충북 음성군 대소면 오류리 55", "전", 350, "2억 7천"],
                                       ["충북 음성군 원남면 하당리 77", "전", 400, "1억 8천"]])
    listings = clean(LocalFolderCollector(root).collect(), TODAY).listings
    assert len(listings) == 3
    req = RequirementParser(None, ["음성군"]).parse("음성군에 물류창고 지을 땅 30억에서 40억")
    ranked, _, _ = rank(listings, req, today=TODAY)
    assert ranked[0].listing.address == NAME


def test_missing_price_is_reported(tmp_path):
    root = tmp_path / "매물"
    rows = [r for r in parcel_list_rows() if r[0] != "매매가"]
    write_xlsx(listing_dir(root) / f"{NAME}.xlsx", rows)
    result = clean(LocalFolderCollector(root).collect(), TODAY)
    assert result.error_count == 1 and "가격" in result.issues[0].message


def test_etc_region_folder(tmp_path):
    """'기타지역' 같은 폴더는 주소 앞에 붙이지 않고, 시/군은 엑셀 소재지에서 보충"""
    root = tmp_path / "매물"
    write_xlsx(root / "기타지역" / "문백면 옥성리 123(공장부지)" / "정보.xlsx", [
        ["소재지", "충청북도 진천군 문백면 옥성리 123", None, None],
        ["면적", "5,000㎡", "매매가", "12억"],
        ["지목", "공장용지", None, None],
    ])
    write_xlsx(root / "기타지역" / "충주시 앙성면 매물.xlsx", [  # 지번 없는 이름 + 정보지 형태
        ["소재지", "충청북도 충주시 앙성면 단암리 55"], ["면적(평)", "800"], ["매매가", "4억 5천"],
    ])
    write_xlsx(root / "기타지역" / "장부.xlsx", [["소재지", "면적(평)", "매매가"],
                                               ["경기도 이천시 장호원읍 이황리 7", 500, "3억"]])
    collector = LocalFolderCollector(root)
    listings = clean(collector.collect(), TODAY).listings
    addrs = sorted(l.address for l in listings)
    assert addrs == ["경기도 이천시 장호원읍 이황리 7", "충청북도 진천군 문백면 옥성리 123",
                     "충청북도 충주시 앙성면 단암리 55"]
    assert not any("기타지역" in a for a in addrs)
    assert collector.folder_counts(listings) == {"기타지역": 3}
    req = RequirementParser(None, ["진천군", "충주시"]).parse("진천 공장 부지")
    ranked, _, _ = rank(listings, req, today=TODAY)
    assert "진천군" in ranked[0].listing.address


def test_gitajiyeok_cities_are_searchable(tmp_path):
    """매물/기타지역/ 아래 문경시·밀양시·봉화군 매물 (시 폴더 안 / 바로 아래 폴더 / 파일 하나) 모두 검색"""
    root = tmp_path / "매물"
    write_xlsx(root / "기타지역" / "문경시" / "산북면 대하리 12(임야)" / "정보.xlsx",
               [["면적(평)", 3000], ["매매가", "1억 5천"], ["지목", "임야"]])
    write_xlsx(root / "기타지역" / "밀양시 산내면 가인리 5(전원주택지)" / "정보.xlsx",
               [["소재지", "경상남도 밀양시 산내면 가인리 5"], ["면적(평)", 300], ["매매가", "1억 2천"], ["지목", "대"]])
    write_xlsx(root / "기타지역" / "봉화군 물야면 오전리 77.xlsx",
               [["면적(평)", 1500], ["매매가", "9천만원"], ["지목", "전"]])
    write_xlsx(listing_dir(root) / f"{NAME}.xlsx", parcel_list_rows())
    collector = LocalFolderCollector(root)
    listings = clean(collector.collect(), TODAY).listings
    assert collector.folder_counts(listings) == {"기타지역": 3, "음성군": 1}
    addrs = {l.address for l in listings}
    assert "문경시 산북면 대하리 12" in addrs  # 시 폴더 이름을 주소 앞에 붙임
    assert not any(a.startswith("기타지역") for a in addrs)
    known = sorted({t for l in listings for t in l.address.split() if t[-1] in "시군읍면리"})
    p = RequirementParser(None, known)
    expect = {"문경 임야": "문경시", "밀양 쪽 전원주택 땅": "밀양시", "봉화군 농지": "봉화군", "경북 땅": None}
    for q, city in expect.items():
        ranked, _, _ = rank(listings, p.parse(q), today=TODAY)
        assert ranked, q
        if city:
            assert all(city in s.listing.address for s in ranked), q
    # 경북 = 문경시 + 봉화군 (밀양시는 경남, 음성군은 충북)
    ranked, _, _ = rank(listings, p.parse("경북 땅"), today=TODAY)
    assert {s.listing.address.split()[0] for s in ranked} == {"문경시", "봉화군"}
    ranked, _, _ = rank(listings, p.parse("안동 땅"), today=TODAY)  # 매물이 없는 지역
    assert ranked == []
