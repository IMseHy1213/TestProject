"""'매물 특징' 표시 형식 테스트: '● 대소IC-15분 거리(중부 고속도로) / 2필지 일괄 매매'"""

from pathlib import Path

from openpyxl import Workbook

from real_estate_agent.collectors.listing_folder import feature_lines, format_features
from real_estate_agent.collectors.local_folder import LocalFolderCollector
from real_estate_agent.cleaner import clean


def test_feature_lines_and_format():
    assert feature_lines("● 대소IC-15분 거리(중부 고속도로)\n• 2차선 도로와 접해 있음") == [
        "● 대소IC-15분 거리(중부 고속도로)", "● 2차선 도로와 접해 있음"]
    assert feature_lines("정남향 평지, 채광 우수") == ["● 정남향 평지, 채광 우수"]
    assert feature_lines("- 계곡 인접 / ※ 분할 가능") == ["● 계곡 인접", "※ 분할 가능"]
    assert format_features(["● 대소IC-15분 거리(중부 고속도로)"], 2) == "● 대소IC-15분 거리(중부 고속도로) / 2필지 일괄 매매"
    assert format_features([], 1) == ""


def _folder_listing(tmp_path: Path, name: str, sheet_rows: list[list], parcels: list[list] | None = None):
    folder = tmp_path / "매물" / "음성군" / name
    folder.mkdir(parents=True)
    wb = Workbook()
    ws = wb.active
    for r in sheet_rows:
        ws.append(r)
    if parcels:
        ws.append([])
        ws.append(["번호", "소재지", "지목", "면적(㎡)", "금액"])
        for p in parcels:
            ws.append(p)
    wb.save(folder / f"{name}.xlsx")
    records = LocalFolderCollector(tmp_path / "매물").collect()
    return clean(records).listings[0]


def test_folder_features_under_label_with_tag_and_parcels(tmp_path):
    name = "음성군 대소면 삼정리 100 외 1필지(상가 부지)"
    l = _folder_listing(tmp_path, name, [
        ["소재지", "충청북도 음성군 대소면 삼정리 100"],
        ["매매가", "5억"],
        ["매물 특징", "● 대소IC-15분 거리(중부 고속도로)"],
        [None, "● 2차선 도로와 접해 있음"],
        [None, "● 주변 상가 밀집"],
    ], parcels=[[1, "충청북도 음성군 대소면 삼정리 100", "대", 500, 25000],
                [2, "충청북도 음성군 대소면 삼정리 101", "대", 500, 25000]])
    assert l.features == ("● 대소IC-15분 거리(중부 고속도로) / ● 2차선 도로와 접해 있음 / ● 주변 상가 밀집"
                          " / 2필지 일괄 매매")
    assert "상가 부지" not in l.features and l.use_tag == "상가 부지"  # 용도는 따로 (주소 옆 표시)
    assert "상가 부지" in l.description and "주변 상가" in l.description  # 검색에는 그대로 쓰임


def test_features_written_below_label(tmp_path):
    l = _folder_listing(tmp_path, "음성군 금왕읍 무극리 55", [
        ["소재지", "충청북도 음성군 금왕읍 무극리 55"], ["면적", "300평"], ["매매가", "2억"],
        ["매물특징"], ["대소IC-15분 거리(중부 고속도로)"], ["정남향"], [], ["연락처", "010-0000-0000"],
    ])
    assert l.features == "● 대소IC-15분 거리(중부 고속도로) / ● 정남향"


def test_ledger_and_web_listing_features(tmp_path):
    from real_estate_agent.models import LandListing

    l = LandListing("T-1", "경기도 양평군 서종면 문호리 1", 1000, 10000, description="정남향 평지, 채광 우수")
    assert l.features == "● 정남향 평지, 채광 우수"


def test_card_shows_features_row(tmp_path, monkeypatch):
    import re
    import shutil

    from real_estate_agent.config import Settings
    from real_estate_agent.web.app import create_app

    sample = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
    data = tmp_path / "토지"
    shutil.copytree(sample, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    c = create_app(settings).test_client()
    token = re.search(r'name="csrf" value="([^"]+)"', c.get("/").get_data(as_text=True)).group(1)
    html = c.post("/search", data={"q": "양평 땅", "csrf": token}, follow_redirects=True).get_data(as_text=True)
    assert "<dt>매물 특징</dt>" in html and '<dd class="features">● ' in html
    assert '<p class="desc">' not in html  # 특징은 정보 칸 안으로 옮김


def test_parcel_table_note_column_is_not_features(tmp_path):
    l = _folder_listing(tmp_path, "음성군 생극면 방축리 88 외 1필지", [
        ["매물특징", "● 평탄 작업 완료"],
    ], parcels=[[1, "충청북도 음성군 생극면 방축리 88", "전", 1000, 5000],
                [2, "충청북도 음성군 생극면 방축리 89", "전", 1200, 6000]])
    assert l.features == "● 평탄 작업 완료 / 2필지 일괄 매매"

    l2 = _folder_listing(tmp_path / "b", "음성군 생극면 차곡리 7 외 1필지", [["면적", "600평"], ["매매가", "1억"]], parcels=None)
    assert l2.features == "2필지 일괄 매매"


def test_county_plus_township_narrows_to_township():
    from real_estate_agent.requirement_parser import RequirementParser, narrow_regions

    addrs = ["충청북도 음성군 대소면 삼정리 100", "충청북도 음성군 원남면 하당리 3", "경기도 여주시 가남읍 1"]
    assert narrow_regions(["대소면", "음성군"], addrs) == ["대소면"]
    assert narrow_regions(["여주시", "대소면"], addrs) == ["여주시", "대소면"]  # 서로 다른 지역은 둘 다
    assert narrow_regions(["음성군"], addrs) == ["음성군"]
    p = RequirementParser(None, ["음성군", "대소면", "원남면", "여주시", "가남읍"], addrs)
    assert p.parse("음성군 대소면 땅").regions == ["대소면"]
    assert sorted(p.parse("여주나 음성 땅").regions) == ["여주시", "음성군"]
