"""실제 매물 폴더를 sample_data 처럼 '매물 종류 → 지역' 순서로 정리 (매물/음성군 → 매물/토지/음성군)."""

import re

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.file_uploads import default_folder, destination_folders
from real_estate_agent.folder_layout import apply_plan, make_plan, undo_last


def _sheet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    wb.save(path)


HEAD = ["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "특이사항"]


@pytest.fixture
def old_layout(tmp_path):
    """예전 모양: 매물 폴더 바로 아래에 지역 폴더 + 공장 폴더는 이미 종류별."""
    data = tmp_path / "매물"
    _sheet(data / "음성군" / "장부.xlsx", [HEAD, ["L-1", "충청북도 음성군 대소면 오류리 1", "잡종지", "계획관리지역", 600, "3억", "창고 부지"]])
    _sheet(data / "음성군" / "감곡면 문촌리 5(물류부지)" / "정보.xlsx",
           [["소재지", "충청북도 음성군 감곡면 문촌리 5"], ["면적(평)", 900], ["매매가", "5억"], ["지목", "잡종지"]])
    _sheet(data / "기타지역" / "문경시" / "장부.xlsx", [HEAD, ["L-3", "경상북도 문경시 산북면 대하리 12", "임야", "", 3000, "2억", ""]])
    _sheet(data / "웹입력" / "웹입력매물.xlsx", [HEAD + ["입력일시"], ["W-1", "충청북도 진천군 덕산읍 용몽리 7", "대", "", 200, "1억", "", ""]])
    _sheet(data / "공장" / "임대" / "진천군" / "장부.xlsx",
           [["매물번호", "소재지", "대지면적(평)", "건물면적(평)", "보증금", "월세"], ["F-1", "충청북도 진천군 이월면 신계리 60", 800, 400, "8천만", "500만"]])
    (data / "README.txt").write_text("안내", encoding="utf-8")
    return data


def make_agent(data, tmp_path):
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "r.db", log_dir=tmp_path / "logs", llm_providers=["rule"]))
    a.sync()
    return a


def test_organize_moves_land_under_kind_folder_and_keeps_ids(old_layout, tmp_path):
    a = make_agent(old_layout, tmp_path)
    before = {l.listing_id: (l.property_type, l.address) for l in a.listings()}
    a.storage.add_feedback("손님", next(iter(before)), "like", "")  # 기록은 매물 번호에 붙어 있음

    plan = make_plan(old_layout)
    assert sorted(plan.moves) == [("기타지역", "토지/기타지역"), ("웹입력", "토지/웹입력"), ("음성군", "토지/음성군")]
    r = apply_plan(old_layout, tmp_path / "logs", plan)
    assert not r.failed and r.log_file.exists()
    assert sorted(p.name for p in old_layout.iterdir()) == ["README.txt", "공장", "토지"]
    assert (old_layout / "토지" / "음성군" / "감곡면 문촌리 5(물류부지)" / "정보.xlsx").exists()

    a.sync()
    after = {l.listing_id: (l.property_type, l.address) for l in a.listings()}
    assert after == before  # 같은 매물, 같은 번호 (관리자 수정·관심 기록이 그대로 이어짐)
    assert not make_plan(old_layout).needed

    undo = undo_last(old_layout, tmp_path / "logs")
    assert not undo.failed and (old_layout / "음성군" / "장부.xlsx").exists() and not (old_layout / "토지").exists()


def test_merge_into_existing_land_folder_and_conflicts(old_layout, tmp_path):
    _sheet(old_layout / "토지" / "음성군" / "다른장부.xlsx", [HEAD])
    _sheet(old_layout / "토지" / "음성군" / "장부.xlsx", [HEAD])  # 같은 이름 -> 옮기지 않고 알림
    plan = make_plan(old_layout)
    assert ("음성군/감곡면 문촌리 5(물류부지)", "토지/음성군/감곡면 문촌리 5(물류부지)") in plan.moves
    assert any("음성군/장부.xlsx" in c for c in plan.conflicts)
    apply_plan(old_layout, tmp_path / "logs", plan)
    assert (old_layout / "음성군" / "장부.xlsx").exists()  # 충돌 파일은 제자리
    assert not (old_layout / "음성군" / "감곡면 문촌리 5(물류부지)").exists()


def test_new_files_go_to_kind_folders(old_layout, tmp_path):
    apply_plan(old_layout, tmp_path / "logs")
    assert default_folder(old_layout) == "토지/웹업로드"
    folders = destination_folders(old_layout)
    assert {"토지/음성군", "토지/기타지역", "공장/임대", "공장/임대/진천군"} <= set(folders)
    assert not any("문촌리" in f for f in folders)  # 매물 1건짜리 폴더는 목록에 없음

    from real_estate_agent.listing_editor import parse_form

    a = make_agent(old_layout, tmp_path)
    form = {"address": "충청북도 음성군 맹동면 봉현리 9", "area": "500", "area_unit": "pyeong", "price": "",
            "property_type": "공장", "deal_type": "임대", "deposit": "5천", "rent": "300"}
    a.add_listing(parse_form(form).values)
    assert (old_layout / "공장" / "웹입력" / "웹입력매물.xlsx").exists()
    a.add_listing(parse_form({**form, "property_type": "토지", "price": "2억", "address": "충청북도 음성군 맹동면 봉현리 10"}).values)
    assert (old_layout / "토지" / "웹입력" / "웹입력매물.xlsx").exists()
    kinds = {l.address.split()[-1]: l.property_type for l in a.listings() if "봉현리" in l.address}
    assert kinds == {"9": "공장", "10": "토지"}


def test_factory_like_folder_is_reported_not_moved(tmp_path):
    data = tmp_path / "매물"
    (data / "공장매물" / "음성군").mkdir(parents=True)
    plan = make_plan(data)
    assert not plan.moves and any("공장매물" in w for w in plan.warnings)


def test_admin_page_organizes(old_layout, tmp_path, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    app = create_app(Settings(data_dir=old_layout, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                              llm_providers=["rule"]))
    c = app.test_client()

    def token(path):
        return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)

    c.post("/admin/login", data={"password": "office-pass-1", "csrf": token("/admin")})
    assert "종류별로 정리하기" in c.get("/admin").get_data(as_text=True)
    page = c.get("/admin/organize-folders").get_data(as_text=True)
    assert "음성군  →  토지/음성군" in page
    html = c.post("/admin/organize-folders", data={"op": "apply", "csrf": token("/admin/organize-folders")},
                  follow_redirects=True).get_data(as_text=True)
    assert "종류별로 정리했습니다" in html and "종류별로 정리하기" not in html
    assert (old_layout / "토지" / "음성군").is_dir()
    assert "토지 4건 · 공장 1건" in html


def test_open_file_is_left_in_place_without_duplicates(old_layout, tmp_path, monkeypatch):
    """윈도에서 엑셀로 열어 둔 파일: 옮기지 못하면 그 자리에 그대로 (반쯤 복사·삭제되지 않음), 매물 수도 그대로."""
    import os

    import real_estate_agent.folder_layout as fl

    a = make_agent(old_layout, tmp_path)
    before = {l.listing_id for l in a.listings()}
    real_rename = os.rename

    def locked(src, dst):
        if str(src).endswith("정보.xlsx"):
            raise PermissionError("[WinError 32] 다른 프로세스가 파일을 사용 중")
        return real_rename(src, dst)

    monkeypatch.setattr(fl.os, "rename", locked)
    r = apply_plan(old_layout, tmp_path / "logs")
    assert len(r.failed) == 1 and "정보.xlsx" in r.failed[0]
    assert (old_layout / "음성군" / "감곡면 문촌리 5(물류부지)" / "정보.xlsx").exists()
    assert not (old_layout / "토지" / "음성군" / "감곡면 문촌리 5(물류부지)" / "정보.xlsx").exists()
    a.sync()
    assert {l.listing_id for l in a.listings()} == before


def test_find_missing_explains_why(old_layout, tmp_path):
    from real_estate_agent.folder_layout import find_missing

    a = make_agent(old_layout, tmp_path)
    apply_plan(old_layout, tmp_path / "logs")
    # 옮긴 뒤 한 파일이 망가지고(매매가 없음), 한 파일은 사라졌다고 해 봄
    _sheet(old_layout / "토지" / "음성군" / "장부.xlsx", [HEAD, ["L-1", "충청북도 음성군 대소면 오류리 1", "잡종지", "", 600, None, ""]])
    (old_layout / "토지" / "기타지역" / "문경시" / "장부.xlsx").unlink()
    a.sync()
    found = {m.office_no: m for m in find_missing(a)}
    assert set(found) == {"L-1", "L-3"}
    assert "가격" in found["L-1"].reason and found["L-1"].old_file == "음성군/장부.xlsx"
    assert found["L-1"].now_file == "토지/음성군/장부.xlsx"
    assert "찾을 수 없습니다" in found["L-3"].reason
