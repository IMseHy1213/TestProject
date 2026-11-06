"""지금 매물 목록에 없는 매물번호(연습용 샘플·지운 매물·AI 가 지어낸 번호)는 화면에 보이지 않게."""

import re
from pathlib import Path

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import SAMPLE_DB_PATH, Settings, get_settings, is_sample_data
from real_estate_agent.listing_numbers import MISSING, hide_unknown, known_numbers, unknown_in
from real_estate_agent.reliability.guardrails import Evidence, check_answer

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


def real_folder(tmp_path) -> Path:
    """사무실 실제 매물 폴더 흉내 (샘플과 다른 매물번호)."""
    data = tmp_path / "매물" / "진천군"
    data.mkdir(parents=True)
    wb = Workbook()
    ws = wb.active
    ws.append(["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "도로", "특이사항"])
    ws.append(["R-101", "충청북도 진천군 덕산읍 용몽리 10", "공장용지", "계획관리지역", 1200, "8억", "8m 도로 접함", "공장 부지"])
    ws.append(["R-102", "충청북도 진천군 이월면 송림리 20", "대", "계획관리지역", 300, "2억", "6m 도로 접함", "창고 가능"])
    wb.save(data / "장부.xlsx")
    return data.parent


def test_sample_data_gets_its_own_db(monkeypatch):
    monkeypatch.delenv("REA_DB_PATH", raising=False)
    assert is_sample_data(SAMPLE) and get_settings(data_dir=SAMPLE).db_path == SAMPLE_DB_PATH
    assert get_settings(data_dir="/tmp/somewhere").db_path != SAMPLE_DB_PATH
    assert get_settings(data_dir=SAMPLE, db_path="/tmp/x.db").db_path == Path("/tmp/x.db")  # 직접 정하면 그대로


def test_hide_unknown():
    known = {"R-101", "S.24250409"}
    text = "최저가 E-3003(2억 1,000만원) · R-101 · S.24250409 · 금왕IC 5분 · 2026-11-06"
    out = hide_unknown(text, known)
    assert "E-3003" not in out and MISSING in out and "R-101" in out and "S.24250409" in out and "2026-11-06" in out
    assert unknown_in(text, known) == ["E-3003"]


def test_ai_answer_with_nonexistent_number_is_rejected(tmp_path):
    a = RealEstateAgent(Settings(data_dir=real_folder(tmp_path), db_path=tmp_path / "r.db", llm_providers=["rule"]))
    a.sync()
    assert "R-101" in known_numbers(a)
    problems = check_answer("E-3002 매물을 추천합니다.", Evidence.from_observations([]), a)
    assert any("없는 매물번호" in p and "E-3002" in p for p in problems)


@pytest.fixture
def shared_db_apps(tmp_path, monkeypatch):
    """예전처럼 샘플로 연습한 기록이 실제 운영 DB 에 남아 있는 상황."""
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    db = tmp_path / "agent.db"
    practice = create_app(Settings(data_dir=SAMPLE, db_path=db, log_dir=tmp_path / "logs", llm_providers=["rule"]))
    real = create_app(Settings(data_dir=real_folder(tmp_path), db_path=db, log_dir=tmp_path / "logs",
                               llm_providers=["rule"]))
    for app in (practice, real):
        app.config["TESTING"] = True
    return practice, real


def _csrf(c, path="/agent"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_old_sample_answers_are_masked(shared_db_apps):
    from real_estate_agent.web.app import VISITOR_COOKIE

    practice, real = shared_db_apps
    c1 = practice.test_client()
    c1.post("/agent/ask", data={"q": "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "csrf": _csrf(c1)})
    html = c1.get("/agent").get_data(as_text=True)
    assert "E-3003" in html  # 연습 중에는 샘플 매물이 보임
    vid = c1.get_cookie(VISITOR_COOKIE).value
    c2 = real.test_client()
    c2.set_cookie(VISITOR_COOKIE, vid)
    html = c2.get("/agent").get_data(as_text=True)  # 같은 방문자가 실제 운영 화면을 엶 → 예전 대화가 이어서 보임
    assert "음성군에 창고" in html and MISSING in html
    assert not re.search(r"E-30\d\d", html)  # 샘플 매물번호는 답변·실행 과정 어디에도 없음
    assert "지금 매물 목록에 없어 가렸습니다" in html


def test_old_sample_alerts_are_not_replayed(shared_db_apps):
    practice, real = shared_db_apps
    hub = practice.config["REALTIME"]
    hub.bus.publish("alert.match", {"listing": {"id": "E-3002", "name": "E-3002", "price": "2억 7,000만원"},
                                    "message": "찾으시던 조건에 맞는 새 매물: E-3002"}, owner="web:someone")
    hub.bus.publish("listing.added", {"listing": {"id": "E-3002", "name": "E-3002"}, "source": "샘플"})
    bus = real.config["REALTIME"].bus
    assert not [e for e in bus.since(0, "web:someone") if e.topic == "alert.match"]  # 손님에게 다시 보내지 않음
    admin_feed = bus.recent(10, see_all=True)
    assert admin_feed and all("E-3002" not in str(e.payload) for e in admin_feed)


def test_sample_banner(shared_db_apps):
    practice, real = shared_db_apps
    assert "연습용 <strong>샘플 매물</strong>" in practice.test_client().get("/").get_data(as_text=True)
    assert "샘플 매물</strong>" not in real.test_client().get("/").get_data(as_text=True)
