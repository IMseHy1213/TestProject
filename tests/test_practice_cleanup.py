"""연습용 저장 파일 분리 + 실제 운영 DB 에 섞인 연습 기록만 지우는 정리 기능."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import (REAL_DB_PATH, SAMPLE_DB_PATH, SAMPLE_LOG_DIR, Settings, get_settings)
from real_estate_agent.practice_cleanup import clean, scan, summary
from real_estate_agent.tool_agent.service import AgentService

from test_listing_numbers import real_folder

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


def test_practice_has_its_own_files(monkeypatch):
    monkeypatch.delenv("REA_DB_PATH", raising=False)
    monkeypatch.delenv("REA_LOG_DIR", raising=False)
    s = get_settings(data_dir=SAMPLE)
    assert s.db_path == SAMPLE_DB_PATH and s.log_dir == SAMPLE_LOG_DIR
    assert get_settings(data_dir=SAMPLE, db_path=REAL_DB_PATH).db_path == SAMPLE_DB_PATH  # 실제 DB 를 가리켜도 보호
    real = get_settings()
    assert real.db_path == REAL_DB_PATH or str(real.db_path).endswith("agent.db")


@pytest.fixture
def mixed_db(tmp_path):
    """예전처럼 연습(샘플)과 실제 운영이 같은 DB 를 쓴 상황."""
    db = tmp_path / "data" / "agent.db"
    db.parent.mkdir()
    practice = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=db, log_dir=tmp_path / "l", llm_providers=["rule"]))
    practice.sync()
    sp = AgentService(practice)
    cid_practice, t1 = sp.ask("web:v1", None, "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    sp.rate("web:v1", cid_practice, t1["id"], 1)
    sp.listing_feedback("web:v1", cid_practice, t1["listing_ids"][0], "like")
    from real_estate_agent.realtime import RealtimeHub

    RealtimeHub(practice).bus.publish("alert.match", {"listing": {"id": "E-3003", "name": "E-3003"}}, owner="web:v1")
    cid_mixed, _ = sp.ask("web:v3", None, "양평 땅 2개 비교해줘")

    real = RealEstateAgent(Settings(data_dir=real_folder(tmp_path), db_path=db, log_dir=tmp_path / "l",
                                    llm_providers=["rule"]))
    real.sync()
    sr = AgentService(real)
    cid_real, _ = sr.ask("web:v2", None, "진천군 땅 보여줘")
    sr.ask("web:v3", cid_mixed, "진천군 공장 땅 있어?")  # 연습하던 대화에서 실제 질문
    return {"db": db, "real": real, "practice": cid_practice, "mixed": cid_mixed, "real_cid": cid_real}


def _conv(db, cid):
    row = sqlite3.connect(db).execute("SELECT data FROM agent_conversations WHERE cid = ?", (cid,)).fetchone()
    return json.loads(row[0]) if row else None


def test_clean_removes_only_practice(mixed_db):
    db, real = mixed_db["db"], mixed_db["real"]
    conn = sqlite3.connect(db)
    plan = scan(conn)
    assert "E-3003" in plan["ids"] and "R-101" not in plan["ids"]
    counts = dict(summary(plan))
    assert counts["연습용 샘플 매물"] >= 20 and counts["샘플 매물이 나온 AI 상담 대화"] == 2
    assert counts["그 대화의 답변 평가(👍/👎)"] == 1 and counts["방문자 기억 속 샘플 매물 (관심·제외에서만 뺌)"] == 1
    assert counts["실시간 알림·이벤트"] >= 1

    result = clean(conn, db)
    assert Path(result["backup"]).exists() and Path(result["backup"]).parent.name == "backup"
    assert _conv(db, mixed_db["practice"]) is None  # 연습만 한 대화는 통째로
    mixed = _conv(db, mixed_db["mixed"])  # 섞인 대화는 실제 질문만 남김
    assert [t["question"] for t in mixed["turns"]] == ["진천군 공장 땅 있어?"] and mixed["history"] == []
    assert len(_conv(db, mixed_db["real_cid"])["turns"]) == 1  # 실제 상담은 그대로
    rows = conn.execute("SELECT listing_id FROM listings").fetchall()
    assert {r[0] for r in rows} == {"R-101", "R-102"}  # 실제 매물만 남음
    mem = json.loads(conn.execute("SELECT data FROM agent_memory WHERE owner = 'web:v1'").fetchone()[0])
    assert not mem["liked"]
    assert not [p for (p,) in conn.execute("SELECT payload FROM realtime_events") if "E-3003" in p]
    assert not any(n for _, n in summary(scan(conn)))  # 두 번 하면 지울 것이 없음
    # 백업에는 원래 기록이 그대로
    assert sqlite3.connect(result["backup"]).execute("SELECT COUNT(*) FROM agent_conversations").fetchone()[0] == 3


def test_cli(mixed_db, capsys):
    from real_estate_agent.cli import run_clean_practice

    settings = Settings(data_dir=mixed_db["real"].settings.data_dir, db_path=mixed_db["db"])
    assert run_clean_practice(settings, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "샘플 매물이 나온 AI 상담 대화: 2" in out and "아무것도 지우지 않았습니다" in out
    assert _conv(mixed_db["db"], mixed_db["practice"]) is not None
    assert run_clean_practice(settings, yes=True) == 0 and "정리했습니다" in capsys.readouterr().out
    assert _conv(mixed_db["db"], mixed_db["practice"]) is None
    assert run_clean_practice(settings, yes=True) == 0 and "정리할 것이 없습니다" in capsys.readouterr().out
    assert run_clean_practice(Settings(data_dir=SAMPLE, db_path=mixed_db["db"])) == 1  # 연습용 DB 는 대상 아님


def test_admin_page(mixed_db, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    app = create_app(Settings(data_dir=mixed_db["real"].settings.data_dir, db_path=mixed_db["db"],
                              log_dir=mixed_db["db"].parent / "logs", llm_providers=["rule"]))
    app.config["TESTING"] = True
    c = app.test_client()
    tok = lambda p: re.search(r'name="csrf" value="([^"]+)"', c.get(p).get_data(as_text=True)).group(1)  # noqa: E731
    assert "관리자 로그인" in c.get("/admin/clean-practice").get_data(as_text=True)
    c.post("/admin/login", data={"password": "office-pass-1", "csrf": tok("/admin")})
    page = c.get("/admin/clean-practice").get_data(as_text=True)
    assert "샘플 매물이 나온 AI 상담 대화" in page and "백업하고 연습 기록 지우기" in page
    assert "🧹 연습 기록 정리" in c.get("/admin").get_data(as_text=True)
    r = c.post("/admin/clean-practice", data={"csrf": tok("/admin/clean-practice")})
    assert r.status_code == 302
    admin_html = c.get("/admin").get_data(as_text=True)  # 정리 결과 안내 (한 번만 보임)
    assert "연습 기록" in admin_html and "정리했습니다" in admin_html and "data/backup" in admin_html.replace("\\", "/")
    assert "정리할 것이 없습니다" in c.get("/admin/clean-practice").get_data(as_text=True)
    assert anon_forbidden(app)


def anon_forbidden(app) -> bool:
    c = app.test_client()
    page = c.get("/admin/clean-practice").get_data(as_text=True)
    return "관리자 로그인" in page
