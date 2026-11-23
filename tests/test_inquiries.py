"""매물 접수 폼: 첫 화면 사무실 안내 아래 → 검사 → 저장 → 문자(솔라피, 시험에서는 가짜 서버)."""

import hashlib
import hmac
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from real_estate_agent import inquiries
from real_estate_agent.config import Settings
from real_estate_agent.web.app import create_app

SAMPLE_LAND = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
GOOD = {"purpose": "찾아요", "kind": "공장", "name": "홍길동", "phone": "010 2468 1357", "agree": "1",
        "message": "음성군 대소면 쪽 임대 공장 200평 정도 찾습니다"}


class FakeSolapi:
    def __init__(self, status=200, body=None):
        self.calls, self.status, self.body = [], status, body or {"statusCode": "2000", "statusMessage": "정상 접수"}

    def __call__(self, url, json=None, headers=None, **kw):
        self.calls.append({"url": url, "json": json, "headers": headers})
        outer = self

        class R:
            status_code = outer.status

            def json(self):
                return outer.body
        return R()


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("REA_ADMIN_PASSWORD", "pw1234")
    for k, v in {"SMS_TO": "010-9999-0000", "SMS_FROM": "043-877-5242", "SOLAPI_API_KEY": "NCSTESTKEY1234",
                 "SOLAPI_API_SECRET": "SECRETTESTVALUE123"}.items():
        monkeypatch.setenv(k, v)
    fake = FakeSolapi()
    import requests

    monkeypatch.setattr(requests, "post", lambda url, **kw: fake(url, **kw))
    # 실제 운영처럼(연습 모드가 아님) 매물 폴더를 임시 위치에
    data = tmp_path / "매물"
    shutil.copytree(SAMPLE_LAND, data / "토지")
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "i.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    app.config["TESTING"] = True
    return app, app.test_client(), fake


def token(c, path="/"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def submit(c, data):
    return c.post("/inquiry", data={**data, "csrf": token(c)}, follow_redirects=True).get_data(as_text=True)


def wait_sent(app, n=1):
    import threading
    import time

    for _ in range(100):
        if not [t for t in threading.enumerate() if t.name == "inquiry-sms"]:
            break
        time.sleep(0.02)
    return inquiries.recent(app.config["AGENT"].storage.conn)


def test_form_is_on_first_page_right_below_office_info(web):
    _, c, _ = web
    html = c.get("/").get_data(as_text=True)
    office, form = html.index('aria-label="사무실 안내"'), html.index('id="inquiry"')
    assert office < form and "영업 시간" in html[office:form]
    for k in ("토지", "공장", "아파트"):
        assert f'value="{k}"' in html
    assert 'name="name"' in html and 'name="phone"' in html and 'name="message"' in html and "제출하기" in html
    assert 'href="#inquiry" class="nav-inquiry"' in html  # 위쪽 '📝 매물 접수' 버튼으로 바로 이동


def test_form_on_every_customer_page_not_admin(web):
    """첫 화면뿐 아니라 AI 상담·검색 결과에도 (위치·문의 전화·영업 시간 바로 아래), 관리자 화면에는 없음."""
    app, c, fake = web
    for path in ("/", "/agent"):
        html = c.get(path).get_data(as_text=True)
        office, form = html.index('aria-label="사무실 안내"'), html.index('id="inquiry"')
        assert office < form and 'class="inquiry-form"' in html, path
    tok = token(c)
    html = c.post("/search", data={"q": "음성군 땅", "csrf": tok}, follow_redirects=True).get_data(as_text=True)
    assert 'class="inquiry-form"' in html
    c.post("/admin/login", data={"password": "pw1234", "csrf": token(c, "/admin")})
    assert 'id="inquiry"' not in c.get("/admin").get_data(as_text=True)
    assert 'id="inquiry"' not in c.get("/admin/inquiries").get_data(as_text=True)


def test_submit_from_ai_consult_returns_there(web):
    app, c, fake = web
    tok = token(c, "/agent")
    r = c.post("/inquiry", data={**GOOD, "csrf": tok, "back": "/agent"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/agent#inquiry")
    assert "접수되었습니다" in c.get("/agent").get_data(as_text=True)
    r = c.post("/inquiry", data={**GOOD, "csrf": token(c), "back": "//evil.example/x"})
    assert r.headers["Location"].endswith("/#inquiry")  # 바깥 주소로는 돌려보내지 않음


def test_static_files_have_version(web):
    _, c, _ = web
    html = c.get("/").get_data(as_text=True)
    assert re.search(r'style\.css\?v=\d+', html)  # 고친 뒤 브라우저가 옛 화면 파일을 쓰지 않게


def test_submit_saves_and_texts_the_office(web):
    app, c, fake = web
    html = submit(c, GOOD)
    assert "접수되었습니다" in html
    rows = wait_sent(app)
    assert len(rows) == 1 and rows[0]["phone"] == "010-2468-1357" and rows[0]["sms_status"] == "sent"
    call = fake.calls[0]
    assert call["url"] == inquiries.SOLAPI_URL
    msg = call["json"]["message"]
    assert (msg["to"], msg["from"]) == ("01099990000", "0438775242")
    assert "[매물 접수 - 찾아요] 공장" in msg["text"] and "홍길동" in msg["text"] and "010-2468-1357" in msg["text"]
    assert "200평" in msg["text"]
    assert call["headers"]["Authorization"].startswith("HMAC-SHA256 apiKey=NCSTESTKEY1234, date=")


@pytest.mark.parametrize("bad, err", [
    ({"kind": ""}, "매물 종류"), ({"purpose": ""}, "내놓으시는지"), ({"name": ""}, "이름"), ({"phone": "123"}, "전화번호"),
    ({"message": ""}, "문의 사항"), ({"agree": ""}, "동의"), ({"message": "가" * 501}, "500자"),
])
def test_validation(web, bad, err):
    app, c, fake = web
    html = submit(c, {**GOOD, **bad})
    assert err in html and "접수되었습니다" not in html
    assert inquiries.recent(app.config["AGENT"].storage.conn) == [] and not fake.calls
    if bad.get("name") is None:
        assert "홍길동" in html  # 다시 쓰지 않아도 되게 적은 내용은 남겨 둠


def test_spam_and_rate_limit(web):
    app, c, fake = web
    assert "접수되었습니다" in submit(c, {**GOOD, "website": "http://spam"})
    assert inquiries.recent(app.config["AGENT"].storage.conn) == []  # 자동 입력: 저장·문자 안 함
    for _ in range(3):
        assert "접수되었습니다" in submit(c, GOOD)
    assert "너무 잦습니다" in submit(c, GOOD)
    wait_sent(app)
    assert len(fake.calls) == 3


def test_sms_failure_keeps_inquiry_and_admin_can_resend(web):
    app, c, fake = web
    fake.status, fake.body = 400, {"errorCode": "InvalidSender", "errorMessage": "발신번호 미등록"}
    submit(c, GOOD)
    row = wait_sent(app)[0]
    assert row["sms_status"] == "failed" and "발신번호 미등록" in row["sms_detail"]
    c.post("/admin/login", data={"password": "pw1234", "csrf": token(c, "/admin")})
    admin = c.get("/admin").get_data(as_text=True)
    assert "📝 매물 접수 (1건 새로)" in admin
    page = c.get("/admin/inquiries").get_data(as_text=True)
    assert "홍길동" in page and "010-2468-1357" in page and "❌ 실패" in page
    fake.status, fake.body = 200, {"statusMessage": "정상 접수"}
    c.post("/admin/inquiries/action", data={"op": "resend", "id": row["id"], "csrf": token(c, "/admin/inquiries")})
    assert inquiries.recent(app.config["AGENT"].storage.conn)[0]["sms_status"] == "sent"
    c.post("/admin/inquiries/action", data={"op": "done", "id": row["id"], "csrf": token(c, "/admin/inquiries")})
    c.post("/admin/inquiries/action", data={"op": "delete", "id": row["id"], "csrf": token(c, "/admin/inquiries")})
    assert inquiries.recent(app.config["AGENT"].storage.conn) == []


def test_customer_cannot_see_inquiries(web):
    app, c, _ = web
    submit(c, GOOD)
    wait_sent(app)
    assert "010-2468-1357" not in c.get("/").get_data(as_text=True)
    assert "홍길동" not in c.get("/admin/inquiries").get_data(as_text=True)  # 로그인 화면


def test_no_sms_settings_and_practice_mode(tmp_path, monkeypatch):
    for k in ("SMS_TO", "SMS_FROM", "SOLAPI_API_KEY", "SOLAPI_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    from real_estate_agent.config import SAMPLE_DATA_DIR

    for data, expected in ((tmp_path / "매물", "off"), (SAMPLE_DATA_DIR, "practice")):
        if data != SAMPLE_DATA_DIR:
            shutil.copytree(SAMPLE_LAND, data / "토지")
        app = create_app(Settings(data_dir=data, db_path=tmp_path / f"{expected}.db", log_dir=tmp_path / "l",
                                  llm_providers=["rule"]))
        c = app.test_client()
        assert "접수되었습니다" in submit(c, GOOD)
        assert wait_sent(app)[0]["sms_status"] == expected


def test_save_sms_settings_to_key_file(tmp_path, monkeypatch):
    import real_estate_agent.config as config
    from real_estate_agent.setup_tools import save_sms_settings

    key_file = tmp_path / "my_api_key.py"
    key_file.write_text('OPENAI_API_KEY = "sk-keep"\nSMS_TO = ""\n', encoding="utf-8")
    monkeypatch.setattr(config, "KEY_FILE", key_file)
    for k in ("SMS_TO", "SMS_FROM", "SOLAPI_API_KEY", "SOLAPI_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    save_sms_settings("01075810000", "043 877 5242", "NCSABCDEF12345", "SECRET1234567890")
    text = key_file.read_text(encoding="utf-8")
    assert 'SMS_TO = "010-7581-0000"' in text and 'SMS_FROM = "043-877-5242"' in text
    assert 'OPENAI_API_KEY = "sk-keep"' in text and 'SOLAPI_API_SECRET = "SECRET1234567890"' in text
    with pytest.raises(ValueError):
        save_sms_settings("12345")


def test_auth_header_and_phone_format():
    now = datetime(2026, 11, 23, 5, 0, 0, tzinfo=timezone.utc)
    h = inquiries.solapi_auth_header("KEY", "SECRET", now, salt="abc")
    sig = hmac.new(b"SECRET", b"2026-11-23T05:00:00Zabc", hashlib.sha256).hexdigest()
    assert h == f"HMAC-SHA256 apiKey=KEY, date=2026-11-23T05:00:00Z, salt=abc, signature={sig}"
    assert inquiries.normalize_phone("+82 10-1234-5678") == "010-1234-5678"
    assert inquiries.normalize_phone("0438775242") == "043-877-5242"
    assert inquiries.normalize_phone("02-123-4567") == "02-123-4567"
    assert inquiries.normalize_phone("1234") == ""


def test_owner_can_list_own_property(web):
    """'매물을 내놓아요': 자기 부동산을 사무실에 내놓는 손님도 접수, 문자와 관리자 목록에 구분이 보임."""
    app, c, fake = web
    page = c.get("/").get_data(as_text=True)
    assert 'name="purpose" value="내놓아요"' in page and "찾는(내놓는) 매물 종류" in page
    submit(c, {**GOOD, "purpose": "내놓아요", "kind": "아파트", "message": "진천읍 아파트 32평 매매로 내놓고 싶어요"})
    rows = wait_sent(app)
    assert rows[0]["purpose"] == "내놓아요" and rows[0]["kind"] == "아파트"
    assert fake.calls[0]["json"]["message"]["text"].startswith("[매물 접수 - 내놓아요] 아파트")
    c.post("/admin/login", data={"password": "pw1234", "csrf": token(c, "/admin")})
    assert "내놓아요" in c.get("/admin/inquiries").get_data(as_text=True)


def test_old_inquiry_table_gets_purpose_column(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, kind TEXT NOT NULL,"
                 " name TEXT NOT NULL, phone TEXT NOT NULL, message TEXT NOT NULL, sms_status TEXT NOT NULL DEFAULT 'pending',"
                 " sms_detail TEXT NOT NULL DEFAULT '', handled INTEGER NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO inquiries (created, kind, name, phone, message) VALUES ('2026-11-20T10:00:00','토지','김','010-1111-2222','x')")
    rows = inquiries.recent(conn)
    assert rows[0]["purpose"] == "찾아요"
