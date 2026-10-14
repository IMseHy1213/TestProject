"""웹 서비스 테스트 (Flask 테스트 클라이언트, 실제 서버/AI 불필요)."""

import io
import re
import shutil
from pathlib import Path

import pytest

from real_estate_agent.config import Settings
from real_estate_agent.web.app import create_app, safe_filename

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def _make(password: str = ""):
        if password:
            monkeypatch.setenv("REA_ADMIN_PASSWORD", password)
        else:
            monkeypatch.delenv("REA_ADMIN_PASSWORD", raising=False)
        data = tmp_path / "토지"
        if not data.exists():
            shutil.copytree(SAMPLE, data)
        settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                            llm_providers=["rule"])
        app = create_app(settings)
        app.config["TESTING"] = True
        return app.test_client()
    return _make


def csrf(client, path="/"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def search(client, q):
    return client.post("/search", data={"q": q, "csrf": csrf(client)}, follow_redirects=True)


def act(client, act, text=""):
    return client.post("/action", data={"act": act, "text": text, "csrf": csrf(client, "/result")},
                       follow_redirects=True)


def test_index_has_search_and_no_name_field(make_client):
    html = make_client().get("/").get_data(as_text=True)
    assert 'name="q"' in html and "찾아보기" in html
    assert "이름" not in html  # 손님 이름 입력 없음
    assert "보유 토지 매물 <strong>22</strong>건" in html


def test_search_shows_first_then_next(make_client):
    c = make_client()
    html = search(c, "음성군채광이 좋은 2억원대 아파트를 보여줘").get_data(as_text=True)
    assert "1순위" in html and "음성군" in html and "2억원~2억 9,999만원" in html
    assert "아파트" in html  # 토지 외 매물 안내
    assert "210-3" not in html and "010-" not in html  # 지번·연락처 숨김
    html = act(c, "next").get_data(as_text=True)
    assert "2순위" in html
    html = act(c, "next", "채광이 별로예요").get_data(as_text=True)
    assert "3순위" in html and "중요도 ↑" in html  # 사유 학습 표시
    html = act(c, "prev").get_data(as_text=True)
    assert "2순위" in html


def test_refine_changes_condition(make_client):
    c = make_client()
    search(c, "양평 전원주택 부지 200평~300평 3억 이하")
    html = act(c, "refine", "좀 더 싼 걸로").get_data(as_text=True)
    assert "2억 5,500만원" in html  # 3억 * 0.85
    html = act(c, "refine", "마음에 안 들어요").get_data(as_text=True)
    assert "순위" in html or "해당 조건에 맞는 매물은 더 이상 없습니다." in html  # 조건 변경이 아니라 다음 순위


def test_explain_api_hides_lot_number(make_client):
    c = make_client()
    search(c, "음성군 채광 좋은 2억원대 땅")
    data = c.get("/api/explain").get_json()
    assert "음성군" in data["text"] and "210-3" not in data["text"]


def test_post_without_csrf_is_rejected(make_client):
    c = make_client()
    assert c.post("/search", data={"q": "양평"}).status_code == 400


def test_security_headers(make_client):
    r = make_client().get("/")
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Frame-Options"] == "DENY"


def test_admin_requires_configured_password(make_client):
    c = make_client()
    assert "관리자 비밀번호 정하기" in c.get("/admin").get_data(as_text=True)  # 처음: 비밀번호 정하는 화면
    r = c.post("/admin/login", data={"password": "", "csrf": csrf(c)})
    assert r.status_code == 400  # 비밀번호 미설정이면 로그인 불가
    assert c.post("/admin/sync", data={"csrf": csrf(c)}).status_code == 403


def test_first_time_setup_login_change_and_lockout(make_client):
    c = make_client()
    r = c.post("/admin/setup", data={"password": "short", "confirm": "short", "csrf": csrf(c, "/admin")})
    assert r.status_code == 400 and "8자 이상" in r.get_data(as_text=True)
    r = c.post("/admin/setup", data={"password": "office-pass-1", "confirm": "other-pass-1",
                                     "csrf": csrf(c, "/admin")})
    assert "서로 다릅니다" in r.get_data(as_text=True)
    before = csrf(c, "/admin")
    r = c.post("/admin/setup", data={"password": "office-pass-1", "confirm": "office-pass-1", "csrf": before},
               follow_redirects=True)
    html = r.get_data(as_text=True)
    assert "매물 현황" in html and "비밀번호를 정했습니다" in html  # 정하고 바로 로그인
    assert csrf(c, "/admin") != before  # 로그인하면 세션 토큰을 새로 만듦

    auth = c.application.config["ADMIN_AUTH"]
    stored = auth.path.read_text(encoding="utf-8")
    assert "office-pass-1" not in stored  # 비밀번호 그대로가 아니라 해시만 저장
    # 이미 정했으면 다시 '정하기' 로 덮어쓸 수 없음
    c.post("/admin/setup", data={"password": "hacker-pass-9", "confirm": "hacker-pass-9", "csrf": csrf(c, "/admin")})
    assert auth.check("office-pass-1") and not auth.check("hacker-pass-9")

    r = c.post("/admin/password", data={"current": "wrong-pass", "password": "new-pass-22", "confirm": "new-pass-22",
                                        "csrf": csrf(c, "/admin")}, follow_redirects=True)
    assert "지금 비밀번호가 맞지 않습니다" in r.get_data(as_text=True)
    r = c.post("/admin/password", data={"current": "office-pass-1", "password": "new-pass-22",
                                        "confirm": "new-pass-22", "csrf": csrf(c, "/admin")}, follow_redirects=True)
    assert "비밀번호를 바꿨습니다" in r.get_data(as_text=True) and auth.check("new-pass-22")

    c.post("/admin/logout", data={"csrf": csrf(c, "/admin")})
    assert "관리자 로그인" in c.get("/admin").get_data(as_text=True)
    auth._failures.clear()
    for _ in range(5):
        auth.record_failure("127.0.0.1")
    r = c.post("/admin/login", data={"password": "new-pass-22", "csrf": csrf(c, "/admin")})
    assert r.status_code == 429 and "잠겼습니다" in r.get_data(as_text=True)  # 맞는 비밀번호도 잠시 막힘
    auth._failures.clear()
    r = c.post("/admin/login", data={"password": "new-pass-22", "csrf": csrf(c, "/admin")})
    assert r.status_code == 302


def test_first_time_setup_only_from_server_pc(make_client):
    c = make_client()
    remote = {"REMOTE_ADDR": "192.168.0.23"}
    html = c.get("/admin", environ_base=remote).get_data(as_text=True)
    assert "서버를 켠 PC" in html and 'name="confirm"' not in html
    token = re.search(r'name="csrf" value="([^"]+)"', c.get("/", environ_base=remote).get_data(as_text=True)).group(1)
    r = c.post("/admin/setup", data={"password": "office-pass-1", "confirm": "office-pass-1", "csrf": token},
               environ_base=remote)
    assert r.status_code == 403 and not c.application.config["ADMIN_AUTH"].configured
    r = c.post("/admin/setup", data={"password": "office-pass-1", "confirm": "office-pass-1", "csrf": token},
               headers={"X-Forwarded-For": "1.2.3.4"})  # 인터넷 공개(프록시) 뒤에서 온 요청도 불가
    assert r.status_code == 403


def test_admin_login_upload_and_full_info(make_client):
    c = make_client("pw1234")
    r = c.post("/admin/login", data={"password": "wrong", "csrf": csrf(c, "/admin")})
    assert r.status_code == 401
    c.post("/admin/login", data={"password": "pw1234", "csrf": csrf(c, "/admin")})
    html = c.get("/admin").get_data(as_text=True)
    assert "매물 현황" in html and "중복" in html

    csv = "매물번호,소재지,지목,면적(평),매매가,도로,특이사항\nW-1,충청북도 음성군 삼성면 덕정리 5,대,200,2억 2천,4m 도로 접함,정남향\n"
    r = c.post("/admin/upload", data={"csrf": csrf(c, "/admin"),
                                       "file": (io.BytesIO(csv.encode("utf-8-sig")), "신규매물.csv")},
               content_type="multipart/form-data", follow_redirects=True)
    assert "업로드" in r.get_data(as_text=True)
    assert "<strong>23</strong>" in c.get("/").get_data(as_text=True)

    r = c.post("/admin/upload", data={"csrf": csrf(c, "/admin"), "file": (io.BytesIO(b"x"), "virus.exe")},
               content_type="multipart/form-data", follow_redirects=True)
    assert "저장할 매물 파일이 없습니다" in r.get_data(as_text=True)  # .exe 등은 저장하지 않음

    html = search(c, "음성군 채광 좋은 2억원대 땅").get_data(as_text=True)
    assert "관리자 정보" in html and "010-" in html  # 관리자는 연락처·지번 확인 가능


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Users\\x\\매물 목록.xlsx") == "매물 목록.xlsx"
    assert safe_filename('a<b>:"c".csv') == "abc.csv"


def test_admin_template_download(make_client):
    c = make_client("pw1234")
    assert c.get("/admin/template").status_code == 403
    c.post("/admin/login", data={"password": "pw1234", "csrf": csrf(c, "/admin")})
    r = c.get("/admin/template")
    assert r.status_code == 200 and r.data[:2] == b"PK"  # xlsx(zip)


def test_web_too_expensive_shows_cheaper(make_client):
    c = make_client()
    html = search(c, "양평 땅").get_data(as_text=True)
    price = lambda h: re.search(r'class="price">([^<]+)<', h).group(1).strip()
    from real_estate_agent.normalizer import parse_price_manwon
    prev = parse_price_manwon(price(html))
    for _ in range(4):
        html = act(c, "next", "너무 비싸요").get_data(as_text=True)
        if "해당 조건에 맞는 매물은 더 이상 없습니다." in html:
            assert "더 저렴한 매물만 찾았는데" in html
            break
        now = parse_price_manwon(price(html))
        assert now < prev and "저렴한 매물만 보여드려요" in html
        prev = now


def test_not_found_message(make_client):
    c = make_client()
    for q in ["부산 땅 보여줘", "문경시 임야", "양평 땅 100억 이상"]:
        html = search(c, q).get_data(as_text=True)
        assert "해당 조건에 맞는 매물이 없습니다." in html, q
        assert "1순위" not in html, q


def test_rate_limit_shows_reason_and_explain_is_cached(tmp_path, monkeypatch):
    from real_estate_agent.agent import RealEstateAgent
    from real_estate_agent.llm.base import LLMClient, LLMUnavailable

    class RateLimited(LLMClient):
        name = "openai"
        model = "fake"
        calls = 0

        def complete(self, system, user, max_tokens=2000):
            RateLimited.calls += 1
            if "추천 설명" in system:
                return "AI 설명입니다."
            raise LLMUnavailable("분당 한도", retry_after=30, kind="rate_limit")

    monkeypatch.delenv("REA_ADMIN_PASSWORD", raising=False)
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs", llm_providers=["rule"])
    agent = RealEstateAgent(settings)
    agent.llm.clients = [RateLimited()]
    app = create_app(settings, agent=agent)
    app.config["TESTING"] = True
    c = app.test_client()
    search(c, "음성군 2억원대 땅")  # 조건 추출에서 429 -> 30초 쉼
    html = c.get("/").get_data(as_text=True)
    assert "AI 잠시 쉬는 중" in html and "분당 사용 한도" in html

    agent.llm.cooldown_until.clear()
    first = c.get("/api/explain").get_json()
    n = RateLimited.calls
    second = c.get("/api/explain").get_json()  # 같은 매물은 다시 GPT 를 부르지 않음
    assert first == second and RateLimited.calls == n
