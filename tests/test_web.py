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
    assert "순위" in html  # 조건 변경이 아니라 다음 순위


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
    assert "REA_ADMIN_PASSWORD" in c.get("/admin").get_data(as_text=True)
    r = c.post("/admin/login", data={"password": "", "csrf": csrf(c)})
    assert r.status_code == 401  # 비밀번호 미설정이면 로그인 불가
    assert c.post("/admin/sync", data={"csrf": csrf(c)}).status_code == 403


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
    assert "지원하지 않는 형식" in r.get_data(as_text=True)

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
        if "더 보여드릴 매물이 없습니다" in html:
            assert "더 저렴한 매물만 찾았는데" in html
            break
        now = parse_price_manwon(price(html))
        assert now < prev and "저렴한 매물만 보여드려요" in html
        prev = now
