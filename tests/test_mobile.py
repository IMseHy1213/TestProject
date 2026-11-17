"""휴대폰에서 쓰기: 같은 와이파이 주소·QR 코드, '홈 화면에 추가'(manifest), 휴대폰 폭 화면."""

import json
import re
import shutil
from pathlib import Path

import pytest

from real_estate_agent.config import Settings
from real_estate_agent.mobile import _private, phone_urls, qr_svg, qr_text
from real_estate_agent.web.app import create_app

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REA_ADMIN_PASSWORD", "pw1234")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "m.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    app.config["TESTING"] = True
    return app, app.test_client()


def token(c, path):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_private_addresses_only():
    assert _private("192.168.0.12") and _private("10.0.0.5") and _private("172.16.3.4")
    assert not _private("127.0.0.1") and not _private("8.8.8.8") and not _private("169.254.1.1") and not _private("x")
    assert phone_urls(8000, ["192.168.0.12"]) == ["http://192.168.0.12:8000"]


def test_qr_codes():
    svg = qr_svg("http://192.168.0.12:8000")
    assert svg.startswith("<svg") and "viewBox" in svg
    assert "█" in qr_text("http://192.168.0.12:8000")


def test_manifest_and_icons_for_home_screen(client):
    app, c = client
    html = c.get("/").get_data(as_text=True)
    assert 'rel="manifest"' in html and 'name="viewport"' in html and "apple-touch-icon" in html
    r = c.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.mimetype == "application/manifest+json"
    m = json.loads(r.get_data(as_text=True))
    assert m["display"] == "standalone" and m["short_name"] == "매물 추천" and m["start_url"] == "/"
    for icon in m["icons"]:
        assert c.get(icon["src"]).status_code == 200
    assert c.get("/static/icons/apple-touch-icon.png").status_code == 200


def test_admin_phone_panel(client):
    app, c = client
    c.post("/admin/login", data={"password": "pw1234", "csrf": token(c, "/admin")})
    html = c.get("/admin").get_data(as_text=True)
    assert "📱 휴대폰으로 보기" in html and "python main.py mobile" in html  # 이 PC 에서만 열려 있을 때: 방법 안내
    app.config["BIND_HOST"], app.config["BIND_PORT"] = "0.0.0.0", 8000  # python main.py mobile 로 켰을 때
    import real_estate_agent.mobile as mobile

    real = mobile.lan_addresses
    mobile.lan_addresses = lambda: ["192.168.0.12"]
    try:
        html = c.get("/admin").get_data(as_text=True)
    finally:
        mobile.lan_addresses = real
    assert "http://192.168.0.12:8000" in html and '<div class="phone-qr"><svg' in html


def test_admin_listing_rows_have_labels_for_phone_cards(client):
    app, c = client
    c.post("/admin/login", data={"password": "pw1234", "csrf": token(c, "/admin")})
    html = c.get("/admin/listings").get_data(as_text=True)
    assert 'data-label="가격"' in html and 'class="table-wrap listings-wrap"' in html


def test_cli_has_mobile_command():
    from real_estate_agent.cli import build_parser

    args = build_parser().parse_args(["mobile", "--port", "8010", "--no-browser"])
    assert args.command == "mobile" and args.port == 8010 and args.no_browser
