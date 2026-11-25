"""실제 브라우저에서 '버튼이 여러 개인 폼' 이 누른 버튼 값을 보내는지 (처리 완료 op=done, 별로예요 reason=…).
app.js 의 '두 번 눌림 방지' 가 버튼을 막으면서 값이 빠지던 문제 - 파이썬 테스트 클라이언트는 JS 를 돌리지 않아 못 잡음.
Node 용 Playwright 가 있는 곳(개발·CI)에서만 돌고, 없으면 건너뜀."""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

PW = Path("/opt/node22/lib/node_modules/playwright")
pytestmark = pytest.mark.skipif(not (shutil.which("node") and PW.exists()), reason="Node Playwright 없음")

SCRIPT = r"""
const { chromium } = require(process.argv[2]);
const B = process.argv[3];
(async () => {
  const b = await chromium.launch(); const pg = await (await b.newContext()).newPage();
  const posts = []; pg.on('request', r => { if (r.method() === 'POST') posts.push([r.url().replace(B, ''), decodeURIComponent(r.postData() || '')]); });
  await pg.goto(B + '/'); await pg.check('input[name=kind][value="목장"]');
  await pg.fill('#iq-name', '김철수'); await pg.fill('#iq-phone', '01024681357'); await pg.fill('#iq-message', '시험');
  await pg.check('input[name=agree]'); await Promise.all([pg.waitForNavigation(), pg.locator('#inquiry button').click()]);
  await pg.goto(B + '/admin'); await pg.fill('[name=password]', 'pw1234');
  await Promise.all([pg.waitForNavigation(), pg.locator('form button').first().click()]);
  await Promise.all([pg.waitForNavigation(), pg.locator('.inq-list li').first().locator('button[value=done]').click()]);
  const html = await pg.content();
  console.log(JSON.stringify({ posts, url: pg.url().replace(B, ''), msg: html.includes('처리 완료로 표시했습니다') }));
  await b.close();
})();
"""


def test_named_buttons_send_their_value_in_a_real_browser(tmp_path, monkeypatch):
    from werkzeug.serving import make_server

    from real_estate_agent.config import Settings
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "pw1234")
    data = tmp_path / "매물"
    shutil.copytree(Path(__file__).resolve().parent.parent / "sample_data" / "토지", data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "b.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        js = tmp_path / "b.js"
        js.write_text(SCRIPT, encoding="utf-8")
        out = subprocess.run(["node", str(js), str(PW), f"http://127.0.0.1:{srv.server_port}"], capture_output=True,
                             text=True, timeout=120, env={**os.environ})
    finally:
        srv.shutdown()
    assert out.returncode == 0, out.stderr[-800:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    action = [body for url, body in res["posts"] if url == "/admin/inquiries/action"]
    assert action and "op=done" in action[0]
    assert res["url"] == "/admin#inq-title" and res["msg"]
