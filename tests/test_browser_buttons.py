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
  // 전체 목록 (PC 폭): 표가 옆으로 넘치지 않고 '🗑 지우기' 가 칸 안에 보임
  await pg.setViewportSize({ width: 1280, height: 900 }); await pg.goto(B + '/admin/inquiries');
  const fullList = await pg.evaluate(() => { const w = document.querySelector('.table-wrap'), d = document.querySelector('button[value=delete]');
    const db = d.getBoundingClientRect(), wb = w.getBoundingClientRect();
    return { scroll: w.scrollWidth - w.clientWidth, deleteInside: db.width > 0 && db.right <= wb.right + 1 }; });
  await pg.goto(B + '/admin');
  // 관리자 첫 화면: 칸마다 내용 밑 빈 여백이 똑같이 작음 (옆 칸에 맞춰 늘어나지 않음)
  const blanks = await pg.$$eval('.admin-cols .panel', ps => ps.map(p => { const k = p.lastElementChild.getBoundingClientRect();
    return Math.round(p.getBoundingClientRect().bottom - k.bottom); }));
  pg.on('dialog', d => d.accept());  // '지울까요?' 확인 창
  await Promise.all([pg.waitForNavigation(), pg.locator('.inq-list li').first().locator('button[value=delete]').click()]);
  const deleted = (await pg.content()).includes('지웠습니다');
  // AI 상담: 결과 밑 '▶' -> 매물 접수 폼으로, 찾던 조건이 문의 사항에 들어감
  await pg.goto(B + '/agent'); await pg.fill('[name=q]', '양평 땅 2개 보여줘');
  await Promise.all([pg.waitForLoadState('networkidle'), pg.locator('form.agent-ask button').first().click()]);
  await pg.waitForTimeout(1200);
  await pg.locator('a.no-match-btn').last().click(); await pg.waitForTimeout(900);
  const form = await pg.evaluate(() => { const r = document.querySelector('#inquiry').getBoundingClientRect();
    return { inView: r.top > -10 && r.top < innerHeight, msg: document.querySelector('#iq-message').value,
             focus: document.activeElement && document.activeElement.id }; });
  console.log(JSON.stringify({ posts, url: pg.url().replace(B, ''), msg: html.includes('처리 완료로 표시했습니다'), deleted, form, fullList, blanks }));
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
    assert res["msg"] and res["deleted"]
    assert res["fullList"] == {"scroll": 0, "deleteInside": True}
    assert res["blanks"] and max(res["blanks"]) <= 40  # '매물 추가·수정' 처럼 옆 칸 때문에 생기던 큰 빈 여백 없음  # 전체 목록에서도 지우기 버튼이 잘리지 않음
    assert any("op=delete" in body for url, body in res["posts"] if url == "/admin/inquiries/action")
    f = res["form"]  # AI 상담의 '▶': 폼이 화면에 보이고, 찾던 조건이 들어가고, 이름 칸에 커서
    assert f["inView"] and f["msg"].startswith("AI 상담에서 찾던 조건: 양평 땅 2개 보여줘") and f["focus"] == "iq-name"
