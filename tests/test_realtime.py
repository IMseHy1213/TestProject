"""v0.9 실시간 · 이벤트 기반 Agent 테스트: 이벤트 버스, 변경 감지, 매칭 워커, 폴더 감시, SSE, WebSocket."""

import json
import re
import shutil
import time
from pathlib import Path

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.listing_editor import parse_form
from real_estate_agent.realtime import EventBus, RealtimeHub
from real_estate_agent.tool_agent.service import AgentService

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
WAREHOUSE = {"address": "충청북도 음성군 감곡면 오향리 900", "jimok": "창고용지", "zoning": "계획관리지역", "area": "400",
             "area_unit": "pyeong", "price": "2억", "road_access": "8m 도로 접함", "status": "거래가능",
             "description": "● 창고 부지, 대형차 진입 가능"}


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    agent = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "rt.db", log_dir=tmp_path / "logs",
                                     llm_providers=["rule"]))
    agent.sync()
    svc = AgentService(agent)
    hub = RealtimeHub(agent, lock=svc.lock, interval=0)
    svc.ask("web:A", None, "음성군 창고 지을 땅")     # A 는 음성군 창고 부지를 찾았음 (장기 기억)
    svc.ask("web:B", None, "양평 전원주택 땅")        # B 는 양평
    return agent, svc, hub, data


# ---------------------------------------------------------------- 이벤트 버스 (pub-sub)
def test_bus_filters_by_owner_dedups_and_replays(tmp_path):
    import sqlite3

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    bus = EventBus(conn)
    qa, qb, qadmin = bus.subscribe("A"), bus.subscribe("B"), bus.subscribe(see_all=True)
    e1 = bus.publish("listing.added", {"x": 1})
    e2 = bus.publish("alert.match", {"x": 2}, owner="A", dedup="A|1")
    assert bus.publish("alert.match", {"x": 2}, owner="A", dedup="A|1") is None  # 같은 알림은 한 번만
    assert [qa.get_nowait().id, qa.get_nowait().id] == [e1.id, e2.id]
    assert qb.get_nowait().id == e1.id and qb.empty()          # B 에게 A 의 알림은 안 감
    assert qadmin.qsize() == 2
    assert [e.id for e in bus.since(e1.id, "A")] == [e2.id]     # 다시 연결하면 놓친 것만
    assert bus.since(e1.id, "B") == [] and "owner" not in e2.public()
    bus.unsubscribe(qa)
    assert bus.subscribers == 2


# ---------------------------------------------------------------- 변경 감지 → 매칭 → 맞춤 알림
def test_new_listing_and_price_drop_alert_only_matching_visitor(setup):
    agent, svc, hub, _ = setup
    new_id, _ = agent.add_listing(parse_form(WAREHOUSE).values)
    drop_id = next(l.listing_id for l in agent.listings() if "음성군" in l.address and l.listing_id != new_id)
    old = agent.find_listing(drop_id).price_manwon
    agent.edit_listing(drop_id, {"price_manwon": old - 3000})
    events = hub.detect_changes("관리자")
    assert [e.topic for e in events] == ["listing.added", "listing.price_drop"]
    assert events[1].payload["old_price_manwon"] == old
    hub.drain()
    alerts = [e for e in hub.bus.recent(10, see_all=True) if e.topic == "alert.match"]
    assert {a.owner for a in alerts} == {"web:A"}               # 양평을 찾던 B 에게는 안 감
    assert {a.payload["kind"] for a in alerts} == {"새 매물", "가격 인하"}
    assert all("음성군 · 창고 용도" in a.payload["message"] and a.payload["latency_ms"] >= 0 for a in alerts)
    assert "010-" not in json.dumps([a.payload for a in alerts], ensure_ascii=False)  # 연락처 없음
    # 같은 변경을 다시 감지해도 알림이 또 가지 않음 (멱등성)
    hub._snapshot.pop(new_id)
    hub.detect_changes("관리자")
    hub.drain()
    assert len([e for e in hub.bus.recent(20, see_all=True) if e.topic == "alert.match"]) == 2


def test_rejected_listing_is_not_alerted(setup):
    agent, svc, hub, _ = setup
    target = next(l for l in agent.listings() if "음성군" in l.address)
    svc.memory.update_after("web:A", type("C", (), {"rejected": {target.listing_id: "싫어요"}, "liked": []})(), [])
    agent.edit_listing(target.listing_id, {"price_manwon": target.price_manwon - 1000})
    hub.detect_changes("관리자")
    hub.drain()
    assert not [e for e in hub.bus.recent(10, see_all=True) if e.topic == "alert.match"]


def test_status_change_and_removal_events(setup):
    agent, _, hub, _ = setup
    lid = agent.listings()[0].listing_id
    agent.edit_listing(lid, {"status": "보류"})
    agent.edit_listing(agent.listings()[1].listing_id, {}, hidden=True)
    topics = [e.topic for e in hub.detect_changes("관리자")]
    assert "listing.changed" in topics and "listing.removed" in topics


def test_folder_watch_picks_up_new_excel_file(setup):
    agent, _, hub, data = setup
    assert hub.check_folder() == []  # 바뀐 게 없으면 다시 읽지 않음
    wb = Workbook()
    ws = wb.active
    ws.append(["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "특이사항"])
    ws.append(["N-900", "충청북도 음성군 대소읍 삼정리 77", "창고용지", "계획관리지역", 500, "2억 5천", "물류창고 부지"])
    wb.save(data / "새매물.xlsx")
    events = hub.check_folder()
    assert [e.topic for e in events] == ["listing.added"] and events[0].payload["listing"]["office_no"] == "N-900"
    assert events[0].payload["source"] == "매물 폴더" and hub.stats.syncs == 1
    hub.drain()
    alert = next(e for e in hub.bus.recent(5, see_all=True) if e.topic == "alert.match")
    assert alert.owner == "web:A" and "N-900" in alert.payload["message"]


def test_worker_threads_process_jobs(setup):
    agent, _, hub, _ = setup
    hub.n_workers = 2
    hub.start()
    try:
        agent.add_listing(parse_form(WAREHOUSE).values)
        hub.detect_changes("관리자")
        hub.drain(timeout=5)
        assert hub.stats.alerts == 1 and hub.stats.jobs_done == 1 and hub.stats.as_dict()["match_ms_p50"] is not None
    finally:
        hub.stop()
    assert not hub.running


# ---------------------------------------------------------------- 웹: 관리자 변경 → 손님 화면 실시간 알림 (SSE)
@pytest.fixture
def web(tmp_path, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                              llm_providers=["rule"]))
    app.config["TESTING"] = True
    return app


def token(c, path):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def read_events(resp, n):
    """SSE 응답에서 이벤트 n 개만 읽고 연결을 닫음."""
    out, buf = [], ""
    for chunk in resp.response:
        buf += chunk.decode() if isinstance(chunk, bytes) else chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            if block.startswith("id:"):
                lines = dict(line.split(": ", 1) for line in block.splitlines())
                out.append((lines["event"], json.loads(lines["data"])))
        if len(out) >= n:
            break
    resp.close()
    return out


def test_admin_change_reaches_customer_stream(web):
    customer, admin = web.test_client(), web.test_client()
    customer.post("/agent/ask", data={"q": "음성군 창고 지을 땅", "csrf": token(customer, "/agent")})
    admin.post("/admin/login", data={"password": "office-pass-1", "csrf": token(admin, "/admin")})
    start = web.config["REALTIME"].bus.last_id()
    admin.post("/admin/listings/new", data={**WAREHOUSE, "csrf": token(admin, "/admin/listings")})
    web.config["REALTIME"].drain()

    got = read_events(customer.get(f"/events/stream?since={start}", buffered=False), 1)
    assert got[0][0] == "alert.match" and got[0][1]["payload"]["kind"] == "새 매물"   # 손님에겐 맞춤 알림만
    seen_by_admin = read_events(admin.get(f"/events/stream?since={start}", buffered=False), 2)
    assert [t for t, _ in seen_by_admin] == ["listing.added", "alert.match"]           # 관리자는 전체 흐름
    html = admin.get("/admin").get_data(as_text=True)
    assert "실시간 이벤트" in html and "listing.added" in html and "맞춤 알림 1건" in html
    assert 'src="/static/realtime.js"' in customer.get("/").get_data(as_text=True)


def test_alert_link_prefills_agent_question(web):
    c = web.test_client()
    html = c.get("/agent?ask=E-3004 매물 자세히 알려줘").get_data(as_text=True)
    assert 'value="E-3004 매물 자세히 알려줘"' in html


# ---------------------------------------------------------------- API: SSE · WebSocket (실시간 입력 → 단계별 응답)
@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient

    from real_estate_agent.api import create_api

    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    agent = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "a.db", log_dir=tmp_path / "logs",
                                     llm_providers=["rule"]))
    agent.sync()
    app = create_api(agent=agent)
    return app, TestClient(app)


def test_api_websocket_events_and_realtime_question(api):
    app, client = api
    with client.websocket_connect("/api/v1/ws?visitor_id=visitor-ws-01") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["visitor_id"] == "visitor-ws-01"
        app.state.realtime.bus.publish("listing.added", {"listing": {"name": "테스트 매물"}})
        msg = ws.receive_json()
        while msg["type"] == "ping":
            msg = ws.receive_json()
        assert msg["type"] == "event" and msg["event"]["topic"] == "listing.added"

        ws.send_json({"type": "ask", "question": "양평 땅 2개 비교해줘"})
        kinds, answer = [], None
        while answer is None:
            m = ws.receive_json()
            if m["type"] == "agent":
                kinds.append(m["kind"])
                if m["kind"] == "step" and m["step"]["kind"] == "tool":
                    kinds.append(m["step"]["tool"])
            elif m["type"] == "answer":
                answer = m
        assert "search_listings" in kinds and "compare_listings" in kinds and "tool_start" in kinds and "done" in kinds
        assert len(answer["listings"]) == 2 and answer["conversation_id"]
    stats = client.get("/api/v1/realtime/stats").json()
    assert stats["events"]["listing.added"] == 1


def test_api_event_stream_replays_since(api):
    app, client = api
    bus = app.state.realtime.bus
    start = bus.last_id()
    bus.publish("listing.price_drop", {"listing": {"name": "A"}})
    bus.publish("alert.match", {"message": "다른 사람 알림"}, owner="api:someone-else-1")
    r = client.get(f"/api/v1/events/stream?since={start}&max_seconds=1", headers={"X-Visitor-Id": "visitor-sse-01"})
    events = [line for line in r.text.splitlines() if line.startswith("event: ")]
    assert events == ["event: listing.price_drop"]  # 남의 맞춤 알림은 오지 않음
    assert "retry: 5000" in r.text


def test_admin_written_files_do_not_trigger_extra_folder_reads(setup):
    agent, _, hub, _ = setup
    agent.add_listing(parse_form(WAREHOUSE).values)  # 웹입력 엑셀에 저장됨
    hub.detect_changes("관리자 화면")
    assert hub.check_folder() == [] and hub.stats.syncs == 0
