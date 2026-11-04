"""v0.3 State & Memory (대화 저장·체크포인트·장기 기억) + v0.1 보완 (FastAPI · async · 스트리밍) 테스트."""

import json
import re
import shutil
from pathlib import Path

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import LLMUnavailable, ToolCall, ToolTurn
from real_estate_agent.tool_agent import AgentContext
from real_estate_agent.tool_agent.memory import (dump_messages, is_incomplete, load_ctx, load_messages,
                                                 repair_for_resume)
from real_estate_agent.tool_agent.service import AgentService, Busy

from test_tool_agent import ScriptedLLM, call, router

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


def make_agent(tmp_path, db="s.db"):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / db, log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


# ---------------------------------------------------------------- 직렬화·복구 도우미
def test_message_and_context_roundtrip():
    msgs = [{"role": "user", "content": "양평 땅"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("c1", "search_listings", {"query": "양평"})],
             "raw": {"openai": [{"type": "function_call", "call_id": "c1"}]}},
            {"role": "tool", "call_id": "c1", "name": "search_listings", "content": "{}", "is_error": False}]
    back = load_messages(json.loads(json.dumps(dump_messages(msgs), ensure_ascii=False)))
    assert back[1]["tool_calls"][0] == msgs[1]["tool_calls"][0] and back[1]["raw"] == msgs[1]["raw"]
    ctx = AgentContext(rejected={"A": "비쌈"}, last_shown=["A", "B"], last_search={"query": "양평"})
    assert load_ctx(json.loads(json.dumps(__import__("dataclasses").asdict(ctx)))) == ctx
    assert not is_incomplete(msgs + [{"role": "assistant", "content": "답"}]) and is_incomplete(msgs)


def test_repair_drops_tool_call_without_results():
    cut = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "tool_calls": [ToolCall("a", "x"), ToolCall("b", "y")]},
           {"role": "tool", "call_id": "a", "name": "x", "content": "{}"}]  # b 결과 전에 끊김
    assert repair_for_resume(cut) == cut[:1]
    ok = cut + [{"role": "tool", "call_id": "b", "name": "y", "content": "{}"}]
    assert repair_for_resume(ok) == ok


# ---------------------------------------------------------------- 대화 저장 (서버를 다시 켜도 이어짐)
def test_conversation_survives_restart_and_follow_up_still_works(tmp_path):
    svc = AgentService(make_agent(tmp_path))
    cid, first = svc.ask("web:A", None, "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    shown = first["listing_ids"]
    assert len(shown) == 2

    restarted = AgentService(make_agent(tmp_path))  # 같은 DB 파일, 새 프로그램
    conv = restarted.open("web:A", cid)
    assert conv["cid"] == cid and len(conv["turns"]) == 1 and conv["ctx"].last_shown == shown
    _, second = restarted.ask("web:A", cid, "그 중에 넓은 건?")
    agent = restarted.agent
    assert second["listing_ids"] == [max(shown, key=lambda i: agent.find_listing(i).area_m2)]
    assert len(restarted.open("web:A", cid)["turns"]) == 2


def test_other_visitor_cannot_open_conversation(tmp_path):
    svc = AgentService(make_agent(tmp_path))
    cid, _ = svc.ask("web:A", None, "양평 땅")
    other = svc.open("web:B", cid)
    assert other["cid"] != cid and other["turns"] == []


def test_busy_guard(tmp_path):
    svc = AgentService(make_agent(tmp_path))
    conv = svc.open("web:A", None)
    svc.store.save(conv["cid"], "web:A", [], AgentContext(), [], "running", "앞 질문")
    with pytest.raises(Busy):
        svc.ask("web:A", conv["cid"], "새 질문")


# ---------------------------------------------------------------- 체크포인트 → 이어서 답하기
def test_checkpoint_and_resume_after_crash(tmp_path):
    agent = make_agent(tmp_path)
    agent.llm = router(ScriptedLLM([call("search_listings", query="양평 땅", limit=2),
                                    RuntimeError("서버가 꺼짐")]))  # 검색 결과를 받은 뒤 죽음
    svc = AgentService(agent)
    with pytest.raises(RuntimeError):
        svc.ask("web:A", "c" * 20, "양평 땅 추천해줘")
    conv = svc.open("web:A", "c" * 20)
    assert conv["status"] == "interrupted" and conv["pending"] == "양평 땅 추천해줘"
    assert svc.needs_resume(conv) and conv["history"][-1]["role"] == "tool"  # 검색 결과까지 저장됨

    replay = ScriptedLLM([ToolTurn(text="양평군 매물 2건을 찾았습니다.")])
    agent.llm = router(replay)
    _, turn = svc.resume("web:A", "c" * 20)
    assert turn["answer"] == "양평군 매물 2건을 찾았습니다." and turn["steps"][0]["kind"] == "resume"
    assert not any(s["kind"] == "tool" for s in turn["steps"])  # 검색을 다시 하지 않고 저장된 결과 사용
    assert replay.seen[0][-1]["role"] == "tool"  # 모델은 저장된 도구 결과를 받음
    done = svc.open("web:A", "c" * 20)
    assert done["status"] == "idle" and done["pending"] is None and len(done["turns"]) == 1


# ---------------------------------------------------------------- 장기 기억
def test_long_term_memory_across_conversations(tmp_path):
    agent = make_agent(tmp_path)
    first_id = None

    def reject_first(messages):
        nonlocal first_id
        first_id = json.loads(messages[-1]["content"])["result"]["매물"][0]["id"]
        return call("record_feedback", listing_id=first_id, action="reject", reason="너무 비싸요")

    agent.llm = router(ScriptedLLM([call("search_listings", query="양평 전원주택 땅", limit=3), reject_first,
                                    ToolTurn(text="다음에는 빼고 보여드릴게요.")]))
    svc = AgentService(agent)
    cid, _ = svc.ask("web:A", None, "양평 전원주택 땅 보여줘. 첫 번째는 너무 비싸요")
    svc.reset(cid)  # 새 상담

    view = svc.memory_view("web:A")
    assert view["searches"] and len(view["rejected"]) == 1

    agent.llm = router()  # AI 없이 (규칙 기반) - 같은 기억을 씀
    _, turn = svc.ask("web:A", None, "지난번에 찾던 거 다시 보여줘")
    tools = [s["tool"] for s in turn["steps"] if s["kind"] == "tool"]
    assert tools[:2] == ["recall_memory", "search_listings"]
    assert turn["steps"][0]["kind"] == "memory"
    assert "지난 상담에서 찾으신 조건" in turn["answer"] and "양평군" in turn["answer"]
    assert first_id not in turn["listing_ids"] and agent.find_listing(first_id).listing_id not in turn["listing_ids"]

    assert svc.memory_view("web:B")["empty"]  # 다른 방문자에겐 없음
    svc.forget("web:A")
    assert svc.memory_view("web:A")["empty"] and svc.store.latest_for("web:A") is None


def test_memory_is_given_to_the_model(tmp_path):
    agent = make_agent(tmp_path)
    svc = AgentService(agent)
    svc.memory.update_after("web:A", AgentContext(), [{"조건": "음성군 · 창고 용도", "args": {"query": "음성군 창고"}}])
    fake = ScriptedLLM([ToolTurn(text="네")])
    agent.llm = router(fake)
    seen_system = []
    original = fake.tool_step
    fake.tool_step = lambda system, *a, **k: (seen_system.append(system), original(system, *a, **k))[1]
    svc.ask("web:A", None, "안녕하세요")
    assert "이 손님에 대한 기억" in seen_system[0] and "음성군 · 창고 용도" in seen_system[0]


# ---------------------------------------------------------------- 웹: 스트리밍·쿠키·이어서 답하기
@pytest.fixture
def web(tmp_path, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    settings = Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs", llm_providers=["rule"])
    app = create_app(settings)
    app.config["TESTING"] = True
    return app, app.test_client(), settings


def token(c, path="/agent"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_web_stream_events_and_saved_turns(web):
    app, c, _ = web
    r = c.post("/agent/stream", data={"q": "양평 땅 2개 비교해줘", "csrf": token(c)})
    assert r.mimetype == "text/event-stream"
    events = [json.loads(line[6:]) for line in r.get_data(as_text=True).splitlines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[-2:] == ["done", "saved"] and "tool_start" in kinds
    assert [e["step"]["tool"] for e in events if e["type"] == "step" and e["step"]["kind"] == "tool"] == \
        ["search_listings", "compare_listings"]
    html = c.get("/agent").get_data(as_text=True)
    assert html.count('<article class="turn"') == 1 and "기억하고 있는 내용" in html


def test_visitor_cookie_is_long_lived_but_admin_login_is_not(web):
    app, c, _ = web
    r = c.get("/agent")
    cookies = r.headers.getlist("Set-Cookie")
    vid = next(x for x in cookies if x.startswith("rea_vid="))
    assert "Max-Age=7776000" in vid and "HttpOnly" in vid
    r = c.post("/admin/login", data={"password": "office-pass-1", "csrf": token(c, "/admin")})
    session_cookie = next(x for x in r.headers.getlist("Set-Cookie") if x.startswith("session="))
    assert "Max-Age" not in session_cookie and "Expires" not in session_cookie  # 브라우저를 닫으면 관리자 로그아웃


def test_web_conversation_survives_server_restart(web, tmp_path):
    from real_estate_agent.web.app import create_app

    app, c, settings = web
    c.post("/agent/ask", data={"q": "양평 땅", "csrf": token(c)})
    app2 = create_app(settings)  # 서버 재시작 (같은 비밀키·DB)
    c2 = app2.test_client()
    for cookie in c._cookies.values():  # 같은 브라우저 쿠키
        c2.set_cookie(cookie.key, cookie.value)
    html = c2.get("/agent").get_data(as_text=True)
    assert html.count('<article class="turn"') == 1 and "양평 땅" in html


def test_web_resume_banner_and_forget(web):
    app, c, _ = web
    svc = app.config["AGENT_SERVICE"]
    c.get("/agent")
    with c.session_transaction() as s:
        cid = s["cid"]
    owner = "web:" + c.get_cookie("rea_vid").value
    svc.store.save(cid, owner, [{"role": "user", "content": "양평 땅"}], AgentContext(), [], "interrupted", "양평 땅")
    html = c.get("/agent").get_data(as_text=True)
    assert "중간에 끊겼습니다" in html and "이어서 답하기" in html
    c.post("/agent/resume", data={"csrf": token(c)})
    html = c.get("/agent").get_data(as_text=True)
    assert "중간에 끊겼습니다" not in html and html.count('<article class="turn"') == 1
    c.post("/agent/forget", data={"csrf": token(c)})
    html = c.get("/agent").get_data(as_text=True)
    assert "기억을 모두 지웠습니다" in html and '<article class="turn"' not in html


# ---------------------------------------------------------------- FastAPI JSON API
@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient

    from real_estate_agent.api import create_api

    return TestClient(create_api(agent=make_agent(tmp_path, "api.db")))


def test_api_health_and_search(api):
    h = api.get("/api/v1/health").json()
    assert h["status"] == "ok" and h["listings"] > 0
    r = api.post("/api/v1/listings/search", json={"query": "양평 땅", "sort": "price_asc", "limit": 3})
    body = r.json()
    prices = [x["price_manwon"] for x in body["listings"]]
    assert r.status_code == 200 and prices == sorted(prices) and len(prices) == 3
    assert all(x["region"].endswith("***") and "010-" not in json.dumps(x, ensure_ascii=False)
               for x in body["listings"])
    assert api.post("/api/v1/listings/search", json={"limit": 50}).status_code == 422  # 구조화된 입력 검사
    assert api.get(f"/api/v1/listings/{body['listings'][0]['id']}").status_code == 200
    assert api.get("/api/v1/listings/없는번호").status_code == 404


def test_api_ask_follow_up_and_conversation(api):
    r = api.post("/api/v1/agent/ask", json={"question": "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘"})
    first = r.json()
    vid, cid = first["visitor_id"], first["conversation_id"]
    assert len(first["listings"]) == 2 and first["steps"]
    r = api.post("/api/v1/agent/ask", json={"question": "그 중에 넓은 건?", "conversation_id": cid},
                 headers={"X-Visitor-Id": vid})
    second = r.json()
    assert len(second["listings"]) == 1
    assert second["listings"][0]["id"] == max(first["listings"], key=lambda x: x["area_pyeong"])["id"]
    conv = api.get(f"/api/v1/agent/conversations/{cid}", headers={"X-Visitor-Id": vid}).json()
    assert len(conv["turns"]) == 2 and conv["status"] == "idle"
    assert api.get(f"/api/v1/agent/conversations/{cid}", headers={"X-Visitor-Id": "someone-else"}).status_code == 404
    assert api.delete("/api/v1/agent/memory", headers={"X-Visitor-Id": vid}).status_code == 204


def test_api_stream(api):
    with api.stream("POST", "/api/v1/agent/stream", json={"question": "양평 땅 2개 비교해줘"},
                    headers={"X-Visitor-Id": "visitor-123456"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        cid = r.headers["x-conversation-id"]
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events][-2:] == ["done", "saved"] and events[-1]["cid"] == cid
