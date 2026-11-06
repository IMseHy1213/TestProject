"""v0.8 Distributed / A2A: Agent Card, JSON-RPC(0.3·1.0), 스트리밍, 멱등성, 인증, 상태 전달, 분산 상담팀과 장애 대처."""

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from real_estate_agent.a2a import protocol as p
from real_estate_agent.a2a.client import A2AClient
from real_estate_agent.a2a.distributed import DistributedMultiAgent, context_id_for
from real_estate_agent.a2a.server import create_a2a_app
from real_estate_agent.a2a.state import public_state
from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.tool_agent import AgentContext
from real_estate_agent.tool_agent.memory import LongTermMemory
from real_estate_agent.tool_agent.multi import ANALYST, SEARCHER, SUPERVISOR, MultiAgent
from real_estate_agent.tool_agent.service import AgentService

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
DEAD = "http://127.0.0.1:9/"  # 아무도 듣지 않는 주소


@pytest.fixture(autouse=True)
def _reset_breaker():
    A2AClient.reset_all()
    yield
    A2AClient.reset_all()


def make_agent(tmp_path, name):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / name, log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


@pytest.fixture
def servers(tmp_path):
    """검색·분석 Agent 서버 (각자 자기 DB = 다른 프로세스와 같은 상황)."""
    return {SEARCHER: TestClient(create_a2a_app("search", agent=make_agent(tmp_path, "s.db"), token="")),
            ANALYST: TestClient(create_a2a_app("analysis", agent=make_agent(tmp_path, "a.db"), token=""))}


def rpc(c, method, params, rid=1):
    return c.post("/", json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).json()


def user_msg(text, data=None, context_id=None):
    parts = [p.text_part(text)] + ([p.data_part(data)] if data else [])
    return p.message("user", parts, context_id)


# ---------------------------------------------------------------- Agent Card
def test_agent_card(tmp_path):
    c = TestClient(create_a2a_app("search", agent=make_agent(tmp_path, "c.db"), token="",
                                  base_url="http://127.0.0.1:8101/"))
    card = c.get("/.well-known/agent-card.json").json()
    assert card == c.get("/.well-known/agent.json").json()
    assert card["name"] == "토지 매물 검색 Agent" and card["url"] == "http://127.0.0.1:8101/"
    assert card["capabilities"]["streaming"] is True and card["skills"][0]["id"] == "search_listings"
    assert {i["protocolVersion"] for i in card["supportedInterfaces"]} == {"1.0", "0.3"}
    assert "security" not in card
    secured = TestClient(create_a2a_app("consult", agent=make_agent(tmp_path, "c2.db"), token="t0ken-123"))
    card = secured.get("/.well-known/agent-card.json").json()  # 카드는 누구나 볼 수 있음
    assert card["security"] == [{"bearer": []}] and card["securitySchemes"]["bearer"]["scheme"] == "bearer"


# ---------------------------------------------------------------- JSON-RPC 0.3
def test_send_returns_completed_task_with_public_data(servers):
    c = servers[SEARCHER]
    r = rpc(c, "message/send", {"message": user_msg("음성군에 창고 지을 수 있는 땅 중 싼 거 2개")})
    task = r["result"]
    assert task["kind"] == "task" and task["status"]["state"] == "completed"
    data = p.task_data(task)
    assert len(data["listing_ids"]) == 2 and data["agent"] == SEARCHER
    assert [s["tool"] for s in data["steps"] if s["kind"] == "tool"] == ["search_listings"]  # 검색 도구만
    assert "음성군" in p.task_text(task)
    text = json.dumps(task, ensure_ascii=False)
    assert "010-" not in text
    assert all(x["주소"].endswith("***") for x in data["observations"][0][1]["result"]["매물"])
    got = rpc(c, "tasks/get", {"id": task["id"]})["result"]
    assert got["id"] == task["id"] and got["status"]["state"] == "completed"


def test_context_follow_up_without_state(tmp_path):
    """외부 Agent: 상태를 보내지 않아도 같은 contextId 면 이어서 묻기."""
    c = TestClient(create_a2a_app("consult", agent=make_agent(tmp_path, "c.db"), token=""))
    first = rpc(c, "message/send", {"message": user_msg("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘",
                                                        context_id="ctx-1")})["result"]
    shown = p.task_data(first)["listing_ids"]
    assert len(shown) == 2 and first["contextId"] == "ctx-1"
    second = rpc(c, "message/send", {"message": user_msg("그 중에 넓은 건?", context_id="ctx-1")})["result"]
    assert p.task_data(second)["listing_ids"][0] in shown and "앞서 보여드린 2건" in p.task_text(second)
    other = rpc(c, "message/send", {"message": user_msg("그 중에 넓은 건?", context_id="ctx-2")})["result"]
    assert "앞서 보여드린" not in p.task_text(other)  # 다른 대화와 섞이지 않음


def test_state_travels_between_servers(servers):
    """상태를 메시지에 실으면, 처음 보는 서버도 이어서 묻기를 처리 (서버는 상태를 들고 있지 않아도 됨)."""
    first = rpc(servers[SEARCHER], "message/send",
                {"message": user_msg("음성군에 창고 지을 수 있는 땅 중 싼 거 2개")})["result"]
    data = p.task_data(first)
    fresh = TestClient(servers[SEARCHER].app)  # 같은 앱이지만 contextId 를 주지 않음 = 새 대화
    second = rpc(fresh, "message/send",
                 {"message": user_msg("그 중에 넓은 건?", {"state": data["state"]})})["result"]
    assert p.task_data(second)["listing_ids"][0] in data["listing_ids"]


def test_same_message_id_is_not_run_twice(servers):
    c = servers[SEARCHER]
    msg = user_msg("양평 땅")
    a = rpc(c, "message/send", {"message": msg})["result"]
    b = rpc(c, "message/send", {"message": msg})["result"]
    assert a["id"] == b["id"] and len(c.app.state.executor.tasks) == 1


def test_json_rpc_errors(servers):
    c = servers[SEARCHER]
    assert c.post("/", content=b"{not json").json()["error"]["code"] == p.PARSE_ERROR
    assert c.post("/", json={"method": "message/send"}).json()["error"]["code"] == p.INVALID_REQUEST
    assert rpc(c, "message/fly", {})["error"]["code"] == p.METHOD_NOT_FOUND
    assert rpc(c, "message/send", {"message": {"role": "user", "parts": []}})["error"]["code"] == p.INVALID_PARAMS
    assert rpc(c, "message/send", {"message": user_msg("")})["error"]["code"] == p.INVALID_PARAMS
    assert rpc(c, "message/send", {"message": user_msg("가" * 3000)})["error"]["code"] == p.INVALID_PARAMS
    assert rpc(c, "tasks/get", {"id": "nope"})["error"]["code"] == p.TASK_NOT_FOUND
    done = rpc(c, "message/send", {"message": user_msg("양평 땅")})["result"]
    assert rpc(c, "tasks/cancel", {"id": done["id"]})["error"]["code"] == p.TASK_NOT_CANCELABLE
    assert rpc(c, "tasks/pushNotificationConfig/set", {})["error"]["code"] == p.PUSH_NOT_SUPPORTED
    assert rpc(c, "ListTasks", {})["error"]["code"] == p.UNSUPPORTED_OPERATION


def test_bad_listing_ids_fail_the_task(servers):
    r = rpc(servers[ANALYST], "message/send", {"message": user_msg("비교", {"listing_ids": ["없는-id"]})})["result"]
    assert r["status"]["state"] == "failed" and "없는-id" in p.task_text(r)
    r = rpc(servers[ANALYST], "message/send", {"message": user_msg("비교", {"state": "x"})})["result"]
    assert r["status"]["state"] == "failed"


def test_token_required(tmp_path):
    app = create_a2a_app("search", agent=make_agent(tmp_path, "t.db"), token="s3cret-token")
    c = TestClient(app)
    assert c.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "x"}}).status_code == 401
    with pytest.raises(p.A2AError) as e:
        A2AClient("http://testserver/", token="wrong", http=c).send("양평 땅")
    assert e.value.code == 401
    task = A2AClient("http://testserver/", token="s3cret-token", http=c).send("양평 땅")
    assert task["status"]["state"] == "completed"


def test_stream_events(servers):
    events = list(A2AClient("http://testserver/", http=servers[ANALYST]).stream(
        "두 매물 비교해줘", {"listing_ids": ["Y-2001", "Y-2002"]}))
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "task" and kinds[-2:] == ["artifact-update", "status-update"]
    assert events[-1]["final"] is True and events[-1]["status"]["state"] == "completed"
    progress = [p.get_text(e["status"]["message"]["parts"]) for e in events
                if e["kind"] == "status-update" and e["status"].get("message") and not e["final"]]
    assert any("compare_listings" in line for line in progress)


# ---------------------------------------------------------------- A2A 1.0 형식
def test_v1_methods(servers):
    c = servers[SEARCHER]
    msg = {"messageId": "v1-m1", "contextId": "v1-ctx", "role": "ROLE_USER", "parts": [{"text": "양평 땅 2개"}]}
    r = rpc(c, "SendMessage", {"message": msg})["result"]
    task = r["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED" and "kind" not in task
    assert task["artifacts"][0]["parts"][0]["text"] and "data" in task["artifacts"][0]["parts"][1]
    assert task["status"]["message"]["role"] == "ROLE_AGENT"
    assert rpc(c, "GetTask", {"id": task["id"]})["result"]["id"] == task["id"]
    assert rpc(c, "CancelTask", {"id": task["id"]})["error"]["code"] == p.TASK_NOT_CANCELABLE
    body = c.post("/", json={"jsonrpc": "2.0", "id": 2, "method": "SendStreamingMessage",
                             "params": {"message": {**msg, "messageId": "v1-m2", "parts": [{"text": "그 중에 싼 건?"}]}}})
    events = [json.loads(line[6:])["result"] for line in body.text.splitlines() if line.startswith("data: ")]
    assert set(events[0]) == {"task"} and set(events[-1]) == {"statusUpdate"}
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert any("artifactUpdate" in e for e in events)


# ---------------------------------------------------------------- 상태 전달 (개인정보)
def test_state_uses_public_ids():
    listing = SimpleNamespace(listing_id="S.24250409(홍길동님)")
    fake = SimpleNamespace(listings=lambda: [listing], hidden_listings=[])
    ctx = AgentContext(last_shown=[listing.listing_id], seen=[listing.listing_id],
                       rejected={listing.listing_id: "비쌈"}, run_searches=[{"조건": "x"}])
    state = public_state(fake, ctx)
    assert "홍길동" not in json.dumps(state, ensure_ascii=False)
    assert state["last_shown"] == ["S.24250409"] and "run_searches" not in state


# ---------------------------------------------------------------- 분산 상담팀
QUESTIONS = ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "그 중에 넓은 건?",
             "공장 지을 수 있는 땅 중 넓은 거 3개 보여줘", "음성군 말고 다른 지역에도 있어?", "정말 다른 지역에는 없어?"]


def remote_clients(servers):
    return {who: A2AClient("http://testserver/", http=c) for who, c in servers.items()}


def test_distributed_matches_in_process_team(tmp_path, servers):
    web = make_agent(tmp_path, "w.db")
    dist = DistributedMultiAgent(web, clients=remote_clients(servers), ctx=AgentContext())
    local = MultiAgent(web, ctx=AgentContext())
    hd = hl = []
    for q in QUESTIONS:
        a, hd = dist.run(q, hd)
        b, hl = local.run(q, hl)
        assert (a.answer, a.listing_ids) == (b.answer, b.listing_ids), q
        remote = [s for s in a.steps if s.kind == "remote"]
        assert remote and all("A2A http://testserver/" in s.summary for s in remote)
        assert all(h["ok"] for h in dist.hops)
    first = dist.run(QUESTIONS[0], [])[0]
    assert {s.agent for s in first.steps} == {SUPERVISOR, SEARCHER, ANALYST}
    assert any("최저가" in b for b in first.badges.values())


def test_remote_down_falls_back_and_breaker_opens(tmp_path, servers):
    web = make_agent(tmp_path, "w.db")
    clients = {SEARCHER: A2AClient("http://testserver/", http=servers[SEARCHER]),
               ANALYST: A2AClient(DEAD, timeout=2, retries=0)}
    dist = DistributedMultiAgent(web, clients=clients, ctx=AgentContext())
    r, _ = dist.run("여주 공장 부지 2개 비교해줘")
    expected, _ = MultiAgent(web, ctx=AgentContext()).run("여주 공장 부지 2개 비교해줘")
    assert r.answer == expected.answer and r.listing_ids == expected.listing_ids
    notes = [s.summary for s in r.steps if s.kind == "fallback" and s.agent == SUPERVISOR]
    assert any("대신 처리" in n for n in notes)
    assert not clients[ANALYST].available()  # 서킷 브레이커: 잠시 시도하지 않음
    t0 = time.perf_counter()
    r2, _ = dist.run("여주 공장 부지 2개 비교해줘")
    assert time.perf_counter() - t0 < 2 and r2.answer == expected.answer
    assert any("서킷 브레이커" in s.summary for s in r2.steps if s.kind == "fallback")


def test_memory_stays_on_web_server(tmp_path, servers):
    web = make_agent(tmp_path, "w.db")
    memory = LongTermMemory(web.storage.conn)
    dist = DistributedMultiAgent(web, clients=remote_clients(servers), ctx=AgentContext(), memory=memory,
                                 owner="visitor-1")
    dist.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개")
    mem = memory.load("visitor-1")
    assert mem["searches"] and "음성군" in mem["searches"][-1]["조건"]
    sent = [json.dumps(t["history"][0], ensure_ascii=False)
            for t in servers[SEARCHER].app.state.executor.tasks.values()]
    assert sent and all("visitor-1" not in s for s in sent)


def test_service_distributed_mode_without_servers(tmp_path, monkeypatch):
    import real_estate_agent.a2a.distributed as d

    monkeypatch.setattr(d, "SEARCH_URL", DEAD)
    monkeypatch.setattr(d, "ANALYSIS_URL", DEAD)
    monkeypatch.setattr(d, "REMOTE_TIMEOUT", 2.0)
    service = AgentService(make_agent(tmp_path, "svc.db"))
    cid, turn = service.ask("v1", None, "여주 공장 부지 2개 비교해줘", mode="distributed")
    assert turn["mode"] == "distributed" and len(turn["listing_ids"]) == 2
    _, turn2 = service.ask("v1", cid, "그 중에 넓은 건?", mode="distributed")
    assert len(turn2["listing_ids"]) == 1 and turn2["listing_ids"][0] in turn["listing_ids"]
    assert context_id_for(cid) != cid and len(context_id_for(cid)) == 24


# ---------------------------------------------------------------- 실제 네트워크 (uvicorn)
def test_real_http_round_trip(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_a2a_app("search", agent=make_agent(tmp_path, "h.db"), token="",
                         base_url=f"http://127.0.0.1:{port}/")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        client = A2AClient(f"http://127.0.0.1:{port}", timeout=10)
        for _ in range(100):
            try:
                card = client.card()
                break
            except p.A2AUnavailable:
                time.sleep(0.05)
        assert card["url"] == f"http://127.0.0.1:{port}/"
        task = client.send("음성군 창고 땅 2개", context_id="net-1")
        assert task["status"]["state"] == "completed" and len(p.task_data(task)["listing_ids"]) == 2
        assert client.last_ms > 0
        events = list(client.stream("그 중에 싼 건?", context_id="net-1"))
        assert events[-1]["final"] and "앞서 보여드린 2건" in p.task_text(client.get_task(events[0]["id"]))
    finally:
        server.should_exit = True
        thread.join(5)
