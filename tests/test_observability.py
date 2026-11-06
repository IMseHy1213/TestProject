"""v1.1 Observable: 추적(trace·span·traceparent), 지표(Prometheus), 토큰·비용, 구조화 로그, 저장·관측 화면·OTLP."""

import json
import logging
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from real_estate_agent import observability
from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import ToolTurn
from real_estate_agent.llm.base import record_usage
from real_estate_agent.observability import TRACER
from real_estate_agent.observability.dashboard import recent, summary, text_waterfall, trace_tree
from real_estate_agent.observability.metrics import (AGENT_REQUESTS, LLM_CALLS, LLM_COST, LLM_TOKENS, METRICS,
                                                     RELIABILITY, TOOL_CALLS, Counter, Histogram)
from real_estate_agent.observability.tracing import parse_traceparent, redact, traceparent
from real_estate_agent.tool_agent import AgentContext, ToolAgent

from test_tool_agent import ScriptedLLM, call, router

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture(autouse=True)
def _clean():
    TRACER.shutdown()
    TRACER.ring.clear()
    METRICS.reset()
    observability._PRICES = None
    yield
    TRACER.shutdown()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=SAMPLE, db_path=tmp_path / "o.db", log_dir=tmp_path / "logs", llm_providers=["rule"])


@pytest.fixture
def agent(settings):
    a = RealEstateAgent(settings)
    a.sync()
    return a


def spans_of(trace_id):
    return {s.name: s for s in TRACER.trace(trace_id)}


# ---------------------------------------------------------------- 추적 기본
def test_spans_nest_and_record_errors():
    with TRACER.span("outer", a=1) as outer:
        tp = traceparent()
        with TRACER.span("inner") as inner:
            pass
        with pytest.raises(ValueError):
            with TRACER.span("boom"):
                raise ValueError("고장")
    assert inner.trace_id == outer.trace_id and inner.parent_id == outer.span_id and not outer.parent_id
    boom = spans_of(outer.trace_id)["boom"]
    assert boom.status == "error" and "고장" in boom.status_msg
    assert tp == f"00-{outer.trace_id}-{outer.span_id}-01" and parse_traceparent(tp) == (outer.trace_id, outer.span_id)
    assert parse_traceparent("00-" + "0" * 32 + "-" + "1" * 16 + "-01") is None and parse_traceparent("x") is None
    with TRACER.span("remote", traceparent=tp) as remote:  # 다른 서버에서 이어 받기
        pass
    assert remote.trace_id == outer.trace_id and remote.parent_id == outer.span_id


def test_redact():
    assert redact("집주인 010-1234-5678 번호, 용몽리 10 땅") == "집주인 *** 번호, 용몽리 *** 땅"
    assert len(redact("가" * 300)) == 101


# ---------------------------------------------------------------- Agent 추적·지표
def test_agent_run_is_traced(agent):
    r, _ = ToolAgent(agent).run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    assert re.fullmatch(r"[0-9a-f]{32}", r.trace_id)
    spans = TRACER.trace(r.trace_id)
    root = next(s for s in spans if s.name == "agent.run")
    assert not root.parent_id and root.attrs["policy"] == "rule" and root.attrs["listings"] == 2
    tools = [s for s in spans if s.name.startswith("tool.")]
    assert [s.name for s in tools] == ["tool.search_listings", "tool.compare_listings", "tool.check_land_use"]
    assert all(s.parent_id == root.span_id for s in tools) and tools[0].attrs["results"] == 2
    assert AGENT_REQUESTS.get(mode="single", policy="rule", stopped="answer") == 1
    assert TOOL_CALLS.get(tool="search_listings", ok="true") == 1


def test_llm_calls_tokens_and_cost(agent, monkeypatch):
    monkeypatch.setenv("REA_LLM_PRICES", "openai:2/8")

    def answer(messages):
        record_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=1200, output_tokens=300)))
        return ToolTurn(text="양평 땅 2건을 찾았습니다.")

    def search(messages):
        record_usage(SimpleNamespace(usage={"input_tokens": 1000, "output_tokens": 50}))
        return call("search_listings", query="양평 땅", limit=2)

    r, _ = ToolAgent(agent, llm=router(ScriptedLLM([search, answer]))).run("양평 땅 2개")
    llm = [s for s in TRACER.trace(r.trace_id) if s.name == "llm.tool_step"]
    root = next(s for s in TRACER.trace(r.trace_id) if s.name == "agent.run")
    assert len(llm) == 2 and all(s.parent_id == root.span_id for s in llm)  # AI 호출 스레드에서도 같은 trace
    assert (llm[0].attrs["input_tokens"], llm[1].attrs["output_tokens"]) == (1000, 300)
    assert LLM_TOKENS.get(provider="openai", type="input") == 2200 and LLM_CALLS.get(provider="openai", status="ok") == 2
    assert LLM_COST.get(provider="openai") == pytest.approx((2200 * 2 + 350 * 8) / 1e6)


def test_llm_failure_span(agent):
    from real_estate_agent.llm import LLMUnavailable

    r, _ = ToolAgent(agent, llm=router(ScriptedLLM([LLMUnavailable("서버 오류", kind="server")]))).run("양평 땅")
    failed = [s for s in TRACER.trace(r.trace_id) if s.name == "llm.tool_step"][0]
    assert failed.status == "error" and failed.attrs["status_kind"] == "server"
    assert LLM_CALLS.get(provider="openai", status="server") == 1
    root = next(s for s in TRACER.trace(r.trace_id) if s.name == "agent.run")
    assert any(e["name"] == "reliability.llm_fallbacks" for e in root.events)  # v1.0 안전장치가 trace 에 남음
    assert RELIABILITY.get(kind="llm_fallbacks") == 1


def test_multi_agent_trace(agent):
    from real_estate_agent.tool_agent.multi import MultiAgent

    r, _ = MultiAgent(agent, ctx=AgentContext()).run("여주 공장 부지 2개 비교해줘")
    names = [(s.name, s.attrs.get("agent")) for s in TRACER.trace(r.trace_id) if s.name == "agent.run"]
    assert {a for _, a in names} == {"총괄 Agent", "검색 Agent", "분석 Agent"}
    assert AGENT_REQUESTS.total(mode="multi") == 1 and AGENT_REQUESTS.total(mode="single") == 0  # 질문은 한 번만 셈


def test_trace_crosses_a2a(tmp_path, agent):
    from fastapi.testclient import TestClient

    from real_estate_agent.a2a.client import A2AClient
    from real_estate_agent.a2a.distributed import DistributedMultiAgent
    from real_estate_agent.a2a.server import create_a2a_app
    from real_estate_agent.tool_agent.multi import ANALYST, SEARCHER

    A2AClient.reset_all()
    clients = {w: A2AClient("http://testserver/", http=TestClient(create_a2a_app(role, agent=agent, token="")))
               for w, role in ((SEARCHER, "search"), (ANALYST, "analysis"))}
    r, _ = DistributedMultiAgent(agent, clients=clients).run("여주 공장 부지 2개 비교해줘")
    spans = TRACER.trace(r.trace_id)
    by_id = {s.span_id: s for s in spans}
    serves = [s for s in spans if s.name == "a2a.serve"]
    assert len(serves) == 2 and all(by_id[s.parent_id].name == "a2a.client" for s in serves)
    assert {s.attrs["role"] for s in serves} == {"search", "analysis"}


# ---------------------------------------------------------------- 지표 형식
def test_prometheus_format():
    c, h = Counter("t_total", "테스트"), Histogram("t_seconds", "시간", buckets=(0.1, 1.0))
    c.inc(2, a='x"y')
    h.observe(0.05, k="v")
    h.observe(0.5, k="v")
    text = "\n".join([*c.lines(), *h.lines()])
    assert 't_total{a="x\\"y"} 2' in text
    assert 't_seconds_bucket{k="v",le="0.1"} 1' in text and 't_seconds_bucket{k="v",le="1"} 2' in text
    assert 't_seconds_bucket{k="v",le="+Inf"} 2' in text and 't_seconds_count{k="v"} 2' in text
    assert h.quantile(0.5) == 0.1 and h.quantile(0.99) == 1.0


# ---------------------------------------------------------------- 저장·관측 화면·CLI
def test_sqlite_store_and_dashboard(settings, agent):
    observability.setup(settings, "test", json_logs=False)
    t = ToolAgent(agent, ctx=AgentContext())
    first, h = t.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    t.run("그 중에 넓은 건?", h)
    TRACER.flush()
    s = summary(agent.storage.conn, None, 24)
    assert s["questions"] == 2 and s["success_rate"] == 100.0 and s["p95_ms"] is not None
    assert {r["tool"] for r in s["tools"]} >= {"search_listings", "compare_listings"}
    rows = recent(agent.storage.conn, None, 10)
    assert [r["question"] for r in rows] == ["그 중에 넓은 건?", "음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘"]
    tree = trace_tree(agent.storage.conn, None, first.trace_id)
    assert tree[0]["name"] == "agent.run" and tree[0]["depth"] == 0 and tree[1]["depth"] == 1
    assert all(0 <= r["offset_pct"] <= 100 for r in tree)
    assert "tool.search_listings" in text_waterfall(tree)


def test_cli_trace_and_observe(settings, agent, capsys):
    from real_estate_agent.cli import run_observe, run_trace

    observability.setup(settings, "test", json_logs=False)
    ToolAgent(agent).run("양평 땅 2개 비교해줘")
    TRACER.flush()
    assert run_trace(agent, "last") == 0 and "tool.compare_listings" in capsys.readouterr().out
    assert run_observe(agent) == 0 and "질문 1건" in capsys.readouterr().out
    assert run_trace(agent, "f" * 32) == 1


class _Collector(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Collector.received.append((self.path, json.loads(body)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_otlp_export(settings, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("REA_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        observability.setup(settings, "web", sqlite=False, json_logs=False)
        with TRACER.span("agent.run", question="양평 땅") as root:
            with TRACER.span("tool.search_listings") as child:
                child.event("reliability.tool_retries")
        TRACER.flush()
        path, body = _Collector.received[-1]
        assert path == "/v1/traces"
        rs = body["resourceSpans"][0]
        assert rs["resource"]["attributes"][0]["value"]["stringValue"] == "real-estate-agent.web"
        spans = {s["name"]: s for s in rs["scopeSpans"][0]["spans"]}
        assert spans["tool.search_listings"]["parentSpanId"] == root.span_id
        assert spans["agent.run"]["traceId"] == root.trace_id and len(root.trace_id) == 32
        assert spans["tool.search_listings"]["events"][0]["name"] == "reliability.tool_retries"
    finally:
        server.shutdown()


# ---------------------------------------------------------------- 구조화 로그
def test_json_logs_carry_trace_id(settings):
    from real_estate_agent.observability.logs import setup_json_logs

    path = setup_json_logs(settings.log_dir, "test")
    log = logging.getLogger("real_estate_agent.test")
    with TRACER.span("agent.run") as sp:
        log.warning("손님 연락처 010-9999-8888 문의")
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    row = lines[-1]
    assert row["trace_id"] == sp.trace_id and row["level"] == "WARNING" and "010-9999" not in row["msg"]


# ---------------------------------------------------------------- 웹·API
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


def _csrf(c, path="/agent"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_web_observe_pages_and_metrics(web):
    c = web.test_client()
    c.post("/agent/ask", data={"q": "양평 땅 2개 비교해줘", "csrf": _csrf(c)})
    assert c.get("/admin/observe").status_code == 200 and "관리자 로그인" in c.get("/admin/observe").get_data(as_text=True)
    c.post("/admin/login", data={"password": "office-pass-1", "csrf": _csrf(c, "/admin")})
    html = c.get("/admin/observe").get_data(as_text=True)
    assert "양평 땅 2개 비교해줘" in html and "search_listings" in html
    trace_id = re.search(r"/admin/traces/([0-9a-f]{32})", html).group(1)
    page = c.get(f"/admin/traces/{trace_id}").get_data(as_text=True)
    assert "agent.run" in page and "tool.compare_listings" in page and "<rect" in page
    assert c.get("/admin/traces/nothex").status_code == 404 and c.get("/admin/traces/" + "a" * 32).status_code == 404
    assert f"/admin/traces/{trace_id}" in c.get("/agent").get_data(as_text=True)  # 관리자에게는 추적 링크
    text = c.get("/metrics").get_data(as_text=True)
    assert 'rea_agent_requests_total{mode="single",policy="rule",stopped="answer"} 1' in text
    assert "rea_http_requests_total" in text and "rea_listings " in text
    anon = web.test_client()
    assert anon.get("/metrics", environ_base={"REMOTE_ADDR": "10.1.2.3"}).status_code == 403
    assert anon.get("/metrics").status_code == 200  # 이 PC(127.0.0.1)


def test_api_metrics_token(agent, monkeypatch):
    from fastapi.testclient import TestClient

    from real_estate_agent.api import create_api

    api = TestClient(create_api(agent=agent))
    api.post("/api/v1/agent/ask", json={"question": "양평 땅"})
    assert "rea_agent_requests_total" in api.get("/metrics").text
    assert 'route="/api/v1/agent/ask"' in api.get("/metrics").text
    monkeypatch.setenv("REA_METRICS_TOKEN", "scrape-secret")
    assert api.get("/metrics").status_code == 403
    assert api.get("/metrics", headers={"Authorization": "Bearer scrape-secret"}).status_code == 200
