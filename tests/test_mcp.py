"""v0.6 MCP: 서버(도구·리소스·프롬프트·오류)와, MCP 서버 도구로 움직이는 우리 Agent."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.mcp_server import PUBLIC_TOOLS, create_mcp_server
from real_estate_agent.tool_agent import AgentContext, ToolAgent
from real_estate_agent.tool_agent.mcp_client import McpToolSource, _plain_schema

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def agent(tmp_path):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / "r.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


@pytest.fixture
def server(agent):
    return create_mcp_server(agent)


def _client():
    try:
        from mcp import Client
    except ImportError:
        pytest.skip("mcp 2.x 의 같은 프로세스 Client 가 필요")
    return Client


def run(coro):
    return asyncio.run(coro)


def test_tools_list_is_read_only(server):
    Client = _client()

    async def go():
        async with Client(server) as c:
            return [t.name for t in (await c.list_tools()).tools]

    names = run(go())
    assert set(names) == set(PUBLIC_TOOLS)
    assert "record_feedback" not in names and "recall_memory" not in names


def test_search_and_privacy(server):
    Client = _client()

    async def go():
        async with Client(server) as c:
            r = await c.call_tool("search_listings", {"query": "음성군 창고 지을 땅", "limit": 3})
            return r

    r = run(go())
    assert not r.is_error
    data = r.structured_content
    text = json.dumps(data, ensure_ascii=False)
    assert data["매물"], data
    for item in data["매물"]:
        assert item["주소"].endswith("***")  # 시/도 + 시/군/구까지만
    assert "010-" not in text  # 연락처 없음


def test_tool_error_message_reaches_client(server):
    Client = _client()

    async def go():
        async with Client(server) as c:
            return await c.call_tool("get_listing", {"listing_id": "없는-매물"})

    r = run(go())
    assert r.is_error
    assert "없는-매물" in r.content[0].text or "찾을 수 없" in r.content[0].text


def test_resources_and_prompt(server):
    Client = _client()

    async def go():
        async with Client(server) as c:
            summary = await c.read_resource("listings://summary")
            prompt = await c.get_prompt("land_consultation", {"question": "공장 부지 있어요?"})
            return summary, prompt

    summary, prompt = run(go())
    data = json.loads(summary.contents[0].text)
    assert data["전체_매물_수"] > 0 and data["지역별"]
    assert "공장 부지 있어요?" in prompt.messages[0].content.text


def test_plain_schema_flattens_optional():
    s = _plain_schema({"type": "object", "required": ["a"],
                       "properties": {"a": {"type": "string", "title": "A"},
                                      "b": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}}})
    assert s == {"type": "object", "required": ["a"],
                 "properties": {"a": {"type": "string"}, "b": {"type": "integer", "default": None}}}


def test_agent_runs_on_mcp_tools(agent, server):
    """MCP 클라이언트로 서버 도구를 받아 우리 Agent 가 사용 + 이어서 묻기도 유지."""
    _client()
    with McpToolSource(server) as src:
        assert {t.name for t in src.tools()} == set(PUBLIC_TOOLS)
        t = ToolAgent(agent, ctx=AgentContext(), tool_source=src)
        first, h = t.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
        assert len(first.listing_ids) == 2 and src.calls >= 2
        assert all(s.tool in PUBLIC_TOOLS for s in first.steps if s.tool)
        second, _ = t.run("그 중에 넓은 건?", h)
        assert len(second.listing_ids) == 1 and second.listing_ids[0] in first.listing_ids
