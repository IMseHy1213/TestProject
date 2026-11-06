"""v0.6 MCP 서버: 사무실 토지 매물 도구를 Model Context Protocol 로 공개.

실행 (stdio):  python main.py mcp
  - Claude Desktop 등 MCP 를 지원하는 프로그램이 이 명령을 실행해 표준 입출력으로 대화합니다.
  - HTTP 로 띄우려면: python main.py mcp --http --port 8002   (http://127.0.0.1:8002/mcp)

제공하는 것
  tools     : search_listings, get_listing, check_land_use, compare_listings, list_regions  (모두 읽기 전용)
  resources : listings://summary (지역별 매물 수),  listings://{listing_id} (매물 하나)
  prompts   : land_consultation (토지 상담 시작 문장)

개인정보: 손님 화면과 같은 기준 (주소는 시/도 + 시/군/구까지, 연락처·매물번호 속 이름 없음).
쓰기 도구(피드백 기록 등)는 공개하지 않습니다.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Any, Optional

log = logging.getLogger(__name__)

SERVER_NAME = "real-estate-land-agent"
INSTRUCTIONS = ("충북 음성의 토지 전문 중개사무소가 보유한 토지 매물을 찾고, 비교하고, 건축 가능 여부를 1차 판단하는 도구입니다. "
                "가격 단위는 만원(1억=10000)입니다. 주소는 시/도+시/군/구까지만 제공됩니다.")
PUBLIC_TOOLS = ("search_listings", "get_listing", "check_land_use", "compare_listings", "list_regions")


def _server_class():
    """mcp 2.x 는 MCPServer, 1.x 는 FastMCP (설치된 버전에 맞춰 사용)."""
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def _tool_error_class():
    """도구가 일부러 낸 오류 (메시지가 모델에게 그대로 전달됨). 일반 예외는 SDK 가 메시지를 숨김."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError
    return ToolError


def create_mcp_server(agent=None, settings=None, lock: Optional[threading.RLock] = None):
    """MCP 서버 객체를 만듦 (테스트에서는 같은 프로세스 안에서 바로 연결)."""
    from .agent import RealEstateAgent
    from .config import get_settings
    from .tool_agent.tools import AgentContext, ToolRegistry, brief, display_name, shown_region

    if agent is None:
        agent = RealEstateAgent(settings or get_settings())
        agent.sync()
    lock = lock or threading.RLock()
    registry = ToolRegistry(agent, AgentContext(), lock)
    server = _server_class()(SERVER_NAME, instructions=INSTRUCTIONS)
    schemas = {t.name: t for t in registry.tools.values()}
    tool_error = _tool_error_class()

    def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = {k: v for k, v in args.items() if v is not None}
        obs = registry.run(name, args)
        if not obs["ok"]:
            raise tool_error(obs["error"])  # MCP 오류 결과 (isError) - 모델이 읽고 인자를 고칠 수 있게 메시지 포함
        return obs["result"]

    # ------------------------------------------------------------------ tools
    @server.tool(name="search_listings", description=schemas["search_listings"].description)
    def search_listings(query: str = "", regions: Optional[list[str]] = None,
                        exclude_regions: Optional[list[str]] = None, purpose: Optional[str] = None,
                        budget_min_manwon: Optional[int] = None, budget_max_manwon: Optional[int] = None,
                        area_min_pyeong: Optional[float] = None, area_max_pyeong: Optional[float] = None,
                        keywords: Optional[list[str]] = None, sort: str = "score", limit: int = 5,
                        use_previous_conditions: bool = False, other_regions: bool = False,
                        recheck: bool = False) -> dict[str, Any]:
        # 이어서 묻기('그 중에', '다른 지역', '정말 없어?')는 이 서버 연결(손님 한 명) 안에서 직전 검색을 기억해 처리
        return run("search_listings", dict(query=query, regions=regions, exclude_regions=exclude_regions,
                                           purpose=purpose, budget_min_manwon=budget_min_manwon,
                                           budget_max_manwon=budget_max_manwon, area_min_pyeong=area_min_pyeong,
                                           area_max_pyeong=area_max_pyeong, keywords=keywords, sort=sort, limit=limit,
                                           use_previous_conditions=use_previous_conditions or None,
                                           other_regions=other_regions or None, recheck=recheck or None))

    @server.tool(name="get_listing", description=schemas["get_listing"].description)
    def get_listing(listing_id: str) -> dict[str, Any]:
        return run("get_listing", {"listing_id": listing_id})

    @server.tool(name="check_land_use", description=schemas["check_land_use"].description)
    def check_land_use(listing_id: str, purpose: str) -> dict[str, Any]:
        return run("check_land_use", {"listing_id": listing_id, "purpose": purpose})

    @server.tool(name="compare_listings", description=schemas["compare_listings"].description)
    def compare_listings(listing_ids: list[str]) -> dict[str, Any]:
        return run("compare_listings", {"listing_ids": listing_ids})

    @server.tool(name="list_regions", description=schemas["list_regions"].description)
    def list_regions(limit: int = 40) -> dict[str, Any]:
        return run("list_regions", {"limit": limit})

    # ------------------------------------------------------------------ resources
    @server.resource("listings://summary", name="listings_summary", mime_type="application/json",
                     description="보유 토지 매물 수와 지역별 매물 수")
    def listings_summary() -> str:
        from collections import Counter

        with lock:
            listings = agent.listings()
            by_region = Counter(shown_region(l) for l in listings)
        return json.dumps({"전체_매물_수": len(listings), "지역별": dict(by_region.most_common())}, ensure_ascii=False)

    @server.resource("listings://{listing_id}", name="listing", mime_type="application/json",
                     description="매물 하나 (공개 정보)")
    def listing_resource(listing_id: str) -> str:
        with lock:
            real = registry.resolve(listing_id)
            l = agent.find_listing(real) if real else None
            if l is None:
                raise ValueError(f"매물 '{listing_id}' 을(를) 찾을 수 없습니다")
            return json.dumps({**brief(l), "이름": display_name(l)}, ensure_ascii=False)

    # ------------------------------------------------------------------ prompts
    @server.prompt(name="land_consultation", description="토지 매물 상담을 시작하는 프롬프트")
    def land_consultation(question: str) -> str:
        return ("당신은 토지 전문 중개사무소의 상담 보조입니다. 아래 손님 질문에 답하려면 search_listings 로 매물을 찾고, "
                "필요하면 compare_listings·check_land_use 로 확인한 뒤, 매물번호와 가격·면적을 들어 2~4문장으로 답하세요. "
                "도구 결과에 없는 매물은 말하지 마세요.\n\n손님 질문: " + question)

    server._real_estate_agent = agent  # 테스트용
    return server


def main(settings=None, http: bool = False, host: str = "127.0.0.1", port: int = 8002) -> int:
    """stdio 로 실행할 때는 표준 출력이 MCP 통신 전용이므로 로그는 표준 에러로만."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = create_mcp_server(settings=settings)
    if http:
        print(f"MCP 서버 (Streamable HTTP): http://{host}:{port}/mcp", file=sys.stderr)
        server.run(transport="streamable-http", host=host, port=port)
    else:
        server.run(transport="stdio")
    return 0
