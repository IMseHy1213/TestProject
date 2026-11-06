"""MCP 클라이언트: MCP 서버가 공개한 도구를 우리 Agent 의 도구로 씀 (v0.6 MCP 통합).

    [MCP host = 우리 AI 상담 Agent]
        └─ McpToolSource (MCP client) ── stdio / HTTP / 같은 프로세스 ──▶ [MCP server]
              tools/list  → 도구 이름·설명·인자 형식(JSON schema)을 받아 Agent 도구 목록에 등록
              tools/call  → Agent(GPT/Claude/규칙)가 도구를 고르면 서버에 실행을 요청하고 결과를 관찰로 돌려줌

Agent 루프는 동기 코드라, MCP 세션(비동기)은 별도 스레드의 이벤트 루프에서 계속 열어 두고 호출만 넘깁니다.
mcp 2.x (Client) 와 1.x (ClientSession) 를 모두 지원합니다.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional

from .tools import Tool, ToolError

CALL_TIMEOUT = 60  # 도구 호출 하나 최대 대기 (초)


def _get(obj: Any, *names: str) -> Any:
    """mcp 2.x 는 snake_case(input_schema), 1.x 는 camelCase(inputSchema)."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


class McpToolSource:
    def __init__(self, target: Any, timeout: float = CALL_TIMEOUT):
        """target: MCP 서버 객체(같은 프로세스) / StdioServerParameters(하위 프로세스) / URL(Streamable HTTP)"""
        self.target = target
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._client = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self.server_tools: list[Any] = []
        self.calls = 0

    # ------------------------------------------------------------------ 연결
    def start(self) -> "McpToolSource":
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._session(), self._loop)
        if not self._ready.wait(self.timeout):
            raise TimeoutError("MCP 서버 연결 시간 초과")
        if self._error:
            raise RuntimeError(f"MCP 서버에 연결하지 못했습니다: {self._error}") from self._error
        return self

    async def _session(self) -> None:
        self._stop = asyncio.Event()
        try:
            try:
                from mcp import Client  # mcp 2.x
            except ImportError:
                Client = None
            if Client is not None:
                async with Client(self.target) as client:
                    await self._ready_with(client)
            else:  # mcp 1.x: stdio 만
                from mcp import ClientSession
                from mcp.client.stdio import stdio_client

                async with stdio_client(self.target) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await self._ready_with(session)
        except BaseException as e:  # noqa: BLE001 - 연결 실패를 start() 에 알림
            self._error = e
            self._ready.set()

    async def _ready_with(self, client) -> None:
        self._client = client
        self.server_tools = list(_get(await client.list_tools(), "tools") or [])
        self._ready.set()
        await self._stop.wait()

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "McpToolSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ 도구
    def tools(self) -> list[Tool]:
        """MCP 서버 도구 -> Agent 도구 (이름·설명·인자 형식은 서버가 준 그대로)."""
        out = []
        for t in self.server_tools:
            name = _get(t, "name")
            schema = _get(t, "input_schema", "inputSchema") or {"type": "object", "properties": {}}
            out.append(Tool(name, (_get(t, "description") or "") + " (MCP 서버 도구)", _plain_schema(schema),
                            (lambda n: lambda **kw: self.call(n, kw))(name)))
        return out

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """tools/call -> 결과 (구조화된 결과가 있으면 그것, 없으면 글자를 JSON 으로 읽음). 서버 오류는 ToolError."""
        self.calls += 1
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, args), self._loop)
        result = fut.result(self.timeout)
        text = "\n".join(_get(c, "text") or "" for c in (_get(result, "content") or []))
        if _get(result, "is_error", "isError"):
            raise ToolError(text.replace(f"Error executing tool {name}: ", "") or f"MCP 도구 '{name}' 오류")
        structured = _get(result, "structured_content", "structuredContent")
        if isinstance(structured, dict):
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return {"text": text}


def _plain_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """MCP(pydantic) 인자 형식의 'X 또는 null' 표현(anyOf)을 단순한 type 으로 정리 -> 우리 인자 검사·LLM 도구 정의에 그대로 사용."""
    props = {}
    for key, spec in (schema.get("properties") or {}).items():
        spec = dict(spec)
        any_of = spec.pop("anyOf", None)
        if any_of:
            non_null = [s for s in any_of if s.get("type") != "null"]
            if len(non_null) == 1:
                spec.update(non_null[0])
        spec.pop("title", None)
        props[key] = spec
    out = {"type": "object", "properties": props}
    if schema.get("required"):
        out["required"] = list(schema["required"])
    return out


def stdio_server_params(settings) -> Any:
    """우리 MCP 서버를 하위 프로세스로 띄우는 설정 (python main.py --data-dir … mcp)."""
    import sys

    from mcp import StdioServerParameters

    from ..config import PROJECT_ROOT

    return StdioServerParameters(command=sys.executable,
                                 args=[str(PROJECT_ROOT / "main.py"), "--data-dir", str(settings.data_dir),
                                       "--db", str(settings.db_path), "mcp"],
                                 cwd=str(PROJECT_ROOT))
