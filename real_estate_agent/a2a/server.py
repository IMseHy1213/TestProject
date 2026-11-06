"""v0.8 A2A 서버: 검색 Agent / 분석 Agent / 상담 Agent 를 각각 독립된 서버(프로세스)로 띄움.

    python main.py a2a-serve --role search    → http://127.0.0.1:8101/   (검색 Agent)
    python main.py a2a-serve --role analysis  → http://127.0.0.1:8102/   (분석 Agent)
    python main.py a2a-serve --role consult   → http://127.0.0.1:8100/   (상담 Agent 하나 - 다른 회사의 Agent 가 우리에게 물을 때)

- Agent Card(/.well-known/agent-card.json)로 자기 소개, JSON-RPC 로 일을 받고 Task(상태 + 산출물)로 답함.
- 대화 상태는 요청에 실려 오면 그것을 쓰고(분산 상담팀), 없으면 contextId 별로 잠깐 기억 (외부 Agent 의 이어서 묻기).
- 같은 messageId 로 다시 보내면 같은 Task 를 돌려줌 (재전송해도 두 번 실행하지 않음 = 멱등성).
- REA_A2A_TOKEN 을 정하면 Authorization: Bearer 토큰이 있어야 일을 받음. 기본은 내 PC(127.0.0.1)에서만 접속.
- 개인정보: 손님 화면과 같은 기준 (주소는 시/도 + 시/군/구, 연락처·지번·매물번호 속 이름 없음), 장기 기억은 쓰지 않음.
"""


import asyncio
import hmac
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from . import protocol as p
from .state import id_maps, private_state, public_state

log = logging.getLogger(__name__)

MAX_BODY = 256 * 1024
MAX_TEXT = 2000
MAX_TASKS = 500
MAX_CONTEXTS = 500
MAX_HISTORY_IN = 12


@dataclass
class Role:
    key: str
    label: str            # trace 에 표시할 이름
    name: str             # Agent Card 이름
    description: str
    tools: Optional[set[str]]
    prompt: str
    port: int
    skills: list[dict] = field(default_factory=list)
    max_steps: int = 4


def roles() -> dict[str, Role]:
    from ..tool_agent.loop import SYSTEM_PROMPT
    from ..tool_agent.multi import ANALYSIS_PROMPT, ANALYSIS_TOOLS, ANALYST, SEARCH_PROMPT, SEARCH_TOOLS, SEARCHER

    return {
        "search": Role("search", SEARCHER, "토지 매물 검색 Agent",
                       "충북 음성 토지 전문 중개사무소의 매물 검색 담당. 지역·용도·예산·면적 조건으로 찾고, "
                       "'그 중에', '다른 지역에도', '정말 없어?' 같은 이어서 묻기를 처리합니다. 가격 단위는 만원.",
                       SEARCH_TOOLS, SEARCH_PROMPT, 8101,
                       [{"id": "search_listings", "name": "토지 매물 검색",
                         "description": "조건에 맞는 토지 매물을 점수·가격·면적 순으로 찾음 (이어서 묻기 지원)",
                         "tags": ["부동산", "토지", "검색"],
                         "examples": ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개", "그 중에 넓은 건?",
                                      "음성군 말고 다른 지역에도 있어?"]}]),
        "analysis": Role("analysis", ANALYST, "토지 매물 분석 Agent",
                         "매물 상세, 여러 매물 비교, 창고·공장·주택 건축 가능 여부(1차 판단)를 맡습니다. "
                         "data 부분의 listing_ids 또는 state.last_shown 의 매물이 대상입니다.",
                         ANALYSIS_TOOLS, ANALYSIS_PROMPT, 8102,
                         [{"id": "compare_listings", "name": "매물 비교", "description": "가격·면적·평당가 비교",
                           "tags": ["비교"], "examples": ["두 매물 비교해줘"]},
                          {"id": "check_land_use", "name": "건축 가능 여부",
                           "description": "지목·용도지역·도로로 창고/공장/주택 건축 가능성 1차 판단",
                           "tags": ["토지이용"], "examples": ["창고 지을 수 있어?"]}]),
        "consult": Role("consult", "상담 Agent", "토지 상담 Agent",
                        "충북 음성 토지 전문 중개사무소의 상담 Agent. 매물을 찾고 비교하고 건축 가능 여부를 1차 판단해 "
                        "한국어로 짧게 답합니다. 같은 contextId 로 보내면 이어서 묻기가 됩니다.",
                        None, SYSTEM_PROMPT, 8100,
                        [{"id": "land_consultation", "name": "토지 매물 상담",
                          "description": "질문 하나로 검색·비교·건축 판단까지", "tags": ["부동산", "토지", "상담"],
                          "examples": ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘"]}], 6),
    }


def agent_card(role: Role, base_url: str, token: bool) -> dict[str, Any]:
    card: dict[str, Any] = {
        "protocolVersion": p.PROTOCOL_VERSION, "name": role.name, "description": role.description,
        "url": base_url, "preferredTransport": "JSONRPC", "version": "0.8.0",
        # A2A 1.0 클라이언트용: 같은 주소에서 1.0 / 0.3 형식을 모두 받음
        "supportedInterfaces": [{"url": base_url, "protocolBinding": "JSONRPC", "protocolVersion": p.PROTOCOL_VERSION_1},
                                {"url": base_url, "protocolBinding": "JSONRPC", "protocolVersion": "0.3"}],
        "provider": {"organization": "토지 전문 공인중개사무소 (충북 음성)", "url": base_url},
        "capabilities": {"streaming": True, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": role.skills,
    }
    if token:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


class A2AExecutor:
    """메시지 하나 -> Agent 실행 -> Task. (웹 서버 없이도 쓸 수 있게 분리)"""

    def __init__(self, agent, role: Role, lock: Optional[threading.RLock] = None, llm=None):
        self.agent, self.role, self.llm = agent, role, llm
        self.lock = lock or threading.RLock()
        self.tasks: OrderedDict[str, dict] = OrderedDict()
        self.by_message: dict[str, str] = {}
        self.contexts: OrderedDict[str, tuple[Any, list]] = OrderedDict()
        self._mu = threading.Lock()

    # ------------------------------------------------------------------ Task 보관
    def _keep(self, task: dict, message_id: str) -> None:
        with self._mu:
            self.tasks[task["id"]] = task
            self.by_message[message_id] = task["id"]
            while len(self.tasks) > MAX_TASKS:
                old_id, _ = self.tasks.popitem(last=False)
                self.by_message = {m: t for m, t in self.by_message.items() if t != old_id}

    def get(self, task_id: str) -> Optional[dict]:
        with self._mu:
            return self.tasks.get(task_id)

    def existing(self, message_id: str) -> Optional[dict]:
        with self._mu:
            tid = self.by_message.get(message_id)
            return self.tasks.get(tid) if tid else None

    def cancel(self, task_id: str) -> dict:
        task = self.get(task_id)
        if task is None:
            raise p.A2AError(p.TASK_NOT_FOUND, "Task 를 찾을 수 없습니다")
        with self._mu:
            if task["status"]["state"] in p.TERMINAL:
                raise p.A2AError(p.TASK_NOT_CANCELABLE, f"이미 끝난 Task 입니다 ({task['status']['state']})")
            task["status"] = {"state": "canceled", "timestamp": p.now()}  # 실행 중인 결과는 버림
        return task

    # ------------------------------------------------------------------ 실행
    def validate(self, params: Any) -> dict:
        msg = params.get("message") if isinstance(params, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "user" or not isinstance(msg.get("parts"), list) \
                or not isinstance(msg.get("messageId"), str) or not msg["messageId"]:
            raise p.A2AError(p.INVALID_PARAMS, "params.message 에 role='user', parts(목록), messageId 가 필요합니다")
        text = p.get_text(msg["parts"])
        if not text:
            raise p.A2AError(p.INVALID_PARAMS, "글자(text) 부분이 비어 있습니다")
        if len(text) > MAX_TEXT:
            raise p.A2AError(p.INVALID_PARAMS, f"글자는 {MAX_TEXT}자까지입니다")
        if msg.get("taskId"):
            raise p.A2AError(p.UNSUPPORTED_OPERATION, "끝난 Task 를 이어가는 기능은 없습니다. contextId 만 주세요")
        return msg

    def start(self, msg: dict) -> dict:
        task = {"kind": "task", "id": p.new_id(), "contextId": msg.get("contextId") or p.new_id(),
                "status": {"state": "submitted", "timestamp": p.now()}, "history": [msg], "artifacts": [],
                "metadata": {"agent": self.role.name}}
        self._keep(task, msg["messageId"])
        return task

    def run(self, task: dict, on_update: Optional[Callable[[dict], None]] = None) -> dict:
        """Task 를 끝까지 실행 (on_update: 진행 상황 이벤트 - 스트리밍용)."""
        from ..tool_agent import ToolAgent
        from ..tool_agent.loop import _loads
        from ..tool_agent.multi import text_history
        from ..tool_agent.tools import AgentContext

        msg = task["history"][0]
        cid = task["contextId"]
        emit = on_update or (lambda e: None)
        task["status"] = {"state": "working", "timestamp": p.now()}
        emit({"kind": "status-update", "taskId": task["id"], "contextId": cid, "status": task["status"], "final": False})
        started = time.perf_counter()
        try:
            text, data = p.get_text(msg["parts"]), p.get_data(msg["parts"])
            with self._mu:
                cached = self.contexts.get(cid)
            if "state" in data:
                ctx = private_state(self.agent, data["state"])
            else:
                ctx = cached[0] if cached else AgentContext()
            history = data.get("history") if "history" in data else (cached[1] if cached else [])
            history = [{"role": h["role"], "content": str(h["content"])[:MAX_TEXT]} for h in (history or [])
                       if isinstance(h, dict) and h.get("role") in {"user", "assistant"} and h.get("content")
                       ][-MAX_HISTORY_IN:]
            to_public, to_real = id_maps(self.agent)
            ids = data.get("listing_ids") or []
            if ids:
                if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                    raise ValueError("listing_ids 는 글자 목록이어야 합니다")
                real = [to_real(i) for i in ids]
                if not all(self.agent.find_listing(r) for r in real):
                    raise ValueError("없는 매물 id: " + ", ".join(i for i, r in zip(ids, real)
                                                                 if not self.agent.find_listing(r)))
                ctx.last_shown = list(dict.fromkeys(real))
                text += f"\n(대상 매물 id: {', '.join(ids)})"

            def step_event(e: dict) -> None:
                if e.get("type") == "step":
                    s = e["step"]
                    line = f"[{s.get('agent') or self.role.label}] " + (
                        f"{s['tool']} → {s['summary']}" if s["kind"] == "tool" else f"{s['kind']} · {s['summary']}")
                    emit({"kind": "status-update", "taskId": task["id"], "contextId": cid, "final": False,
                          "status": {"state": "working", "timestamp": p.now(),
                                     "message": p.message("agent", [p.text_part(line), p.data_part({"step": s})],
                                                          cid, task["id"])}})

            runner = ToolAgent(self.agent, llm=self.llm, ctx=ctx, lock=self.lock, log_dir="", only=self.role.tools,
                               system_prompt=self.role.prompt, name=self.role.label, on_event=step_event,
                               max_steps=self.role.max_steps)
            if self.role.key != "consult":
                runner.answer_label = "총괄 Agent 에게 보고"
                runner.final_answerer = False
            result, messages = runner.run(text, history)
            start = max(i for i, m in enumerate(messages) if m["role"] == "user")
            observations = [[m["name"], _loads(m["content"])] for m in messages[start:] if m["role"] == "tool"]
            with self._mu:
                self.contexts[cid] = (ctx, text_history(messages)[-MAX_HISTORY_IN:])
                self.contexts.move_to_end(cid)
                while len(self.contexts) > MAX_CONTEXTS:
                    self.contexts.popitem(last=False)
            payload = {
                "agent": self.role.label, "answer": result.answer,
                "listing_ids": [to_public(i) for i in result.listing_ids],
                "badges": {to_public(k): v for k, v in result.badges.items()},
                "steps": [asdict(s) for s in result.steps], "observations": observations,
                "state": public_state(self.agent, ctx), "run_searches": getattr(runner, "last_run_searches", []),
                "policy": result.policy, "stopped": result.stopped, "tool_calls": result.tool_calls,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            artifact = {"artifactId": p.new_id(), "name": "consultation_result",
                        "parts": [p.text_part(result.answer), p.data_part(payload)]}
            with self._mu:
                if task["status"]["state"] == "canceled":
                    return task
                task["artifacts"] = [artifact]
                task["status"] = {"state": "completed", "timestamp": p.now(),
                                  "message": p.message("agent", [p.text_part(result.answer)], cid, task["id"])}
            emit({"kind": "artifact-update", "taskId": task["id"], "contextId": cid, "artifact": artifact,
                  "append": False, "lastChunk": True})
        except ValueError as e:
            task["status"] = {"state": "failed", "timestamp": p.now(),
                              "message": p.message("agent", [p.text_part(f"요청을 처리할 수 없습니다: {e}")], cid,
                                                   task["id"])}
        except Exception as e:  # noqa: BLE001 - 실패도 Task 상태로 알림
            log.exception("A2A task 실패")
            task["status"] = {"state": "failed", "timestamp": p.now(),
                              "message": p.message("agent", [p.text_part(f"처리 중 오류: {type(e).__name__}")], cid,
                                                   task["id"])}
        emit({"kind": "status-update", "taskId": task["id"], "contextId": cid, "status": task["status"], "final": True})
        return task


def create_a2a_app(role_key: str = "consult", agent=None, settings=None, base_url: Optional[str] = None,
                   token: Optional[str] = None, lock: Optional[threading.RLock] = None, llm=None):
    """A2A 서버 (FastAPI). token 이 없으면 REA_A2A_TOKEN 환경변수를 씀 (둘 다 없으면 인증 없음)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..agent import RealEstateAgent
    from ..config import get_settings

    role = roles()[role_key]
    if agent is None:
        agent = RealEstateAgent(settings or get_settings())
        agent.sync()
    token = token if token is not None else os.getenv("REA_A2A_TOKEN", "")
    base_url = base_url or f"http://127.0.0.1:{role.port}/"
    executor = A2AExecutor(agent, role, lock, llm)
    card = agent_card(role, base_url, bool(token))
    app = FastAPI(title=role.name + " (A2A)", version="0.8.0", docs_url=None, redoc_url=None)
    app.state.executor = executor

    @app.get(p.CARD_PATH)
    @app.get(p.OLD_CARD_PATH)
    async def get_card() -> JSONResponse:
        return JSONResponse(card)

    @app.post("/")
    async def rpc(request: Request):
        if token:
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                return JSONResponse({"error": "인증이 필요합니다 (Authorization: Bearer …)"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        body_bytes = await request.body()
        if len(body_bytes) > MAX_BODY:
            return JSONResponse(p.rpc_error(None, p.INVALID_REQUEST, "요청이 너무 큽니다"), status_code=413)
        try:
            body = json.loads(body_bytes)
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(p.rpc_error(None, p.PARSE_ERROR, "JSON 형식이 아닙니다"))
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            return JSONResponse(p.rpc_error(body.get("id") if isinstance(body, dict) else None, p.INVALID_REQUEST,
                                            "JSON-RPC 2.0 요청이 아닙니다"))
        req_id, method, params = body.get("id"), body["method"], body.get("params") or {}
        v1 = method in p.V1_METHODS  # A2A 1.0 형식으로 온 요청 -> 답도 1.0 형식으로
        if v1:
            method = p.V1_METHODS[method]
            if isinstance(params, dict) and "message" in params:
                params = {**params, "message": p.from_v1_message(params["message"])}
        task_out = (lambda t: p.to_v1_task(t)) if v1 else (lambda t: t)
        try:
            if method == "message/send":
                msg = executor.validate(params)
                task = executor.existing(msg["messageId"])
                if task is None:
                    task = await asyncio.to_thread(executor.run, executor.start(msg))
                return JSONResponse(p.rpc_result(req_id, {"task": p.to_v1_task(task)} if v1 else task))
            if method == "message/stream":
                msg = executor.validate(params)
                return StreamingResponse(_stream(executor, msg, req_id, v1), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            if method == "tasks/get":
                task = executor.get(str((params or {}).get("id", "")))
                if task is None:
                    raise p.A2AError(p.TASK_NOT_FOUND, "Task 를 찾을 수 없습니다")
                return JSONResponse(p.rpc_result(req_id, task_out(task)))
            if method == "tasks/cancel":
                return JSONResponse(p.rpc_result(req_id, task_out(executor.cancel(str((params or {}).get("id", ""))))))
            if method in {"ListTasks", "SubscribeToTask", "GetExtendedAgentCard", "tasks/resubscribe"}:
                raise p.A2AError(p.UNSUPPORTED_OPERATION, f"지원하지 않는 기능: {method}")
            if method.startswith("tasks/pushNotificationConfig/") or "PushNotificationConfig" in method:
                raise p.A2AError(p.PUSH_NOT_SUPPORTED, "푸시 알림은 지원하지 않습니다 (message/stream 을 쓰세요)")
            raise p.A2AError(p.METHOD_NOT_FOUND, f"없는 method: {method}")
        except p.A2AError as e:
            return JSONResponse(p.rpc_error(req_id, e.code, e.message, e.data))

    return app


async def _stream(executor: A2AExecutor, msg: dict, req_id: Any, v1: bool = False):
    """message/stream: Task → 진행 상황(status-update) → 산출물(artifact-update) → 끝(final) 을 SSE 로."""
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    task = executor.existing(msg["messageId"])
    out = p.to_v1_event if v1 else (lambda e: e)
    if task is not None and task["status"]["state"] in p.TERMINAL:  # 같은 메시지 재전송 -> 결과만 다시
        yield _sse(req_id, out(task))
        return
    task = executor.start(msg)
    yield _sse(req_id, out(dict(task)))
    job = loop.run_in_executor(None, executor.run, task, lambda e: loop.call_soon_threadsafe(events.put_nowait, e))
    while True:
        try:
            event = await asyncio.wait_for(events.get(), 1.0)
        except asyncio.TimeoutError:
            if job.done() and events.empty():
                break
            continue
        yield _sse(req_id, out(event))
        if event.get("kind") == "status-update" and event.get("final"):
            break
    await job


def _sse(req_id: Any, result: dict) -> str:
    return f"data: {json.dumps(p.rpc_result(req_id, result), ensure_ascii=False, default=str)}\n\n"


def main(role_key: str, settings=None, host: str = "127.0.0.1", port: Optional[int] = None) -> int:
    import uvicorn

    role = roles()[role_key]
    port = port or role.port
    shown_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    app = create_a2a_app(role_key, settings=settings, base_url=f"http://{shown_host}:{port}/")
    print(f"{role.name} (A2A) 시작: http://{shown_host}:{port}/   Agent Card: http://{shown_host}:{port}{p.CARD_PATH}")
    if os.getenv("REA_A2A_TOKEN"):
        print("  인증: REA_A2A_TOKEN (Authorization: Bearer …)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
