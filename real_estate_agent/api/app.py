"""토지 매물 추천 Agent JSON API (FastAPI).

실행:  python main.py api        →  http://127.0.0.1:8001/docs  (Swagger 화면에서 바로 시험)

- async/await : 모든 엔드포인트가 async. 매물 검색·Agent 실행(동기 코드·LLM 호출)은 asyncio.to_thread 로
                별도 스레드에서 돌려 서버가 다른 요청을 계속 받음
- 구조화 출력 : 요청·응답을 Pydantic 모델로 정의 (형식이 틀린 요청은 422 로 거절, /docs 에 스키마 표시)
- 스트리밍     : POST /api/v1/agent/stream - Agent 가 도구를 쓰는 과정을 Server-Sent Events 로 실시간 전달
- 개인정보     : 손님 화면과 같은 기준 (주소는 시/도 + 시/군/구, 연락처 없음, 매물번호 속 이름 가림)
- 방문자 구분  : X-Visitor-Id 헤더 (없으면 새로 만들어 응답에 돌려줌) → 대화·장기 기억이 방문자별로 저장
"""

from __future__ import annotations

import asyncio
import re
import secrets
import threading
import time
from typing import Any, AsyncIterator, Literal, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent import RealEstateAgent, mask_address, public_office_no
from ..config import Settings, get_settings
from ..models import format_price
from ..tool_agent.service import AgentService, Busy
from ..tool_agent.tools import PURPOSES, ToolRegistry, display_name, tool_id

API_VERSION = "0.3.0"
_VISITOR = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
Sort = Literal["score", "price_asc", "price_desc", "area_desc", "price_per_pyeong_asc"]


# ---------------------------------------------------------------------- 구조화된 요청/응답 (Pydantic)
class ListingOut(BaseModel):
    id: str = Field(description="매물 id (상세 조회·비교에 사용)")
    office_no: Optional[str] = Field(None, description="엑셀에 적힌 매물번호")
    name: str = Field(description="손님에게 부르는 이름 (매물번호, 없으면 '음성군 물류부지 15,005평 매물')")
    region: str = Field(description="시/도 + 시/군/구 (나머지는 가림)")
    price_manwon: int
    price_text: str
    area_pyeong: int
    price_per_pyeong_manwon: int
    jimok: Optional[str] = None
    zoning: Optional[str] = None
    use_tag: Optional[str] = None
    features: list[str] = []
    badges: list[str] = Field(default_factory=list, description="최저가, 가장 넓음, 창고 가능 …")


class SearchRequest(BaseModel):
    query: str = Field("", max_length=300, examples=["음성군 창고 지을 땅 3억 이하"])
    regions: list[str] = []
    purpose: Optional[Literal[tuple(PURPOSES)]] = None  # type: ignore[valid-type]
    budget_max_manwon: Optional[int] = Field(None, ge=0)
    area_min_pyeong: Optional[float] = Field(None, ge=0)
    sort: Sort = "score"
    limit: int = Field(5, ge=1, le=10)


class SearchResponse(BaseModel):
    understood: str = Field(description="이해한 조건")
    total: int = Field(description="조건에 맞는 매물 수")
    listings: list[ListingOut]
    message: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300, examples=["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘"])
    conversation_id: Optional[str] = Field(None, description="이어서 묻기: 앞 응답의 conversation_id")
    mode: Literal["single", "multi", "distributed"] = Field(
        "single", description="single: Agent 하나 / multi: 총괄 → 검색·분석 Agent (v0.7) / distributed: 검색·분석 Agent 가 A2A 서버 (v0.8)")


class StepOut(BaseModel):
    n: int
    kind: str
    tool: str = ""
    args: dict[str, Any] = {}
    ok: Optional[bool] = None
    summary: str = ""
    ms: float = 0
    agent: str = Field("", description="멀티 Agent 에서 이 단계를 한 Agent (총괄 / 검색 / 분석)")


class AskResponse(BaseModel):
    conversation_id: str
    visitor_id: str
    answer: str
    listings: list[ListingOut]
    steps: list[StepOut]
    policy: str = Field(description="답을 만든 쪽: openai / anthropic / rule")
    total_ms: float


class TurnOut(BaseModel):
    question: str
    answer: str
    listing_ids: list[str]
    at: Optional[str] = None


class ConversationOut(BaseModel):
    conversation_id: str
    status: str
    pending_question: Optional[str] = None
    turns: list[TurnOut]


class Health(BaseModel):
    status: str
    version: str
    listings: int
    llm: str


# ---------------------------------------------------------------------- 앱
def create_api(settings: Optional[Settings] = None, agent: Optional[RealEstateAgent] = None,
               lock: Optional[threading.RLock] = None, start_realtime: bool = False) -> FastAPI:
    from ..realtime import RealtimeHub

    settings = settings or get_settings()
    if agent is None:
        agent = RealEstateAgent(settings)
        agent.sync()
    lock = lock or threading.RLock()
    service = AgentService(agent, lock)
    hub = RealtimeHub(agent, lock)

    @asynccontextmanager
    async def lifespan(_app):  # 서버가 켜질 때 매물 폴더 감시·매칭 워커 시작, 꺼질 때 정지
        if start_realtime:
            hub.start()
        yield
        hub.stop()

    app = FastAPI(title="토지 매물 추천 Agent API", version=API_VERSION, lifespan=lifespan,
                  description="부동산 사무실 토지 매물 검색과 AI 상담(Tool-use Agent). 주소는 시/도 + 시/군/구까지만 제공합니다.")
    app.state.agent, app.state.service, app.state.realtime = agent, service, hub

    # v1.1 관측: span 저장 + JSON 로그 + 요청 지표
    import hmac
    import os

    from .. import observability
    from ..observability.metrics import HTTP_LATENCY, HTTP_REQUESTS, METRICS

    observability.setup(settings if settings.db_path else agent.settings, "api")
    METRICS.gauge("rea_listings", "보유 매물 수", lambda: len(agent.listings()))
    METRICS.gauge("rea_active_consultations", "지금 처리 중인 AI 상담 수", lambda: service.bulkhead.active)

    @app.middleware("http")
    async def http_metrics(request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "other")  # 주소 패턴별
        HTTP_REQUESTS.inc(route=route, method=request.method, status=response.status_code)
        if response.media_type != "text/event-stream":
            HTTP_LATENCY.observe(time.perf_counter() - t0, route=route)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> PlainTextResponse:
        """Prometheus 지표. 이 PC(127.0.0.1)에서만, 또는 REA_METRICS_TOKEN 을 Bearer 로."""
        token = os.getenv("REA_METRICS_TOKEN", "")
        given = request.headers.get("authorization", "")
        local = request.client is not None and request.client.host in {"127.0.0.1", "::1", "testclient"}
        if not ((token and hmac.compare_digest(given.encode(), f"Bearer {token}".encode())) or (not token and local)):
            raise HTTPException(403, "지표는 이 PC 또는 REA_METRICS_TOKEN 으로만 볼 수 있습니다")
        return PlainTextResponse(METRICS.prometheus(), media_type="text/plain; version=0.0.4")

    def owner_of(visitor: Optional[str]) -> tuple[str, str]:
        vid = visitor if visitor and _VISITOR.match(visitor) else secrets.token_urlsafe(18)
        return vid, "api:" + vid

    def listing_out(lid: str, badges: Optional[list[str]] = None) -> Optional[ListingOut]:
        l = agent.find_listing(lid)
        if l is None:
            return None
        return ListingOut(
            id=tool_id(l), office_no=public_office_no(l.office_no) or None, name=display_name(l),
            region=mask_address(l.address), price_manwon=l.price_manwon, price_text=format_price(l.price_manwon),
            area_pyeong=round(l.area_pyeong), price_per_pyeong_manwon=round(l.price_per_pyeong),
            jimok=l.jimok or None, zoning=l.zoning or None, use_tag=l.use_tag or None,
            features=[f.lstrip("● ").strip() for f in (l.features or "").split(" / ") if f.strip()],
            badges=badges or [])

    @app.get("/api/v1/health", response_model=Health, tags=["상태"])
    async def health() -> Health:
        return Health(status="ok", version=API_VERSION, listings=len(agent.listings()), llm=agent.llm.mode())

    @app.get("/api/v1/ready", tags=["상태"])
    async def ready():
        """v1.0 준비 상태: ok / degraded(일부 대체 모드) / fail(503)."""
        from fastapi.responses import JSONResponse

        from ..reliability.health import http_status, readiness

        report = await asyncio.to_thread(readiness, agent, lock, hub, service)
        return JSONResponse(report, status_code=http_status(report))

    @app.post("/api/v1/listings/search", response_model=SearchResponse, tags=["매물"])
    async def search(body: SearchRequest) -> SearchResponse:
        args = {k: v for k, v in body.model_dump().items() if v not in (None, "", [])}
        reg = ToolRegistry(agent, lock=lock)
        obs = await asyncio.to_thread(reg.run, "search_listings", args)
        if not obs["ok"]:
            raise HTTPException(400, obs["error"])
        r = obs["result"]
        items = [listing_out(reg.resolve(x["id"]) or x["id"]) for x in r["매물"]]
        return SearchResponse(understood=r["이해한_조건"], total=r["조건에_맞는_매물_수"],
                              listings=[i for i in items if i], message=r.get("안내"))

    @app.get("/api/v1/listings/{listing_id}", response_model=ListingOut, tags=["매물"])
    async def get_listing(listing_id: str) -> ListingOut:
        reg = ToolRegistry(agent, lock=lock)
        real = reg.resolve(listing_id)
        out = listing_out(real) if real else None
        if out is None:
            raise HTTPException(404, "매물을 찾을 수 없습니다")
        return out

    @app.post("/api/v1/agent/ask", response_model=AskResponse, tags=["AI 상담"])
    async def ask(body: AskRequest, x_visitor_id: Optional[str] = Header(None)) -> AskResponse:
        vid, owner = owner_of(x_visitor_id)
        try:
            cid, turn = await asyncio.to_thread(service.ask, owner, body.conversation_id, body.question, None, body.mode)
        except Busy as e:  # 409 같은 대화 처리 중 / 429 너무 잦음 / 503 상담 몰림 (v1.0)
            headers = {"Retry-After": str(e.retry_after)} if getattr(e, "retry_after", None) else None
            raise HTTPException(e.status, str(e), headers=headers) from e
        listings = [listing_out(i, turn["badges"].get(i)) for i in turn["listing_ids"]]
        return AskResponse(conversation_id=cid, visitor_id=vid, answer=turn["answer"],
                           listings=[x for x in listings if x], steps=[StepOut(**s) for s in turn["steps"]],
                           policy=turn["policy"], total_ms=turn["total_ms"])

    @app.post("/api/v1/agent/stream", tags=["AI 상담"],
              responses={200: {"content": {"text/event-stream": {}},
                               "description": "event: thinking / tool_start / step / done / saved / error"}})
    async def stream(body: AskRequest, x_visitor_id: Optional[str] = Header(None)) -> StreamingResponse:
        vid, owner = owner_of(x_visitor_id)
        conv = service.open(owner, body.conversation_id, reuse_recent=False)

        async def events() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue()

            def produce() -> None:  # Agent 는 스레드에서, 진행 상황은 이벤트 루프로 전달
                for chunk in service.stream(owner, conv["cid"], body.question, mode=body.mode):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=produce, daemon=True).start()
            while (chunk := await q.get()) is not None:
                yield chunk

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Conversation-Id": conv["cid"],
                                          "X-Visitor-Id": vid})

    @app.get("/api/v1/agent/conversations/{conversation_id}", response_model=ConversationOut, tags=["AI 상담"])
    async def conversation(conversation_id: str, x_visitor_id: Optional[str] = Header(None)) -> ConversationOut:
        _, owner = owner_of(x_visitor_id)
        conv = await asyncio.to_thread(service.store.load, conversation_id)
        if conv is None or conv["owner"] != owner:
            raise HTTPException(404, "대화를 찾을 수 없습니다 (X-Visitor-Id 확인)")
        return ConversationOut(conversation_id=conv["cid"], status=conv["status"], pending_question=conv["pending"],
                               turns=[TurnOut(**{k: t.get(k) for k in TurnOut.model_fields}) for t in conv["turns"]])

    # ------------------------------------------------------------------ v0.9 실시간
    @app.get("/api/v1/realtime/stats", tags=["실시간"])
    async def realtime_stats() -> dict[str, Any]:
        return {"running": hub.running, "interval_sec": hub.interval, "alert_threshold": hub.threshold,
                "subscribers": hub.bus.subscribers, "events": hub.bus.counts(), **hub.stats.as_dict()}

    @app.get("/api/v1/events/stream", tags=["실시간"],
             responses={200: {"content": {"text/event-stream": {}},
                              "description": "listing.added / listing.price_drop / listing.changed / listing.removed / "
                                             "system.sync / alert.match(이 방문자 맞춤)"}})
    async def events_stream(since: Optional[int] = None, max_seconds: int = Query(300, ge=1, le=300),
                            x_visitor_id: Optional[str] = Header(None),
                            last_event_id: Optional[str] = Header(None)) -> StreamingResponse:
        """연결 하나는 max_seconds(최대 5분) 뒤 닫힘 -> 클라이언트가 Last-Event-ID 로 다시 연결해 이어 받음."""
        _, owner = owner_of(x_visitor_id)
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else (
            since if since is not None else hub.bus.last_id())

        async def gen() -> AsyncIterator[str]:
            q = hub.bus.subscribe(owner)
            sent = start
            try:
                yield "retry: 5000\n\n"
                for ev in await asyncio.to_thread(hub.bus.since, start, owner):
                    sent = ev.id
                    yield _sse(ev)
                deadline = time.monotonic() + max_seconds
                while time.monotonic() < deadline:
                    ev = await _next_event(q, heartbeat=min(15.0, max(1.0, deadline - time.monotonic())))
                    if ev is None:
                        yield ": ping\n\n"
                        continue
                    if ev.id > sent:
                        sent = ev.id
                        yield _sse(ev)
            finally:
                hub.bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.websocket("/api/v1/ws")
    async def websocket(ws: WebSocket, visitor_id: Optional[str] = None) -> None:
        """양방향 실시간: 서버 → 매물 이벤트·맞춤 알림을 바로 보냄 / 클라이언트 → {"type":"ask","question":...} 를
        보내면 Agent 가 도구를 쓰는 과정을 단계마다 보내고 마지막에 답변."""
        await ws.accept()
        vid, owner = owner_of(visitor_id)
        send_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        async def send(data: dict) -> None:
            async with send_lock:
                await ws.send_json(data)

        await send({"type": "hello", "visitor_id": vid, "last_event_id": hub.bus.last_id()})
        q = hub.bus.subscribe(owner)

        async def push_events() -> None:
            while True:
                ev = await _next_event(q)
                if ev is None:
                    await send({"type": "ping"})
                    continue
                await send({"type": "event", "event": ev.public()})

        async def answer(msg: dict) -> None:
            question = str(msg.get("question") or "").strip()[:300]
            if not question:
                await send({"type": "error", "message": "question 이 비어 있습니다"})
                return
            steps: asyncio.Queue = asyncio.Queue()
            job = asyncio.ensure_future(asyncio.to_thread(
                service.ask, owner, msg.get("conversation_id"), question,
                lambda e: loop.call_soon_threadsafe(steps.put_nowait, e),
                msg.get("mode") if msg.get("mode") in {"multi", "distributed"} else "single"))
            while not job.done() or not steps.empty():
                try:
                    e = await asyncio.wait_for(steps.get(), 0.2)
                except asyncio.TimeoutError:
                    continue
                await send({"type": "agent", "kind": e.get("type"), **{k: v for k, v in e.items() if k != "type"}})
            try:
                cid, turn = job.result()
            except Busy as e:
                await send({"type": "error", "message": str(e)})
                return
            listings = [listing_out(i, turn["badges"].get(i)) for i in turn["listing_ids"]]
            await send({"type": "answer", "conversation_id": cid, "answer": turn["answer"],
                        "listings": [x.model_dump() for x in listings if x]})

        pusher = asyncio.ensure_future(push_events())
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "ask":
                    await answer(msg)
                elif isinstance(msg, dict) and msg.get("type") == "ping":
                    await send({"type": "pong"})
                else:
                    await send({"type": "error", "message": '보낼 수 있는 메시지: {"type":"ask","question":"..."}'})
        except WebSocketDisconnect:
            pass
        finally:
            pusher.cancel()
            hub.bus.unsubscribe(q)

    @app.delete("/api/v1/agent/memory", status_code=204, tags=["AI 상담"])
    async def forget(x_visitor_id: Optional[str] = Header(None)) -> None:
        if not x_visitor_id or not _VISITOR.match(x_visitor_id):
            raise HTTPException(400, "X-Visitor-Id 헤더가 필요합니다")
        await asyncio.to_thread(service.forget, "api:" + x_visitor_id)

    return app


def _sse(ev) -> str:
    import json

    return f"id: {ev.id}\nevent: {ev.topic}\ndata: {json.dumps(ev.public(), ensure_ascii=False, default=str)}\n\n"


async def _next_event(q, heartbeat: float = 15.0, poll: float = 1.0):
    """구독 큐에서 다음 이벤트. heartbeat 초 동안 없으면 None (연결 유지용 ping).
    스레드에서 1초씩 나눠 기다려, 연결이 끊기거나 서버가 꺼질 때 바로 정리되게 함."""
    import queue as _queue

    waited = 0.0
    while waited < heartbeat:
        try:
            return await asyncio.to_thread(q.get, True, poll)
        except _queue.Empty:
            waited += poll
    return None
