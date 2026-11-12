"""Real Estate Recommendation Agent - 웹 서비스 (Flask).

손님 화면
  /                검색창 ("음성군 채광 좋은 2억원대 땅")
  /result          추천 매물 카드 + [마음에 안 들어요 → 다음 순위] / 빠른 사유 / 관심 / 이전 / 조건 변경
  /api/explain     현재 매물 AI 설명 (화면이 먼저 뜬 뒤 불러옴)
  /agent           AI 상담 (v0.2 Tool-use Agent: 도구를 골라 쓰며 여러 단계 질문에 답함, 실행 과정 표시)
관리자 화면 (처음 /admin 에 들어가면 비밀번호를 정하는 화면이 나옴 - 서버를 켠 PC 에서만)
  /admin/listings  매물 관리: 새 매물 추가, 기존 매물 수정·숨기기·되돌리기
  /admin           매물 다시 읽기, 오류·중복 보고서, 매물 파일 업로드, AI 연결 상태, 학습 현황

개인정보: 손님 화면에는 소유자 연락처·원본 파일·정확한 지번을 보여주지 않습니다. (관리자 로그인 시에만 표시)
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import re
import secrets
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from ..admin_auth import MIN_LENGTH as MIN_PASSWORD_LEN
from ..admin_auth import AdminAuth, auth_file
from ..agent import RealEstateAgent, is_condition_change, mask_address, public_office_no
from ..collectors.local_folder import SUPPORTED
from ..file_uploads import default_folder as default_upload_folder
from ..file_uploads import destination_folders, files_in, listing_folder, save_uploads
from ..listing_editor import HIDDEN, STATUSES as LISTING_STATUSES, WEB_FOLDER
from ..listing_editor import describe_changes, form_defaults, parse_form, web_file
from ..models import JIMOK_TYPES
from ..requirement_parser import ZONING_TYPES
from ..config import PROJECT_ROOT, Settings, get_settings, is_sample_data
from ..models import DEAL_TYPES, PROPERTY_TYPES, format_price
from ..recommender import EXHAUSTED_MESSAGE, NOT_FOUND_MESSAGE, RecommendationSession

log = logging.getLogger(__name__)

MAX_QUERY_LEN = 300
MAX_UPLOAD_MB = 200  # 폴더째 올릴 때 전체 크기 (파일 하나는 20MB 까지)
SESSION_TTL_SEC = 2 * 60 * 60
MAX_SESSIONS = 500
# 사무실 안내 (모든 화면 아래쪽). 바뀌면 여기만 고치면 됩니다.
OFFICE_INFO = {
    "address": "충청북도 음성군 대소읍 오태로 137번길 13",
    "phone": "043-877-5242",
    "hours": [("월~토", "10:00 ~ 21:00"), ("일요일", "10:00 ~ 18:00")],
}
QUICK_REASONS = ["너무 비싸요", "너무 좁아요", "위치가 별로예요", "도로가 별로예요", "채광이 별로예요"]
EXAMPLES = ["음성군 채광이 좋은 2억원대 땅", "양평 전원주택 지을 땅 200평 정도 3억 이하",
            "여주에 창고 지을 땅 500평 이상", "귀농용 논이나 밭 1000평 2억 안쪽"]


class SessionStore:
    """접속자별 추천 세션을 서버 메모리에 보관 (2시간 동안 사용 없으면 삭제)."""

    def __init__(self, ttl: int = SESSION_TTL_SEC, max_size: int = MAX_SESSIONS):
        self.ttl, self.max_size = ttl, max_size
        self._data: OrderedDict[str, tuple[RecommendationSession, str, float]] = OrderedDict()

    def put(self, sess: RecommendationSession, query: str) -> str:
        self._evict()
        sid = secrets.token_urlsafe(16)
        self._data[sid] = (sess, query, time.time())
        return sid

    def get(self, sid: Optional[str]) -> Optional[tuple[RecommendationSession, str]]:
        if not sid or sid not in self._data:
            return None
        sess, query, ts = self._data[sid]
        if time.time() - ts > self.ttl:
            del self._data[sid]
            return None
        self._data[sid] = (sess, query, time.time())
        self._data.move_to_end(sid)
        return sess, query

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, (_, _, ts) in self._data.items() if now - ts > self.ttl]:
            del self._data[sid]
        while len(self._data) >= self.max_size:
            self._data.popitem(last=False)


def _secret_key(settings: Settings) -> str:
    key = os.getenv("REA_SECRET_KEY")
    if key:
        return key
    path = settings.db_path.parent / "secret_key"
    try:
        if path.exists():
            return path.read_text().strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        path.write_text(key)
        return key
    except OSError:
        return secrets.token_hex(32)


VISITOR_COOKIE = "rea_vid"          # 방문자 id (장기 기억용, 90일) - 관리자 로그인과는 별개
VISITOR_DAYS = 90
_VID = re.compile(r"^[A-Za-z0-9_-]{20,64}$")


def safe_filename(name: str) -> str:
    """업로드 파일 이름 정리 (한글 유지, 경로·특수문자 제거)."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return name or "upload"


def shown_event(agent, ev):
    """화면·알림으로 다시 보낼 이벤트: 지금 없는 매물에 대한 손님 알림은 빼고, 나머지는 없는 매물번호만 가림."""
    from dataclasses import replace

    from ..listing_numbers import event_listing_exists, hide_in, known_numbers

    known = known_numbers(agent)
    if ev.topic == "alert.match" and not event_listing_exists(ev.payload, known):
        return None
    return replace(ev, payload=hide_in(ev.payload, known))


def create_app(settings: Optional[Settings] = None, agent: Optional[RealEstateAgent] = None) -> Flask:
    settings = settings or get_settings()
    agent = agent or RealEstateAgent(settings)
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=_secret_key(settings),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("REA_COOKIE_SECURE", "0") == "1",  # https 로 배포하면 1
    )
    lock = threading.RLock()  # 매물 목록·DB·학습 모델을 여러 접속자가 동시에 바꾸지 않도록
    store = SessionStore()
    from ..tool_agent.service import AgentService, Busy

    agent_service = AgentService(agent, lock)
    app.config["AGENT_SERVICE"] = agent_service  # 테스트용
    state = {"sync_error": ""}

    def do_sync(detect: bool = True) -> None:
        try:
            agent.sync()
            state["sync_error"] = ""
        except FileNotFoundError as e:
            state["sync_error"] = str(e)
            log.warning("매물 폴더 읽기 실패 - 저장된 매물로 계속: %s", e)
        if detect:
            changed("관리자 화면")

    def changed(source: str) -> None:
        """매물 목록이 바뀐 직후: 실시간 이벤트 발행 (새 매물·가격 인하 → 찾던 손님에게 알림)."""
        hub = app.config.get("REALTIME")
        if hub is not None:
            hub.detect_changes(source)

    with lock:
        do_sync(detect=False)

    # v0.9 실시간: 이벤트 버스 + 변경 감지 + 매칭 워커 (스레드는 run() 에서 시작, 테스트에서는 직접 drain)
    from ..realtime import RealtimeHub

    realtime = RealtimeHub(agent, lock)
    realtime.on_sync = lambda: do_sync(detect=False)
    realtime.bus.display = lambda ev: shown_event(agent, ev)  # 예전(샘플·지운) 매물 이벤트는 다시 보내지 않음
    app.config["REALTIME"] = realtime
    app.config["SAMPLE_MODE"] = is_sample_data(settings.data_dir)

    app.config["AGENT"] = agent  # 테스트용

    # v1.1 관측: span 저장(SQLite·선택 OTLP) + JSON 로그 + 읽을 때 계산하는 지표
    from .. import observability
    from ..observability.metrics import HTTP_LATENCY, HTTP_REQUESTS, METRICS

    observability.setup(settings, "web")
    METRICS.gauge("rea_listings", "보유 매물 수", lambda: len(agent.listings()))
    METRICS.gauge("rea_active_consultations", "지금 처리 중인 AI 상담 수", lambda: agent_service.bulkhead.active)
    METRICS.gauge("rea_waiting_consultations", "상담 자리를 기다리는 수", lambda: agent_service.bulkhead.waiting)

    # ------------------------------------------------------------ 공통
    def is_admin() -> bool:
        return bool(session.get("admin"))

    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(24)
        return session["csrf"]

    @app.before_request
    def start_timer():
        request._t0 = time.perf_counter()  # type: ignore[attr-defined]

    @app.after_request
    def http_metrics(resp):
        route = request.url_rule.rule if request.url_rule else "other"  # 주소 패턴별 (매물 번호마다 따로 세지 않음)
        HTTP_REQUESTS.inc(route=route, method=request.method, status=resp.status_code)
        t0 = getattr(request, "_t0", None)
        if t0 is not None and resp.mimetype != "text/event-stream":
            HTTP_LATENCY.observe(time.perf_counter() - t0, route=route)
        return resp

    def metrics_allowed() -> bool:
        """지표는 관리자 · 이 PC(127.0.0.1) · REA_METRICS_TOKEN 을 가진 수집기(Prometheus)만."""
        token = os.getenv("REA_METRICS_TOKEN", "")
        given = request.headers.get("Authorization", "")
        if token and hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
            return True
        return is_admin() or (not token and request.remote_addr in {"127.0.0.1", "::1"})

    @app.before_request
    def check_csrf():
        if request.method == "POST":
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "보안 토큰이 만료되었습니다. 페이지를 새로고침 해주세요.")

    def visitor_owner() -> str:
        """장기 기억 주인: 브라우저마다 붙는 방문자 id (쿠키, 90일). 없으면 새로 만들어 응답에 심음."""
        vid = request.cookies.get(VISITOR_COOKIE, "")
        if not _VID.match(vid):
            vid = getattr(request, "_new_vid", "") or secrets.token_urlsafe(24)
            request._new_vid = vid  # type: ignore[attr-defined]
        return "web:" + vid

    @app.after_request
    def set_visitor_cookie(resp):
        vid = getattr(request, "_new_vid", "")
        if vid:
            resp.set_cookie(VISITOR_COOKIE, vid, max_age=VISITOR_DAYS * 86400, httponly=True, samesite="Lax",
                            secure=app.config["SESSION_COOKIE_SECURE"])
        return resp

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'; form-action 'self'; base-uri 'self'",
        )
        return resp

    @app.template_filter("kjson")
    def kjson(value) -> str:
        """한글이 \\uXXXX 로 바뀌지 않는 JSON (화면 표시용, 자동 HTML 이스케이프됨)."""
        import json as _json

        return _json.dumps(value, ensure_ascii=False, default=str)

    @app.context_processor
    def inject():
        return {"csrf_token": csrf_token, "is_admin": is_admin(), "format_price": format_price,
                "upload_folders": lambda: destination_folders(settings.data_dir),
                "upload_accept": ",".join(sorted(SUPPORTED)), "default_upload_folder": default_upload_folder(settings.data_dir),
                "office": OFFICE_INFO, "sample_mode": app.config.get("SAMPLE_MODE", False)}

    def current() -> Optional[tuple[RecommendationSession, str]]:
        return store.get(session.get("sid"))

    def listing_view(s, admin: bool) -> dict:
        l = s.listing
        return {
            "id": l.listing_id, "office_no": l.office_no if admin else public_office_no(l.office_no),
            "address": l.address if admin else mask_address(l.address),
            "jimok": l.jimok or "미기재", "zoning": l.zoning or "미기재",
            "area_m2": f"{l.area_m2:,.0f}", "area_pyeong": f"{l.area_pyeong:,.0f}",
            "price": l.price_text, "price_lines": l.price_lines,
            "ppp": f"{l.price_per_pyeong:,.0f}" if l.price_per_pyeong else "",
            "kind": l.kind_label if l.property_type != "토지" else "",
            "building_pyeong": f"{l.building_area_pyeong:,.0f}" if l.building_area_pyeong else "",
            "road": l.road_access or "-", "description": l.description, "features": l.features,
            "use_tag": l.use_tag, "jimok_mix": l.extra.get("_지목구성", ""),
            "zoning_mix": l.extra.get("_용도지역구성", "") or l.extra.get("_지역지구", ""),
            "score": f"{s.score:.1f}", "reasons": s.reasons, "penalties": s.penalties,
            "registered": l.registered_date.isoformat() if l.registered_date else "-",
            # 관리자 전용
            "contact": l.owner_contact if admin else "", "source": l.source if admin else "",
            "source_file": Path(l.source_file).name if admin and l.source_file else "",
        }

    # ------------------------------------------------------------ 손님 화면
    @app.get("/")
    def index():
        with lock:
            agent.maybe_reconnect_llm()  # 꺼져 있으면 몇 분마다 자동으로 다시 연결 시도
            count = len(agent.customer_listings())
        return render_template("index.html", examples=EXAMPLES, count=count, llm=agent.llm.last_used,
                               llm_on=agent.llm.available, llm_mode=agent.llm.mode(),
                               llm_wait=agent.llm.seconds_until_retry(), llm_reason=agent.llm.off_reason(),
                               llm_cool_reason=agent.llm.cooling_reason(),
                               llm_names=agent.llm.names(), no_data=count == 0)

    @app.post("/search")
    def search():
        q = (request.form.get("q") or "").strip()[:MAX_QUERY_LEN]
        if not q:
            return redirect(url_for("index"))
        with lock:
            agent.maybe_reconnect_llm()
        req = agent.understand(q)  # LLM 호출은 잠금 밖에서 (느릴 수 있음)
        with lock:
            sess = agent.recommend(req)
        session["sid"] = store.put(sess, q)
        session["source"] = agent.llm.last_used
        return redirect(url_for("result"))

    @app.get("/result")
    def result():
        cur = current()
        if cur is None:
            return redirect(url_for("index"))
        sess, query = cur
        admin = is_admin()
        with lock:
            top = sess.current()
            others = [listing_view(s, admin) for s in sess.top(6)[1:]]
            learning = list(sess.last_learning)
            ai_changes = list(sess.ai_changes)
            reordered = sess.reordered
            sess.last_learning = []
            sess.ai_changes = []
        return render_template(
            "result.html", query=query, req=sess.req, source=session.get("source", "rule"),
            item=listing_view(top, admin) if top else None, rank=sess.cursor + 1, total=sess.total,
            relaxed=sess.relaxed, stats=sess.filter_stats, notes=sess.req.notes, others=others,
            learning=learning, reordered=reordered, quick_reasons=QUICK_REASONS,
            can_prev=bool(sess.history), pref=sess.opt.pref, listing_count=len(agent.customer_listings()),
            not_found_message=NOT_FOUND_MESSAGE, exhausted_message=EXHAUSTED_MESSAGE,
            ai_summary=getattr(sess.opt.pref, "ai_summary", ""), ai_changes=ai_changes,
            ai_on=agent.llm.available,
        )

    @app.post("/action")
    def action():
        cur = current()
        if cur is None:
            return redirect(url_for("index"))
        sess, _ = cur
        act = request.form.get("act", "")
        text = (request.form.get("text") or "").strip()[:MAX_QUERY_LEN]
        if act == "refine" and text and is_condition_change(text):
            # '좀 더 싼 걸로' -> 조건 변경 (LLM 호출은 느릴 수 있어 잠금 밖에서)
            new_req = agent.parser().apply_feedback(sess.req, text)
            with lock:
                cur_item = sess.current()
                if cur_item:
                    agent.storage.add_feedback("", cur_item.listing.listing_id, "reject", text)
                    sess.reject_current(text)
                sess.update_requirement(new_req)
                agent._save_learning(sess)
            plan = agent.ai_learn(sess)
            with lock:
                agent.apply_ai_learning(sess, plan)
            return redirect(url_for("result"))
        with lock:
            if act in {"next", "refine"}:  # '마음에 안 들어요', '너무 비싸요' -> 다음 순위 (사유 학습)
                agent.reject(sess, text, ai=False)
            elif act == "like":
                agent.like(sess, ai=False)
            elif act == "prev":
                sess.previous()
                return redirect(url_for("result"))
            else:
                return redirect(url_for("result"))
            liked = sess.current().listing.listing_id if act == "like" and sess.current() else None
        plan = agent.ai_learn(sess)  # 생성형 AI 취향 학습 (느릴 수 있어 잠금 밖에서)
        with lock:
            agent.apply_ai_learning(sess, plan)
            if liked:
                sess._bring_to_front(liked)
        return redirect(url_for("result"))

    @app.get("/api/explain")
    def api_explain():
        cur = current()
        if cur is None:
            return jsonify(text=""), 404
        sess, _ = cur
        public = not is_admin()
        with lock:
            top = sess.current()
            cache = sess.__dict__.setdefault("explain_cache", {})
            key = (top.listing.listing_id, public, sess.req.raw_text) if top else None
            cached = cache.get(key) if key else None
        if top is None:
            return jsonify(text="")
        if cached:  # '이전'으로 돌아오거나 새로고침해도 GPT 를 다시 부르지 않음
            return jsonify(text=cached[0], source=cached[1])
        text = agent.explain(sess.req, top, public=public)  # LLM 은 잠금 밖에서
        source = agent.llm.last_used
        if source != "rule":
            with lock:
                cache[key] = (text, source)
        return jsonify(text=text, source=source)

    # ------------------------------------------------------------ AI 상담 (Tool-use Agent)
    AGENT_EXAMPLES = ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "양평 전원주택 부지 3억 이하 추천해줘",
                      "공장 지을 수 있는 땅 중 넓은 거 3개 보여줘", "어느 지역 매물이 제일 많아요?"]

    def agent_card(lid: str, admin: bool, badges: list[str]) -> Optional[dict]:
        l = agent.find_listing(lid)
        if l is None:
            return None
        features = [f.lstrip("● ").strip() for f in (l.features or "").split(" / ") if f.strip()]
        from ..tool_agent.tools import tool_id

        return {"id": l.listing_id, "pid": tool_id(l), "office_no": l.office_no if admin else public_office_no(l.office_no),
                "address": l.address if admin else mask_address(l.address), "price": l.price_text,
                "price_lines": l.price_lines,
                "ppp": f"{l.price_per_pyeong:,.0f}" if l.price_per_pyeong else "",
                "pyeong": f"{l.area_pyeong:,.0f}", "m2": f"{l.area_m2:,.0f}",
                "kind": l.kind_label if l.property_type != "토지" else "",
                "building_pyeong": f"{l.building_area_pyeong:,.0f}" if l.building_area_pyeong else "",
                "jimok": l.jimok or "미기재", "zoning": l.extra.get("_지역지구") or l.zoning or "미기재",
                "features": features[:5], "badges": badges,
                "use_tag": l.use_tag, "contact": l.owner_contact if admin else ""}

    def render_turns(turns: list[dict], admin: bool, ctx=None) -> list[dict]:
        """(지금 매물 목록에 없는 매물번호는 '(지금 없는 매물)' 로 가려서 보여줌)"""
        from ..listing_numbers import hide_in, hide_unknown, known_numbers

        known = known_numbers(agent)
        turns = [{**t, "answer": hide_unknown(t.get("answer", ""), known), "steps": hide_in(t.get("steps", []), known),
                  "stale": hide_unknown(t.get("answer", ""), known) != t.get("answer", "")} for t in turns]
        return _render_turns(turns, admin, ctx)

    def _render_turns(turns: list[dict], admin: bool, ctx=None) -> list[dict]:
        """저장된 질문/답변 -> 화면용 (매물 카드는 지금 매물 정보로 그때그때 만듦: 수정 내용·관리자 여부 반영).
        v1.2: 손님이 ♡/✕ 한 매물은 카드에 표시."""
        out = []
        liked = set(getattr(ctx, "liked", []) or [])
        rejected = set((getattr(ctx, "rejected", {}) or {}).keys())
        with lock:
            for t in turns:
                cards = [c for c in (agent_card(i, admin, (t.get("badges") or {}).get(i, []))
                                     for i in t.get("listing_ids", [])) if c]
                for c in cards:
                    c["state"] = "reject" if c["id"] in rejected else ("like" if c["id"] in liked else "")
                out.append({**t, "cards": cards})
        return out

    @app.get("/agent")
    def agent_page():
        owner = visitor_owner()
        with lock:
            agent.maybe_reconnect_llm()
            # 브라우저에 대화가 없을 때만(브라우저를 껐다 켬) 이 방문자의 최근 대화를 이어서 보여줌
            conv = agent_service.open(owner, session.get("cid"), reuse_recent=not session.get("cid"))
        session["cid"] = conv["cid"]
        busy = conv["status"] == "running" and not agent_service.needs_resume(conv)
        from ..online.feedback import DOWN_REASONS

        return render_template("agent.html", turns=render_turns(conv["turns"], is_admin(), conv["ctx"]),
                               examples=AGENT_EXAMPLES, quick_reasons=QUICK_REASONS, down_reasons=DOWN_REASONS,
                               llm_mode=agent.llm.mode(), llm_names=agent.llm.names(),
                               memory=agent_service.memory_view(owner), pending=conv.get("pending"),
                               needs_resume=agent_service.needs_resume(conv), busy=busy,
                               prefill=(request.args.get("ask") or "")[:MAX_QUERY_LEN],
                               message=session.pop("agent_msg", ""), mode=session.get("agent_mode", "single"))

    def _question() -> str:
        return (request.form.get("q") or "").strip()[:MAX_QUERY_LEN]

    def _mode() -> str:
        """상담 방식: 'single'(Agent 하나) / 'multi'(v0.7 총괄 → 검색·분석 Agent). 고르면 이 브라우저에 기억."""
        from ..tool_agent.service import MODES

        chosen = request.form.get("mode")
        if chosen in MODES:
            session["agent_mode"] = chosen
        mode = session.get("agent_mode", "single")
        return mode if mode in MODES else "single"

    @app.post("/agent/ask")
    def agent_ask():
        """(자바스크립트가 꺼진 경우) 질문 처리 후 화면 다시 그리기."""
        q = _question()
        if not q:
            return redirect(url_for("agent_page"))
        try:
            cid, _ = agent_service.ask(visitor_owner(), session.get("cid"), q, mode=_mode())
            session["cid"] = cid
        except Busy as e:
            session["agent_msg"] = str(e)
        return redirect(url_for("agent_page") + "#last")

    @app.post("/agent/stream")
    def agent_stream():
        """질문 처리 진행 상황을 실시간으로 (Server-Sent Events). 화면은 단계가 끝날 때마다 바로 표시."""
        from flask import Response, stream_with_context

        owner = visitor_owner()
        resume = request.form.get("resume") == "1"
        q = _question()
        if not q and not resume:
            abort(400)
        conv = agent_service.open(owner, session.get("cid"), reuse_recent=not session.get("cid"))
        session["cid"] = conv["cid"]
        body = agent_service.stream(owner, conv["cid"], q, resume=resume, mode=_mode())
        return Response(stream_with_context(body), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ v1.2 손님 반응 (온라인 학습·평가)
    @app.post("/agent/rate")
    def agent_rate():
        """답변 👍/👎 (+ 아쉬운 이유)."""
        turn_id = (request.form.get("turn") or "")[:20]
        try:
            agent_service.rate(visitor_owner(), session.get("cid") or "", turn_id,
                               1 if request.form.get("rating") == "1" else 0, (request.form.get("reason") or "")[:60])
        except (Busy, ValueError) as e:
            session["agent_msg"] = str(e)
        return redirect(url_for("agent_page") + (f"#turn-{turn_id}" if re.fullmatch(r"[0-9a-f]{12}", turn_id) else "#last"))

    @app.post("/agent/listing-feedback")
    def agent_listing_feedback():
        """매물 카드 ♡ 관심 / ✕ 별로예요 → 기록 + 온라인 학습 (다음 검색부터 순서에 반영)."""
        action = "like" if request.form.get("action") == "like" else "reject"
        reason = (request.form.get("reason") or "")[:60]
        try:
            r = agent_service.listing_feedback(visitor_owner(), session.get("cid") or "",
                                               (request.form.get("id") or "")[:100], action, reason)
            learned = ", ".join(r.get("배운_점") or [])
            session["agent_msg"] = ("♡ 관심 매물로 기록했습니다." if action == "like" else
                                    "✕ 이 매물은 앞으로 빼고 찾습니다.") + (
                f" 배운 점: {learned} → 다음 검색부터 순서에 반영합니다." if learned else
                " 다음 검색부터 손님 취향을 반영합니다.")
        except (Busy, ValueError) as e:
            session["agent_msg"] = str(e)
        return redirect(url_for("agent_page") + "#last")

    @app.get("/admin/clean-practice")
    def admin_clean_practice():
        """연습 기록 정리: 지울 것 미리 보기."""
        from ..practice_cleanup import scan, summary

        if not is_admin():
            return login_page()
        plan = scan(agent.storage.conn, lock)
        return render_template("clean_practice.html", rows=summary(plan), protected=len(plan["protected"]),
                               sample_mode=app.config.get("SAMPLE_MODE", False),
                               nothing=not any(n for _, n in summary(plan)))

    @app.post("/admin/clean-practice")
    def admin_clean_practice_run():
        """연습 기록 정리 실행 (지우기 전 DB 백업)."""
        from ..practice_cleanup import clean

        if not is_admin():
            abort(403)
        if app.config.get("SAMPLE_MODE"):
            session["admin_msg"] = "연습용(샘플) 저장 파일은 정리할 필요가 없습니다."
            return redirect(url_for("admin"))
        result = clean(agent.storage.conn, Path(settings.db_path), lock)
        with lock:
            do_sync(detect=False)  # 매물 목록 다시 읽기 (지운 샘플 매물이 남아 보이지 않게)
        n = sum(c for _, c in result["removed"])
        session["admin_msg"] = (f"🧹 연습 기록 {n}건을 정리했습니다. 백업: {result['backup']}" if n
                                else "정리할 연습 기록이 없습니다.")
        return redirect(url_for("admin"))

    @app.get("/admin/organize-folders")
    def admin_organize_folders():
        """매물 폴더 종류별 정리: 옮길 것 미리 보기."""
        from ..folder_layout import describe, find_missing, last_log, make_plan

        if not is_admin():
            return login_page()
        plan = make_plan(settings.data_dir)
        with lock:
            missing = find_missing(agent)
        return render_template("organize_folders.html", lines=describe(plan, 200), plan=plan,
                               data_dir=Path(settings.data_dir).name, can_undo=last_log(settings.log_dir) is not None,
                               missing=missing, message=session.pop("admin_msg", ""),
                               sample_mode=app.config.get("SAMPLE_MODE", False))

    @app.post("/admin/organize-folders")
    def admin_organize_folders_run():
        """정리 실행 (op=apply) 또는 마지막 정리 되돌리기 (op=undo). 끝나면 매물 폴더를 다시 읽음."""
        from ..folder_layout import apply_plan, count_change, undo_last

        if not is_admin():
            abort(403)
        if app.config.get("SAMPLE_MODE"):
            session["admin_msg"] = "연습용(sample_data) 폴더는 이미 종류별(공장/토지)입니다."
            return redirect(url_for("admin"))
        op = request.form.get("op", "")
        with lock:
            before = {l.listing_id for l in agent.listings()}
            if op == "apply":
                r = apply_plan(settings.data_dir, settings.log_dir)
                msg = ("📂 매물 폴더를 종류별로 정리했습니다: " if not r.failed else "⚠ 일부만 정리했습니다: ") + r.summary()
            elif op == "undo":
                r = undo_last(settings.data_dir, settings.log_dir)
                msg = ("↩ 되돌렸습니다: " + r.summary()) if r.moved else "❌ " + "; ".join(r.failed)
            else:
                abort(400)
            do_sync(detect=False)  # 옮긴 매물은 번호가 같아 '새 매물' 알림이 가지 않음
            after = {l.listing_id for l in agent.listings()}
        session["admin_msg"] = f"{msg} · {count_change(before, after)}"
        # 매물이 빠졌으면 이유를 보여 주는 정리 화면으로
        return redirect(url_for("admin_organize_folders" if before - after else "admin"))

    @app.get("/admin/online")
    def admin_online():
        """v1.2 온라인 학습·평가: 만족도, '자동' 방식의 실험 현황, 배운 취향, 아쉬웠던 답."""
        from ..online.learning import model_view
        from ..storage import OFFICE_MODEL

        if not is_admin():
            return login_page()
        with lock:
            learned = agent.storage.conn.execute(
                "SELECT COUNT(*) FROM preferences WHERE name LIKE 'agent:%'").fetchone()[0]
            office = model_view(agent, OFFICE_MODEL)
        return render_template("online.html", s=agent_service.ratings.summary(7),
                               arms=agent_service.bandit.stats(), negatives=agent_service.ratings.recent_negative(10),
                               learned_visitors=learned, office_model=office)

    @app.post("/agent/resume")
    def agent_resume():
        """(자바스크립트가 꺼진 경우) 끊긴 질문 이어서 처리."""
        try:
            agent_service.resume(visitor_owner(), session.get("cid") or "", mode=_mode())
        except (Busy, ValueError) as e:
            session["agent_msg"] = str(e)
        return redirect(url_for("agent_page") + "#last")

    @app.post("/agent/reset")
    def agent_reset():
        agent_service.reset(session.pop("cid", None))
        session["cid"] = agent_service.open(visitor_owner(), None, reuse_recent=False)["cid"]
        return redirect(url_for("agent_page"))

    @app.post("/agent/forget")
    def agent_forget():
        """기억 지우기: 이 브라우저의 상담 기록과 장기 기억 삭제."""
        agent_service.forget(visitor_owner())
        session.pop("cid", None)
        session["agent_msg"] = "기억을 모두 지웠습니다. 지난 상담 내용과 관심·제외 매물 기록이 삭제되었습니다."
        return redirect(url_for("agent_page"))

    # ------------------------------------------------------------ v0.9 실시간 이벤트 (SSE)
    MAX_STREAMS = 6        # 동시에 열어 둘 실시간 연결 수 (웹 서버 스레드를 다 쓰지 않게)
    STREAM_SECONDS = 300   # 연결 하나를 이 시간 뒤 닫음 -> 브라우저가 자동으로 다시 연결 (Last-Event-ID 로 이어 받음)
    HEARTBEAT_SECONDS = 15
    streams = {"open": 0}
    streams_lock = threading.Lock()
    CUSTOMER_TOPICS = {"alert.match"}

    def sse_event(ev) -> str:
        import json as _json

        return f"id: {ev.id}\nevent: {ev.topic}\ndata: {_json.dumps(ev.public(), ensure_ascii=False, default=str)}\n\n"

    # ------------------------------------------------------------ v1.1 관측
    @app.get("/metrics")
    def metrics():
        if not metrics_allowed():
            abort(403)
        return Response(METRICS.prometheus(), mimetype="text/plain; version=0.0.4")

    @app.get("/admin/observe")
    def admin_observe():
        """관측 화면: 질문 수·성공률·응답 시간·AI 비율·토큰·안전장치 + 최근 trace 목록."""
        from ..observability import TRACER
        from ..observability.dashboard import recent, summary

        if not is_admin():
            return login_page()
        TRACER.flush(1)
        try:
            hours = min(24 * 7, max(1.0, float(request.args.get("hours", "24"))))
        except ValueError:
            hours = 24.0
        show = request.args.get("show", "")
        return render_template("observe.html", s=summary(agent.storage.conn, lock, hours), hours=hours, show=show,
                               traces=recent(agent.storage.conn, lock, 40, only_errors=show == "errors",
                                             slow_ms=2000 if show == "slow" else 0),
                               otlp=os.getenv("REA_OTLP_ENDPOINT", ""))

    @app.get("/admin/traces/<trace_id>")
    def admin_trace(trace_id: str):
        """trace 하나: 어느 Agent·AI·도구·원격 서버에서 얼마나 걸렸는지 폭포 그림."""
        from ..observability import TRACER
        from ..observability.dashboard import trace_tree

        if not is_admin():
            return login_page()
        if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
            abort(404)
        TRACER.flush(1)
        rows = trace_tree(agent.storage.conn, lock, trace_id)
        if not rows:
            abort(404)
        return render_template("trace.html", rows=rows, trace_id=trace_id,
                               total_ms=max(r["offset_ms"] + r["dur_ms"] for r in rows),
                               services=sorted({r["service"] for r in rows}))

    # ------------------------------------------------------------ v1.0 상태 점검 (/healthz 는 아래: 살아 있는지만)
    @app.get("/readyz")
    def readyz():
        """손님을 받을 준비가 됐는지: DB · 매물 · AI · 실시간 · 원격 Agent · 동시 상담 자리. 실패면 503."""
        from ..reliability.health import http_status, readiness

        report = readiness(agent, lock, app.config.get("REALTIME"), agent_service)
        return report, http_status(report)

    @app.get("/events/stream")
    def events_stream():
        """실시간 알림: 손님은 자기 맞춤 알림(찾던 조건의 새 매물·가격 인하)만, 관리자는 모든 이벤트."""
        from flask import Response, stream_with_context

        hub = app.config["REALTIME"]
        owner, admin = visitor_owner(), is_admin()
        raw = request.headers.get("Last-Event-ID") or request.args.get("since") or ""
        last_id = int(raw) if raw.isdigit() else hub.bus.last_id()
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        with streams_lock:
            if streams["open"] >= MAX_STREAMS:  # 너무 많으면 1분 뒤 다시 연결하라고만 하고 닫음
                return Response("retry: 60000\n\n", mimetype="text/event-stream", headers=headers)
            streams["open"] += 1

        def visible(ev) -> bool:
            return admin or ev.topic in CUSTOMER_TOPICS

        def gen():
            import queue as _queue

            q = hub.bus.subscribe(owner, see_all=admin)  # 먼저 구독한 뒤 놓친 기록을 보내 사이에 빠지는 이벤트가 없게
            sent = last_id
            try:
                yield "retry: 5000\n\n"
                for ev in hub.bus.since(last_id, owner, see_all=admin):
                    sent = ev.id
                    if visible(ev):
                        yield sse_event(ev)
                deadline = time.time() + STREAM_SECONDS
                while time.time() < deadline:
                    try:
                        ev = q.get(timeout=HEARTBEAT_SECONDS)
                    except _queue.Empty:
                        yield ": ping\n\n"
                        continue
                    if ev.id > sent and visible(ev):
                        sent = ev.id
                        yield sse_event(ev)
            finally:
                hub.bus.unsubscribe(q)
                with streams_lock:
                    streams["open"] -= 1

        return Response(stream_with_context(gen()), mimetype="text/event-stream", headers=headers)

    def realtime_view() -> dict:
        hub = app.config["REALTIME"]
        return {"running": hub.running, "interval": hub.interval, "threshold": hub.threshold,
                "subscribers": hub.bus.subscribers, "streams": streams["open"], "stats": hub.stats.as_dict(),
                "counts": hub.bus.counts(),
                "events": [{**e.public(), "owner": bool(e.owner)} for e in hub.bus.recent(15, see_all=True)]}

    # ------------------------------------------------------------ 관리자 화면
    auth = AdminAuth(auth_file(settings))
    app.config["ADMIN_AUTH"] = auth  # 테스트용

    def is_local_request() -> bool:
        """서버를 켠 PC 에서 직접 접속했는지 (처음 비밀번호 정하기는 이 경우에만 허용)."""
        if request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded"):
            return False  # 프록시(인터넷 공개) 뒤에서 들어온 요청
        return request.remote_addr in {"127.0.0.1", "::1"}

    def login_page(status: int = 200, **kw):
        return render_template("login.html", configured=auth.configured, local=is_local_request(),
                               min_len=MIN_PASSWORD_LEN, **kw), status

    def start_admin_session() -> None:
        """로그인 성공: 세션 고정 공격을 막도록 쿠키 내용을 새로 만들고 (추천 상담은 이어지게) 관리자 표시."""
        sid = session.get("sid")
        source = session.get("source")
        cid = session.get("cid")
        session.clear()
        if cid:
            session["cid"] = cid
        if sid:
            session["sid"] = sid
        if source:
            session["source"] = source
        session["admin"] = True
        session["csrf"] = secrets.token_urlsafe(24)

    @app.get("/admin")
    def admin():
        if not is_admin():
            return login_page()
        with lock:
            report = agent.last_sync
            data = {
                "count": len(agent.listings()),
                "unnumbered": sum(1 for l in agent.listings() if not l.office_no),
                "kinds": " · ".join(f"{k} {n}건" for k, n in sorted(
                    Counter(l.property_type for l in agent.listings()).items(), key=lambda kv: kv[0] != "토지")),
                "report": report.clean.report() if report else "",
                "summary": report.summary() if report else "",
                "synced_at": agent.last_sync_at.strftime("%Y-%m-%d %H:%M:%S") if agent.last_sync_at else "-",
                "office": agent.office_model.describe(),
            }
        from ..reliability.health import readiness

        from ..folder_layout import make_plan

        layout_moves = 0 if app.config.get("SAMPLE_MODE") else len(make_plan(settings.data_dir).moves)
        return render_template("admin.html", data=data, data_dir=str(settings.data_dir), realtime=realtime_view(),
                               layout_moves=layout_moves,
                               ready=readiness(agent, lock, app.config.get("REALTIME"), agent_service),
                               llm_status=agent.llm.status(), sync_error=state["sync_error"],
                               message=session.pop("admin_msg", ""), pw_source=auth.source,
                               min_len=MIN_PASSWORD_LEN)

    @app.post("/admin/login")
    def admin_login():
        who = request.remote_addr or "?"
        wait = auth.locked_for(who)
        if wait:
            return login_page(429, error=f"비밀번호를 여러 번 틀려 잠시 잠겼습니다. 약 {wait // 60 + 1}분 뒤 다시 시도하세요.")
        if not auth.configured:
            return login_page(400)
        pw = request.form.get("password", "")
        if auth.check(pw):
            auth.clear_failures(who)
            start_admin_session()
            return redirect(url_for("admin"))
        auth.record_failure(who)
        time.sleep(1)  # 무차별 대입 지연
        return login_page(401, error="비밀번호가 맞지 않습니다.")

    @app.post("/admin/setup")
    def admin_setup():
        """처음 한 번: 관리자 비밀번호 정하기 (비밀번호가 없고, 서버를 켠 PC 에서 접속했을 때만)."""
        if auth.configured:
            return redirect(url_for("admin"))
        if not is_local_request():
            abort(403)
        try:
            auth.set_password(request.form.get("password", ""), request.form.get("confirm", ""))
        except (ValueError, OSError) as e:
            return login_page(400, error=str(e))
        start_admin_session()
        session["admin_msg"] = "관리자 비밀번호를 정했습니다. 다음부터는 이 비밀번호로 로그인하세요."
        return redirect(url_for("admin"))

    @app.post("/admin/password")
    def admin_password_change():
        if not is_admin():
            abort(403)
        if not auth.check(request.form.get("current", "")):
            time.sleep(1)
            session["admin_msg"] = "❌ 지금 비밀번호가 맞지 않습니다."
            return redirect(url_for("admin"))
        try:
            auth.set_password(request.form.get("password", ""), request.form.get("confirm", ""))
            session["admin_msg"] = "✅ 관리자 비밀번호를 바꿨습니다."
        except (ValueError, OSError) as e:
            session["admin_msg"] = f"❌ 비밀번호를 바꾸지 못했습니다: {e}"
        return redirect(url_for("admin"))

    @app.post("/admin/logout")
    def admin_logout():
        session.pop("admin", None)
        return redirect(url_for("index"))

    @app.post("/admin/sync")
    def admin_sync():
        if not is_admin():
            abort(403)
        with lock:
            do_sync()
        session["admin_msg"] = state["sync_error"] or "매물 폴더를 다시 읽었습니다."
        return redirect(url_for("admin"))

    @app.post("/admin/open-folder")
    def admin_open_folder():
        """서버를 켠 PC 에서 매물 폴더를 탐색기로 열기 (Windows)."""
        if not is_admin():
            abort(403)
        folder = settings.data_dir
        folder.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(folder)  # type: ignore[attr-defined]
            session["admin_msg"] = "매물 폴더를 열었습니다. 엑셀 파일을 넣은 뒤 '매물 폴더 다시 읽기'를 누르세요."
        else:
            session["admin_msg"] = f"이 컴퓨터에서는 폴더를 자동으로 열 수 없습니다. 위치: {folder}"
        return redirect(url_for("admin"))

    @app.post("/admin/ai-key")
    def admin_ai_key():
        """GPT/Claude API 키 저장(my_api_key.py) -> 바로 다시 연결 -> 실제로 한 번 물어봐서 확인."""
        if not is_admin():
            abort(403)
        from ..setup_tools import save_key_to_code_file

        key = request.form.get("key", "")
        model = request.form.get("model", "")
        anthropic_key = request.form.get("anthropic_key", "")
        if key.strip() or anthropic_key.strip():
            try:
                save_key_to_code_file(key, model, anthropic_key)
            except (ValueError, OSError) as e:
                session["admin_msg"] = f"저장하지 못했습니다: {e}"
                return redirect(url_for("admin"))
        with lock:
            agent.reconnect_llm()
        results = agent.test_llm_each()
        if not results:
            session["admin_msg"] = "❌ AI 연결 실패 - " + agent.llm.off_reason()
        else:
            session["admin_msg"] = " / ".join(("✅ " if ok else "❌ ") + detail for ok, detail in results)
        return redirect(url_for("admin"))

    # ------------------------------------------------------------ 관리자: 매물 추가·수정
    def admin_row(l, hidden: bool = False) -> dict:
        return {
            "id": l.listing_id, "office_no": l.office_no, "address": l.address, "jimok": l.jimok or "-",
            "pyeong": f"{l.area_pyeong:,.0f}", "price": l.price_text, "status": l.status, "kind": l.kind_label,
            "edited": bool(l.extra.get("관리자수정")), "hidden": hidden, "web": l.source == "web"
            or Path(l.source_file).parent.name == WEB_FOLDER,
            "file": Path(l.source_file).name if l.source_file else "-",
        }

    def form_page(mode: str, values: dict, listing_id: str = "", errors=(), status: int = 200):
        with lock:
            edit = agent.listing_edit(listing_id) if listing_id else {}
            original = agent.original_listing(listing_id) if listing_id else None
            changes = describe_changes(original, edit) if original and edit else []
            source_file = Path(original.source_file).name if original and original.source_file else ""
            folder = listing_folder(settings.data_dir, original.source_file) if original else ""
            office_no = original.office_no if original else ""
        return render_template("listing_form.html", mode=mode, v=values, listing_id=listing_id, errors=list(errors),
                               jimok_choices=JIMOK_TYPES, zoning_choices=ZONING_TYPES,
                               kind_choices=PROPERTY_TYPES, deal_choices=DEAL_TYPES, statuses=LISTING_STATUSES,
                               changes=changes, hidden=bool(edit.get(HIDDEN)), source_file=source_file,
                               folder=folder, office_no=office_no, folder_files=files_in(settings.data_dir, folder) if listing_id else [],
                               web_file=web_file_label()), status

    def web_file_label() -> str:
        """'매물/토지/웹입력/웹입력매물.xlsx (공장은 매물/공장/웹입력/…)'"""
        from ..folder_layout import is_organized

        land = Path(settings.data_dir.name) / web_file(settings.data_dir).relative_to(settings.data_dir)
        if not is_organized(settings.data_dir):
            return land.as_posix()
        factory = Path(settings.data_dir.name) / web_file(settings.data_dir, "공장").relative_to(settings.data_dir)
        return f"{land.as_posix()} (공장은 {factory.as_posix()})"

    @app.get("/admin/listings")
    def admin_listings():
        if not is_admin():
            return redirect(url_for("admin"))
        q = (request.args.get("q") or "").strip()[:MAX_QUERY_LEN]
        show = request.args.get("show", "")
        try:
            page = max(1, int(request.args.get("page", "1")))
        except ValueError:
            page = 1
        with lock:
            rows = [admin_row(l) for l in agent.listings()] + [admin_row(l, True) for l in agent.hidden_listings]
        if show == "edited":
            rows = [r for r in rows if r["edited"] or r["hidden"]]
        elif show == "hidden":
            rows = [r for r in rows if r["hidden"]]
        elif show == "web":
            rows = [r for r in rows if r["web"]]
        if q:
            words = q.split()
            rows = [r for r in rows if all(w in f"{r['id']} {r['office_no']} {r['address']} {r['jimok']} {r['status']}"
                                       for w in words)]
        rows.sort(key=lambda r: (r["hidden"], r["address"]))
        per = 50
        pages = max(1, (len(rows) + per - 1) // per)
        page = min(page, pages)
        return render_template("admin_listings.html", rows=rows[(page - 1) * per: page * per], total=len(rows),
                               q=q, show=show, page=page, pages=pages,
                               message=session.pop("admin_msg", ""))

    @app.route("/admin/listings/new", methods=["GET", "POST"])
    def admin_listing_new():
        if not is_admin():
            return redirect(url_for("admin"))
        if request.method == "GET":
            return form_page("new", form_defaults())
        form = parse_form(request.form)
        if form.errors:
            return form_page("new", form.raw, errors=form.errors, status=400)
        try:
            with lock:
                listing_id, warnings = agent.add_listing(form.values)
                changed("관리자 화면")
        except (ValueError, OSError) as e:
            return form_page("new", form.raw, errors=[str(e)], status=400)
        session["admin_msg"] = f"✅ 새 매물을 추가했습니다 (매물번호 {listing_id}). 손님 검색에 바로 나옵니다." + (
            f" 참고: {', '.join(warnings)}" if warnings else "")
        return redirect(url_for("admin_listings", q=listing_id))

    @app.route("/admin/listings/edit", methods=["GET", "POST"])
    def admin_listing_edit():
        if not is_admin():
            return redirect(url_for("admin"))
        listing_id = (request.values.get("id") or "").strip()[:100]
        with lock:
            listing = agent.find_listing(listing_id)
        if listing is None:
            session["admin_msg"] = "❌ 매물을 찾을 수 없습니다. (매물 폴더에서 지워졌을 수 있습니다)"
            return redirect(url_for("admin_listings"))
        if request.method == "GET":
            return form_page("edit", form_defaults(listing), listing_id)
        form = parse_form(request.form)
        if form.errors:
            return form_page("edit", form.raw, listing_id, errors=form.errors, status=400)
        try:
            with lock:
                warnings = agent.edit_listing(listing_id, form.values)
                changed("관리자 화면")
        except ValueError as e:
            return form_page("edit", form.raw, listing_id, errors=[str(e)], status=400)
        session["admin_msg"] = f"✅ 매물 {listing_id} 을(를) 수정했습니다. 손님 검색에 바로 반영됩니다." + (
            f" 참고: {', '.join(warnings)}" if warnings else "")
        return redirect(url_for("admin_listings", q=listing_id))

    @app.post("/admin/listings/state")
    def admin_listing_state():
        """숨기기 / 다시 보이기 / 원래대로 되돌리기."""
        if not is_admin():
            abort(403)
        listing_id = (request.form.get("id") or "").strip()[:100]
        op = request.form.get("op", "")
        try:
            with lock:
                if op == "hide":
                    agent.edit_listing(listing_id, {}, hidden=True)
                    msg = f"매물 {listing_id} 을(를) 숨겼습니다. 손님 검색에 나오지 않습니다. ('숨긴 매물'에서 다시 보이게 할 수 있습니다)"
                elif op == "show":
                    agent.edit_listing(listing_id, {}, hidden=False)
                    msg = f"매물 {listing_id} 을(를) 다시 손님 검색에 보이게 했습니다."
                elif op == "revert":
                    agent.revert_listing(listing_id)
                    msg = f"매물 {listing_id} 을(를) 원래 엑셀 내용으로 되돌렸습니다."
                else:
                    abort(400)
                changed("관리자 화면")
            session["admin_msg"] = "✅ " + msg
        except ValueError as e:
            session["admin_msg"] = f"❌ {e}"
        return redirect(url_for("admin_listings", q=listing_id))

    @app.get("/admin/template")
    def admin_template():
        if not is_admin():
            abort(403)
        from ..template import build_template

        return send_file(io.BytesIO(build_template()), as_attachment=True, download_name="토지매물_입력양식.xlsx",
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.post("/admin/upload")
    def admin_upload():
        """매물 파일 여러 개 또는 폴더째 올리기 -> 매물 폴더에 저장 후 다시 읽기."""
        if not is_admin():
            abort(403)
        back = request.form.get("back", "admin")
        listing_id = (request.form.get("id") or "").strip()[:100]

        def go_back():
            if back == "edit" and listing_id:
                return redirect(url_for("admin_listing_edit", id=listing_id))
            return redirect(url_for("admin_listings" if back == "listings" else "admin"))

        folder_files = request.files.getlist("folder")
        paths = request.form.getlist("folder_paths")
        if len(paths) == len(folder_files):  # 화면이 보낸 '폴더/하위/파일' 경로가 더 정확함
            for f, p in zip(folder_files, paths):
                if "/" in p and "/" not in (f.filename or "").replace("\\", "/"):
                    f.filename = p
        files = request.files.getlist("files") + folder_files + request.files.getlist("file")
        dest = request.form.get("dest", default_upload_folder(settings.data_dir))
        if dest == "__new__":
            dest = (request.form.get("new_dest") or "").strip()
            if not dest:
                session["admin_msg"] = "❌ 새로 만들 폴더 이름을 적어 주세요."
                return go_back()
        if back == "edit" and listing_id:  # 이 매물의 폴더에 올리기
            with lock:
                listing = agent.find_listing(listing_id)
            if listing is None:
                session["admin_msg"] = "❌ 매물을 찾을 수 없습니다."
                return redirect(url_for("admin_listings"))
            dest = listing_folder(settings.data_dir, listing.source_file)
        try:
            with lock:
                before = {l.listing_id for l in agent.listings() + agent.hidden_listings}
                result = save_uploads(settings.data_dir, files, dest, replace=request.form.get("replace") == "1")
                do_sync()
                after = agent.listings() + agent.hidden_listings
                saved = {str((settings.data_dir / p).resolve()) for p in result.saved}
                added = [l for l in after if l.listing_id not in before]
                from_files = [l for l in after if str(Path(l.source_file).resolve()) in saved]
                problems = [e for e in (agent.last_sync.file_errors if agent.last_sync else [])
                            if any(Path(p).name in e or Path(p).parent.name in e for p in result.saved)]
        except (ValueError, OSError) as e:
            session["admin_msg"] = f"❌ 올리지 못했습니다: {e}"
            return go_back()
        where = f"'{dest}'" if dest else "매물 폴더 바로 아래"
        msg = f"✅ {where} 에 {result.summary()}. 다시 읽은 결과: 새 매물 {len(added)}건, 올린 파일에서 읽은 매물 {len(from_files)}건."
        if not from_files:
            msg += " ⚠ 올린 파일에서 매물을 읽지 못했습니다. 소재지·면적·매매가 항목이 있는지 확인하세요."
        if problems:
            msg += " 확인 필요: " + "; ".join(problems[:3])
        session["admin_msg"] = msg
        if back != "edit" and len(added) == 1:
            return redirect(url_for("admin_listings", q=added[0].listing_id))
        return go_back()

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def error_page(e):
        msg = {413: f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_MB}MB).", 403: "관리자 로그인이 필요합니다.",
               404: "페이지를 찾을 수 없습니다."}.get(getattr(e, "code", 400), getattr(e, "description", "잘못된 요청"))
        return render_template("error.html", message=msg), getattr(e, "code", 400)

    return app


def run(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True, settings: Optional[Settings] = None):
    """웹 서비스 실행 (Windows 에서도 동작하는 waitress 서버 사용)."""
    from waitress import serve

    app = create_app(settings)
    url = f"http://{'localhost' if host in {'127.0.0.1', '0.0.0.0'} else host}:{port}"
    print(f"\n🌐 웹 서비스 시작: {url}   (종료: Ctrl+C)")
    llm = app.config["AGENT"].llm
    if llm.mode() == "off":
        print(f"   ⚠ AI(GPT) 미연결: {llm.off_reason()}")
        print("     → 프로젝트 폴더의 my_api_key.py 를 열어 OPENAI_API_KEY = \"sk-...\" 에 키를 붙여넣고 저장하세요.")
    else:
        print(f"   🤖 {llm.status()}")
    if host == "0.0.0.0":
        print("   같은 와이파이의 다른 기기에서는 http://<이 PC의 IP 주소>:%d 로 접속하세요." % port)
    if not os.getenv("REA_ADMIN_PASSWORD"):
        print("   ※ 관리자 화면을 쓰려면 .env 에 REA_ADMIN_PASSWORD=비밀번호 를 넣으세요.")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    hub = app.config["REALTIME"].start()  # 매물 폴더 감시 + 매칭 워커 시작
    print(f"   실시간: 매물 폴더를 {hub.interval:g}초마다 확인, 새 매물·가격 인하를 찾던 손님에게 바로 알림")
    serve(app, host=host, port=port, threads=16)


__all__ = ["create_app", "run", "PROJECT_ROOT"]
