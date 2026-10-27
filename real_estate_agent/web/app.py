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
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from ..admin_auth import MIN_LENGTH as MIN_PASSWORD_LEN
from ..admin_auth import AdminAuth, auth_file
from ..agent import RealEstateAgent, is_condition_change, mask_address, public_office_no
from ..collectors.local_folder import SUPPORTED
from ..file_uploads import DEFAULT_FOLDER as DEFAULT_UPLOAD_FOLDER
from ..file_uploads import destination_folders, files_in, listing_folder, save_uploads
from ..listing_editor import HIDDEN, STATUSES as LISTING_STATUSES, WEB_FOLDER
from ..listing_editor import describe_changes, form_defaults, parse_form, web_file
from ..models import JIMOK_TYPES
from ..requirement_parser import ZONING_TYPES
from ..config import PROJECT_ROOT, Settings, get_settings
from ..models import format_price
from ..recommender import EXHAUSTED_MESSAGE, NOT_FOUND_MESSAGE, RecommendationSession

log = logging.getLogger(__name__)

MAX_QUERY_LEN = 300
MAX_UPLOAD_MB = 200  # 폴더째 올릴 때 전체 크기 (파일 하나는 20MB 까지)
SESSION_TTL_SEC = 2 * 60 * 60
MAX_SESSIONS = 500
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


class AgentConversations:
    """접속자별 AI 상담 대화 (대화 기록 + 도구 상태 + 화면 표시용 기록). 2시간 사용 없으면 삭제."""

    MAX_TURNS = 20

    def __init__(self, ttl: int = SESSION_TTL_SEC, max_size: int = MAX_SESSIONS):
        self.ttl, self.max_size = ttl, max_size
        self._data: OrderedDict[str, dict] = OrderedDict()

    def get(self, cid: Optional[str]) -> tuple[str, dict]:
        now = time.time()
        for k in [k for k, v in self._data.items() if now - v["ts"] > self.ttl]:
            del self._data[k]
        if not cid or cid not in self._data:
            while len(self._data) >= self.max_size:
                self._data.popitem(last=False)
            from ..tool_agent import AgentContext

            cid = secrets.token_urlsafe(16)
            self._data[cid] = {"history": [], "ctx": AgentContext(), "turns": [], "ts": now}
        conv = self._data[cid]
        conv["ts"] = now
        self._data.move_to_end(cid)
        return cid, conv

    def reset(self, cid: Optional[str]) -> None:
        self._data.pop(cid or "", None)


def safe_filename(name: str) -> str:
    """업로드 파일 이름 정리 (한글 유지, 경로·특수문자 제거)."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return name or "upload"


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
    conversations = AgentConversations()
    state = {"sync_error": ""}

    def do_sync() -> None:
        try:
            agent.sync()
            state["sync_error"] = ""
        except FileNotFoundError as e:
            state["sync_error"] = str(e)
            log.warning("매물 폴더 읽기 실패 - 저장된 매물로 계속: %s", e)

    with lock:
        do_sync()

    app.config["AGENT"] = agent  # 테스트용

    # ------------------------------------------------------------ 공통
    def is_admin() -> bool:
        return bool(session.get("admin"))

    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(24)
        return session["csrf"]

    @app.before_request
    def check_csrf():
        if request.method == "POST":
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "보안 토큰이 만료되었습니다. 페이지를 새로고침 해주세요.")

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
                "upload_accept": ",".join(sorted(SUPPORTED)), "default_upload_folder": DEFAULT_UPLOAD_FOLDER}

    def current() -> Optional[tuple[RecommendationSession, str]]:
        return store.get(session.get("sid"))

    def listing_view(s, admin: bool) -> dict:
        l = s.listing
        return {
            "id": l.listing_id, "office_no": l.office_no if admin else public_office_no(l.office_no),
            "address": l.address if admin else mask_address(l.address),
            "jimok": l.jimok or "-", "zoning": l.zoning or "-",
            "area_m2": f"{l.area_m2:,.0f}", "area_pyeong": f"{l.area_pyeong:,.0f}",
            "price": format_price(l.price_manwon), "ppp": f"{l.price_per_pyeong:,.0f}",
            "road": l.road_access or "-", "description": l.description, "features": l.features,
            "use_tag": l.use_tag, "jimok_mix": l.extra.get("_지목구성", ""),
            "zoning_mix": l.extra.get("_용도지역구성", ""),
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
            count = len(agent.listings())
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
            can_prev=bool(sess.history), pref=sess.opt.pref, listing_count=len(agent.listings()),
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
                      "E-3004 매물에 공장도 지을 수 있어?", "어느 지역 매물이 제일 많아요?"]

    def agent_card(lid: str, admin: bool, badges: list[str]) -> Optional[dict]:
        l = agent.find_listing(lid)
        if l is None:
            return None
        features = [f.lstrip("● ").strip() for f in (l.features or "").split(" / ") if f.strip()]
        return {"id": l.listing_id, "office_no": l.office_no if admin else public_office_no(l.office_no),
                "address": l.address if admin else mask_address(l.address), "price": format_price(l.price_manwon),
                "ppp": f"{l.price_per_pyeong:,.0f}", "pyeong": f"{l.area_pyeong:,.0f}", "m2": f"{l.area_m2:,.0f}",
                "jimok": l.jimok or "-", "zoning": l.zoning or "-", "features": features[:5], "badges": badges,
                "use_tag": l.use_tag, "contact": l.owner_contact if admin else ""}

    @app.get("/agent")
    def agent_page():
        cid, conv = conversations.get(session.get("cid"))
        session["cid"] = cid
        with lock:
            agent.maybe_reconnect_llm()
        return render_template("agent.html", turns=conv["turns"], examples=AGENT_EXAMPLES,
                               llm_mode=agent.llm.mode(), llm_names=agent.llm.names())

    @app.post("/agent/ask")
    def agent_ask():
        from ..tool_agent import ToolAgent
        from ..tool_agent.loop import step_dict

        q = (request.form.get("q") or "").strip()[:MAX_QUERY_LEN]
        if not q:
            return redirect(url_for("agent_page"))
        cid, conv = conversations.get(session.get("cid"))
        session["cid"] = cid
        # LLM 호출은 잠금 밖에서, 도구 실행(매물 목록·DB 접근)만 잠금 안에서
        runner = ToolAgent(agent, ctx=conv["ctx"], lock=lock)
        result, conv["history"] = runner.run(q, conv["history"])
        admin = is_admin()
        with lock:
            cards = [c for c in (agent_card(i, admin, result.badges.get(i, [])) for i in result.listing_ids) if c]
        conv["turns"].append({"question": q, "answer": result.answer, "steps": [step_dict(s) for s in result.steps],
                              "cards": cards, "policy": result.policy, "total_ms": result.total_ms,
                              "tool_calls": result.tool_calls, "tool_errors": result.tool_errors})
        conv["turns"] = conv["turns"][-AgentConversations.MAX_TURNS:]
        return redirect(url_for("agent_page") + "#last")

    @app.post("/agent/reset")
    def agent_reset():
        conversations.reset(session.pop("cid", None))
        return redirect(url_for("agent_page"))

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
                "report": report.clean.report() if report else "",
                "summary": report.summary() if report else "",
                "synced_at": agent.last_sync_at.strftime("%Y-%m-%d %H:%M:%S") if agent.last_sync_at else "-",
                "office": agent.office_model.describe(),
            }
        return render_template("admin.html", data=data, data_dir=str(settings.data_dir),
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
            "pyeong": f"{l.area_pyeong:,.0f}", "price": format_price(l.price_manwon), "status": l.status,
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
                               jimok_choices=JIMOK_TYPES, zoning_choices=ZONING_TYPES, statuses=LISTING_STATUSES,
                               changes=changes, hidden=bool(edit.get(HIDDEN)), source_file=source_file,
                               folder=folder, office_no=office_no, folder_files=files_in(settings.data_dir, folder) if listing_id else [],
                               web_file=str(Path(settings.data_dir.name) / WEB_FOLDER / web_file(settings.data_dir).name)), status

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
        dest = request.form.get("dest", DEFAULT_UPLOAD_FOLDER)
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
    serve(app, host=host, port=port, threads=8)


__all__ = ["create_app", "run", "PROJECT_ROOT"]
