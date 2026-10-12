"""Real Estate Recommendation Agent - 웹 서비스 (Flask).

손님 화면
  /                검색창 ("음성군 채광 좋은 2억원대 땅")
  /result          추천 매물 카드 + [마음에 안 들어요 → 다음 순위] / 빠른 사유 / 관심 / 이전 / 조건 변경
  /api/explain     현재 매물 AI 설명 (화면이 먼저 뜬 뒤 불러옴)
관리자 화면 (비밀번호: .env 의 REA_ADMIN_PASSWORD)
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

from ..agent import RealEstateAgent, is_condition_change, mask_address
from ..collectors.local_folder import SUPPORTED
from ..config import PROJECT_ROOT, Settings, get_settings
from ..models import format_price
from ..recommender import RecommendationSession

log = logging.getLogger(__name__)

MAX_QUERY_LEN = 300
MAX_UPLOAD_MB = 20
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

    @app.context_processor
    def inject():
        return {"csrf_token": csrf_token, "is_admin": is_admin(), "format_price": format_price}

    def current() -> Optional[tuple[RecommendationSession, str]]:
        return store.get(session.get("sid"))

    def listing_view(s, admin: bool) -> dict:
        l = s.listing
        return {
            "id": l.listing_id,
            "address": l.address if admin else mask_address(l.address),
            "jimok": l.jimok or "-", "zoning": l.zoning or "-",
            "area_m2": f"{l.area_m2:,.0f}", "area_pyeong": f"{l.area_pyeong:,.0f}",
            "price": format_price(l.price_manwon), "ppp": f"{l.price_per_pyeong:,.0f}",
            "road": l.road_access or "-", "description": l.description,
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
            count = len(agent.listings())
        return render_template("index.html", examples=EXAMPLES, count=count, llm=agent.llm.last_used,
                               llm_on=agent.llm.available, no_data=count == 0)

    @app.post("/search")
    def search():
        q = (request.form.get("q") or "").strip()[:MAX_QUERY_LEN]
        if not q:
            return redirect(url_for("index"))
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
            reordered = sess.reordered
            sess.last_learning = []
        return render_template(
            "result.html", query=query, req=sess.req, source=session.get("source", "rule"),
            item=listing_view(top, admin) if top else None, rank=sess.cursor + 1, total=sess.total,
            relaxed=sess.relaxed, stats=sess.filter_stats, notes=sess.req.notes, others=others,
            learning=learning, reordered=reordered, quick_reasons=QUICK_REASONS,
            can_prev=bool(sess.history), pref=sess.opt.pref, listing_count=len(agent.listings()),
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
            return redirect(url_for("result"))
        with lock:
            if act in {"next", "refine"}:  # '마음에 안 들어요', '너무 비싸요' -> 다음 순위 (사유 학습)
                agent.reject(sess, text)
            elif act == "like":
                agent.like(sess)
            elif act == "prev":
                sess.previous()
        return redirect(url_for("result"))

    @app.get("/api/explain")
    def api_explain():
        cur = current()
        if cur is None:
            return jsonify(text=""), 404
        sess, _ = cur
        with lock:
            top = sess.current()
        if top is None:
            return jsonify(text="")
        text = agent.explain(sess.req, top, public=not is_admin())  # LLM 은 잠금 밖에서
        return jsonify(text=text, source=agent.llm.last_used)

    # ------------------------------------------------------------ 관리자 화면
    admin_password = os.getenv("REA_ADMIN_PASSWORD", "")

    @app.get("/admin")
    def admin():
        if not is_admin():
            return render_template("login.html", configured=bool(admin_password))
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
                               message=session.pop("admin_msg", ""))

    @app.post("/admin/login")
    def admin_login():
        pw = request.form.get("password", "")
        if admin_password and hmac.compare_digest(pw.encode(), admin_password.encode()):
            session["admin"] = True
            return redirect(url_for("admin"))
        time.sleep(1)  # 무차별 대입 지연
        return render_template("login.html", configured=bool(admin_password), error=True), 401

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

    @app.get("/admin/template")
    def admin_template():
        if not is_admin():
            abort(403)
        from ..template import build_template

        return send_file(io.BytesIO(build_template()), as_attachment=True, download_name="토지매물_입력양식.xlsx",
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.post("/admin/upload")
    def admin_upload():
        if not is_admin():
            abort(403)
        f = request.files.get("file")
        if not f or not f.filename:
            session["admin_msg"] = "파일을 선택해 주세요."
            return redirect(url_for("admin"))
        name = safe_filename(f.filename)
        if Path(name).suffix.lower() not in SUPPORTED:
            session["admin_msg"] = f"지원하지 않는 형식입니다. ({', '.join(sorted(SUPPORTED))})"
            return redirect(url_for("admin"))
        target_dir = settings.data_dir / "웹업로드"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
            f.save(target)
        except OSError as e:
            session["admin_msg"] = f"저장 실패: {e}"
            return redirect(url_for("admin"))
        with lock:
            do_sync()
        session["admin_msg"] = f"'{name}' 업로드 후 매물을 다시 읽었습니다."
        return redirect(url_for("admin"))

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
    if host == "0.0.0.0":
        print("   같은 와이파이의 다른 기기에서는 http://<이 PC의 IP 주소>:%d 로 접속하세요." % port)
    if not os.getenv("REA_ADMIN_PASSWORD"):
        print("   ※ 관리자 화면을 쓰려면 .env 에 REA_ADMIN_PASSWORD=비밀번호 를 넣으세요.")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, host=host, port=port, threads=8)


__all__ = ["create_app", "run", "PROJECT_ROOT"]
