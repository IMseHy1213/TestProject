"""연습 기록 정리: 실제 운영 DB(data/agent.db)에 섞여 들어간 '연습용 샘플 매물' 기록만 지움.

예전에는 VS Code '샘플 데이터로 연습'이 실제 운영과 같은 DB 를 써서, 가짜 매물(sample_data 폴더의 E-…, T-… 등)과
그 매물로 나눈 상담·알림·추적 기록이 실제 DB 에 남았습니다. (지금은 연습용 DB 가 data/sample.db 로 따로 있음)

지우는 것 (연습용 샘플 매물이 나오는 기록만)
  - 매물 목록의 샘플 매물 (매물 출처 파일이 sample_data 폴더)과 그 매물의 관리자 수정 기록
  - 샘플 매물이 나온 AI 상담 대화와 그 답변 평가(👍/👎), 추적(trace) 기록
  - 샘플 매물에 대한 실시간 알림·이벤트, 예전 추천 화면의 피드백·선제 알림
  - 방문자 장기 기억 속 샘플 매물 (관심·제외 목록에서만 뺌)
지우지 않는 것
  - 실제 매물(매물 폴더), 손님 목록, 관리자 비밀번호, 설정, 실제 매물이 나온 상담
  - 같은 번호가 실제 매물에도 있으면 그 번호는 건드리지 않음
지우기 전에 DB 전체를 data/backup/ 에 복사해 둡니다 (되돌리기 가능).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import threading
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .listing_numbers import NUMBER

TABLES = ("listings", "listing_edits", "feedback", "notifications", "agent_conversations", "agent_ratings",
          "agent_memory", "realtime_events", "obs_spans")


def _is_sample_path(path: str) -> bool:
    parts = Path(str(path).replace("\\", "/")).parts
    return "sample_data" in parts


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _folder_ids() -> set[str]:
    """sample_data 폴더를 따로(임시 DB) 읽어 나온 매물번호들."""
    import logging

    from .agent import RealEstateAgent, public_office_no
    from .config import SAMPLE_DATA_DIR, Settings
    from .tool_agent.tools import tool_id

    out: set[str] = set()
    folder = SAMPLE_DATA_DIR / "토지"
    if not folder.exists():
        return out
    prev = logging.getLogger("real_estate_agent").level
    logging.getLogger("real_estate_agent").setLevel(logging.ERROR)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            a = RealEstateAgent(Settings(data_dir=folder, db_path=Path(tmp) / "s.db", log_dir=Path(tmp),
                                         llm_providers=["rule"]))
            a.sync()
            for l in a.listings() + list(a.hidden_listings):
                out |= {l.listing_id, l.office_no or "", tool_id(l), public_office_no(l.office_no) or ""}
            a.storage.conn.close()
    except Exception:  # noqa: BLE001 - 샘플 폴더를 못 읽어도 DB 표시만으로 정리
        pass
    finally:
        logging.getLogger("real_estate_agent").setLevel(prev)
    out.discard("")
    return out


def practice_ids(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """(연습용 샘플 매물번호, 보호할 실제 매물번호)."""
    from .agent import public_office_no

    sample, real = set(), set()
    if _has_table(conn, "listings"):
        for lid, data in conn.execute("SELECT listing_id, data FROM listings"):
            try:
                d = json.loads(data)
            except ValueError:
                continue
            keys = {lid, d.get("office_no") or "", public_office_no(d.get("office_no") or "") or ""} - {""}
            (sample if _is_sample_path(d.get("source_file", "")) else real).update(keys)
    sample |= _folder_ids()
    return sample - real, real


def _mentions(text: str, ids: set[str]) -> bool:
    return any(m in ids for m in NUMBER.findall(text or ""))


def _h(cid: str) -> str:
    return hashlib.sha256(cid.encode()).hexdigest()[:16]


def scan(conn: sqlite3.Connection, lock: Optional[threading.RLock] = None) -> dict[str, Any]:
    """지울 기록 찾기 (아무것도 바꾸지 않음)."""
    with lock or nullcontext():
        ids, real = practice_ids(conn)
        plan: dict[str, Any] = {"ids": sorted(ids), "protected": sorted(ids & real), "rows": {}}
        if not ids:
            return plan
        ph = ",".join("?" * len(ids))
        rows = plan["rows"]
        if _has_table(conn, "listings"):
            rows["listings"] = [r[0] for r in conn.execute(
                f"SELECT listing_id FROM listings WHERE listing_id IN ({ph})", tuple(ids))]
        for t in ("listing_edits", "feedback", "notifications"):
            if _has_table(conn, t):
                rows[t] = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE listing_id IN ({ph})", tuple(ids)).fetchone()[0]
        convs, traces = [], set()  # 대화마다: 연습 질문(턴)만 빼고, 남는 게 없으면 대화째 지움
        if _has_table(conn, "agent_conversations"):
            for cid, data in conn.execute("SELECT cid, data FROM agent_conversations"):
                try:
                    turns = json.loads(data).get("turns", [])
                except ValueError:
                    continue
                bad = [t for t in turns if set(t.get("listing_ids") or []) & ids or _mentions(t.get("answer", ""), ids)]
                if bad:
                    convs.append({"cid": cid, "delete": len(bad) == len(turns),
                                  "turn_ids": [t.get("id") for t in bad if t.get("id")]})
                    traces |= {t.get("trace_id") for t in bad if t.get("trace_id")}
        rows["agent_conversations"] = convs
        if _has_table(conn, "agent_ratings"):
            rows["agent_ratings"] = sum(
                conn.execute("SELECT COUNT(*) FROM agent_ratings WHERE conv = ? AND turn = ?", (_h(c["cid"]), t)).fetchone()[0]
                for c in convs for t in c["turn_ids"])
        if _has_table(conn, "agent_memory"):
            mem = []
            for owner, data in conn.execute("SELECT owner, data FROM agent_memory"):
                d = json.loads(data)
                if set(d.get("liked") or {}) & ids or set(d.get("rejected") or {}) & ids:
                    mem.append(owner)
            rows["agent_memory"] = mem
        if _has_table(conn, "realtime_events"):
            rows["realtime_events"] = [i for i, payload in conn.execute("SELECT id, payload FROM realtime_events")
                                       if _mentions(payload, ids)]
        if _has_table(conn, "obs_spans"):
            for tid, attrs in conn.execute("SELECT trace_id, attrs FROM obs_spans"):
                if _mentions(attrs or "", ids):
                    traces.add(tid)
            rows["obs_spans"] = sorted(traces)
    return plan


def summary(plan: dict[str, Any]) -> list[tuple[str, int]]:
    """사람이 읽는 요약: (항목, 개수)."""
    r = plan["rows"]
    n = lambda k: (len(r[k]) if isinstance(r.get(k), list) else int(r.get(k) or 0))  # noqa: E731
    return [("연습용 샘플 매물", n("listings")), ("그 매물의 관리자 수정 기록", n("listing_edits")),
            ("샘플 매물이 나온 AI 상담 대화", n("agent_conversations")), ("그 대화의 답변 평가(👍/👎)", n("agent_ratings")),
            ("방문자 기억 속 샘플 매물 (관심·제외에서만 뺌)", n("agent_memory")), ("실시간 알림·이벤트", n("realtime_events")),
            ("추적(trace) 기록", n("obs_spans")), ("예전 추천 화면 피드백·선제 알림", n("feedback") + n("notifications"))]


def backup(db_path: Path) -> Path:
    dest = Path(db_path).parent / "backup" / f"{Path(db_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    try:
        out = sqlite3.connect(str(dest))
        with out:
            src.backup(out)  # 쓰는 중이어도 안전한 SQLite 백업
        out.close()
    finally:
        src.close()
    return dest


def clean(conn: sqlite3.Connection, db_path: Optional[Path] = None, lock: Optional[threading.RLock] = None,
          make_backup: bool = True) -> dict[str, Any]:
    """연습 기록 지우기. 지우기 전에 백업. 반환: {'backup': 경로, 'removed': [(항목, 개수)], 'ids': [...]}"""
    plan = scan(conn, lock)
    result: dict[str, Any] = {"ids": plan["ids"], "protected": plan["protected"], "removed": summary(plan),
                              "backup": None}
    if not plan["ids"] or not any(n for _, n in result["removed"]):
        return result
    if make_backup and db_path:
        result["backup"] = str(backup(db_path))
    ids, r = set(plan["ids"]), plan["rows"]
    ph = ",".join("?" * len(ids))
    with lock or nullcontext():
        for t in ("listings", "listing_edits", "feedback", "notifications"):
            if _has_table(conn, t):
                conn.execute(f"DELETE FROM {t} WHERE listing_id IN ({ph})", tuple(ids))
        for c in r.get("agent_conversations") or []:
            if _has_table(conn, "agent_ratings"):
                conn.executemany("DELETE FROM agent_ratings WHERE conv = ? AND turn = ?",
                                 [(_h(c["cid"]), t) for t in c["turn_ids"]])
            if c["delete"]:
                conn.execute("DELETE FROM agent_conversations WHERE cid = ?", (c["cid"],))
                continue
            d = json.loads(conn.execute("SELECT data FROM agent_conversations WHERE cid = ?", (c["cid"],)).fetchone()[0])
            gone = set(c["turn_ids"])
            d["turns"] = [t for t in d.get("turns", []) if t.get("id") not in gone
                          and not (set(t.get("listing_ids") or []) & ids or _mentions(t.get("answer", ""), ids))]
            d["history"] = []  # AI 가 이어 볼 대화 기록에서도 연습 내용을 뺌 (화면의 질문·답변은 남은 것만)
            ctx = d.get("ctx") or {}
            for k in ("liked", "seen", "last_shown"):
                ctx[k] = [i for i in ctx.get(k) or [] if i not in ids]
            ctx["rejected"] = {k: v for k, v in (ctx.get("rejected") or {}).items() if k not in ids}
            if ctx.get("last_shown") == [] or c["delete"]:
                ctx["last_search"] = {}
            d["ctx"] = ctx
            conn.execute("UPDATE agent_conversations SET data = ? WHERE cid = ?", (json.dumps(d, ensure_ascii=False),
                                                                                 c["cid"]))
        for owner in r.get("agent_memory") or []:
            d = json.loads(conn.execute("SELECT data FROM agent_memory WHERE owner = ?", (owner,)).fetchone()[0])
            d["liked"] = {k: v for k, v in (d.get("liked") or {}).items() if k not in ids}
            d["rejected"] = {k: v for k, v in (d.get("rejected") or {}).items() if k not in ids}
            conn.execute("UPDATE agent_memory SET data = ? WHERE owner = ?", (json.dumps(d, ensure_ascii=False), owner))
        if r.get("realtime_events"):
            conn.executemany("DELETE FROM realtime_events WHERE id = ?", [(i,) for i in r["realtime_events"]])
        if r.get("obs_spans"):
            conn.executemany("DELETE FROM obs_spans WHERE trace_id = ?", [(t,) for t in r["obs_spans"]])
        conn.commit()
    return result
