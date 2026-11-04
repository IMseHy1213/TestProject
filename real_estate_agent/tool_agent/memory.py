"""v0.3 State & Memory: AI 상담의 상태 저장과 장기 기억 (SQLite).

단기 기억 (대화 상태)  : agent_conversations 테이블
  - 대화 기록(모델에게 보낸 메시지·도구 결과), 도구 상태(AgentContext), 화면에 보여준 질문/답변
  - Agent 가 한 단계 진행할 때마다 체크포인트로 저장 -> 서버가 꺼져도 대화가 남고,
    답을 만들다 끊긴 질문은 마지막 체크포인트부터 이어서 진행 (resume)
장기 기억 (손님 프로필): agent_memory 테이블 (방문자별)
  - 최근에 찾은 조건, 관심 매물, 싫다고 한 매물(이유)
  - 다음 상담에서: 싫다고 한 매물은 빼고 찾고, '지난번에 찾던 거' 를 알아듣고, AI 에게 요약해 알려 줌
  - 화면의 '기억 지우기' 로 언제든 삭제
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from typing import Any, Optional

from ..llm import ToolCall
from .tools import AgentContext

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_conversations (
    cid TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    data TEXT NOT NULL,          -- {"history": [...], "ctx": {...}, "turns": [...]}
    status TEXT NOT NULL,        -- idle / running
    pending TEXT,                -- 답을 만드는 중인 질문
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_conversations_owner ON agent_conversations(owner, updated);
CREATE TABLE IF NOT EXISTS agent_memory (
    owner TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated TEXT NOT NULL
);
"""

KEEP_DAYS = 30           # 대화 보관 기간
MAX_SEARCHES = 10        # 기억할 최근 검색 조건 수
STALE_SECONDS = 120      # 'running' 인 채로 이 시간이 지나면 끊긴 것으로 봄


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------- 직렬화
def dump_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [asdict(c) if isinstance(c, ToolCall) else c for c in m["tool_calls"]]
        out.append(m)
    return out


def load_messages(raw: list[dict]) -> list[dict]:
    out = []
    for m in raw or []:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [ToolCall(**c) if isinstance(c, dict) else c for c in m["tool_calls"]]
        out.append(m)
    return out


def dump_ctx(ctx: AgentContext) -> dict:
    return asdict(ctx)


def load_ctx(raw: Optional[dict]) -> AgentContext:
    names = {f.name for f in fields(AgentContext)}
    return AgentContext(**{k: v for k, v in (raw or {}).items() if k in names})


def is_incomplete(history: list[dict]) -> bool:
    """마지막 질문에 대한 최종 답이 아직 없는지 (도중에 끊긴 대화)."""
    if not history:
        return False
    last = history[-1]
    return last["role"] != "assistant" or bool(last.get("tool_calls"))


def repair_for_resume(history: list[dict]) -> list[dict]:
    """끊긴 지점이 '도구 호출은 했는데 결과가 다 오지 않은' 상태면 그 호출부터 다시 하도록 잘라냄."""
    msgs = list(history)
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            wanted = {c.id for c in m["tool_calls"]}
            got = {x.get("call_id") for x in msgs[i + 1:] if x["role"] == "tool"}
            if not wanted <= got:
                return msgs[:i]
            return msgs
        if m["role"] == "user":
            return msgs
    return msgs


# ---------------------------------------------------------------------- 대화 저장소 (단기 기억 + 체크포인트)
class ConversationStore:
    def __init__(self, conn: sqlite3.Connection, lock: Optional[threading.RLock] = None):
        self.conn = conn
        self.lock = lock or threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def load(self, cid: Optional[str]) -> Optional[dict]:
        if not cid:
            return None
        with self.lock:
            row = self.conn.execute("SELECT * FROM agent_conversations WHERE cid=?", (cid,)).fetchone()
        if row is None:
            return None
        data = json.loads(row["data"])
        return {"cid": row["cid"], "owner": row["owner"], "history": load_messages(data.get("history", [])),
                "ctx": load_ctx(data.get("ctx")), "turns": data.get("turns", []), "status": row["status"],
                "pending": row["pending"], "updated": row["updated"]}

    def latest_for(self, owner: str, within_hours: int = 24) -> Optional[dict]:
        """이 방문자가 최근에 하던 대화 (브라우저를 껐다 켜도 이어서 보여주기)."""
        since = (datetime.now() - timedelta(hours=within_hours)).isoformat(timespec="seconds")
        with self.lock:
            row = self.conn.execute("SELECT cid FROM agent_conversations WHERE owner=? AND updated>=? "
                                    "ORDER BY updated DESC LIMIT 1", (owner, since)).fetchone()
        return self.load(row["cid"]) if row else None

    def save(self, cid: str, owner: str, history: list[dict], ctx: AgentContext, turns: list[dict],
             status: str = "idle", pending: Optional[str] = None) -> None:
        data = json.dumps({"history": dump_messages(history), "ctx": dump_ctx(ctx), "turns": turns},
                          ensure_ascii=False, default=str)
        now = _now()
        with self.lock:
            self.conn.execute(
                "INSERT INTO agent_conversations (cid, owner, data, status, pending, created, updated) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(cid) DO UPDATE SET data=excluded.data, status=excluded.status, "
                "pending=excluded.pending, updated=excluded.updated",
                (cid, owner, data, status, pending, now, now))
            self.conn.commit()

    def delete(self, cid: Optional[str]) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM agent_conversations WHERE cid=?", (cid or "",))
            self.conn.commit()

    def delete_owner(self, owner: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM agent_conversations WHERE owner=?", (owner,))
            self.conn.commit()

    def prune(self, days: int = KEEP_DAYS) -> None:
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self.lock:
            self.conn.execute("DELETE FROM agent_conversations WHERE updated<?", (since,))
            self.conn.commit()

    @staticmethod
    def is_stale(conv: dict) -> bool:
        """'답을 만드는 중' 상태로 너무 오래 멈춰 있으면 (서버가 꺼졌거나 오류) 끊긴 것으로 봄."""
        if conv.get("status") != "running":
            return False
        try:
            age = (datetime.now() - datetime.fromisoformat(conv["updated"])).total_seconds()
        except (TypeError, ValueError):
            return True
        return age > STALE_SECONDS


# ---------------------------------------------------------------------- 장기 기억 (방문자 프로필)
class LongTermMemory:
    def __init__(self, conn: sqlite3.Connection, lock: Optional[threading.RLock] = None):
        self.conn = conn
        self.lock = lock or threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    @staticmethod
    def empty() -> dict:
        return {"visits": 0, "first_seen": None, "last_seen": None, "searches": [], "liked": {}, "rejected": {}}

    def load(self, owner: str) -> dict:
        with self.lock:
            row = self.conn.execute("SELECT data FROM agent_memory WHERE owner=?", (owner,)).fetchone()
        data = self.empty()
        if row:
            data.update(json.loads(row["data"]))
        return data

    def save(self, owner: str, data: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO agent_memory (owner, data, updated) VALUES (?,?,?) "
                              "ON CONFLICT(owner) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                              (owner, json.dumps(data, ensure_ascii=False, default=str), _now()))
            self.conn.commit()

    def forget(self, owner: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM agent_memory WHERE owner=?", (owner,))
            self.conn.commit()

    # ------------------------------------------------------------------ 상담 시작/끝
    def apply_to(self, owner: str, ctx: AgentContext) -> dict:
        """상담 시작: 지난 상담에서 싫다고 한 매물은 계속 빼고, 직전 검색 조건을 '지난번 조건'으로 준비."""
        mem = self.load(owner)
        for lid, info in mem["rejected"].items():
            ctx.rejected.setdefault(lid, info.get("reason", "") if isinstance(info, dict) else "")
        for lid in mem["liked"]:
            if lid not in ctx.liked:
                ctx.liked.append(lid)
        if not ctx.last_search and mem["searches"]:
            ctx.last_search = dict(mem["searches"][-1].get("args") or {})
            ctx.last_search_from_memory = True
        return mem

    def update_after(self, owner: str, ctx: AgentContext, searches: list[dict]) -> dict:
        """상담 한 번(질문 하나)이 끝나면: 새로 찾은 조건, 관심/제외 매물을 기억에 반영."""
        mem = self.load(owner)
        today = datetime.now().date().isoformat()
        if mem.get("last_seen") != today:  # 방문 일수 (같은 날 여러 질문은 1번)
            mem["visits"] = int(mem.get("visits") or 0) + 1
        mem["first_seen"] = mem.get("first_seen") or today
        mem["last_seen"] = today
        for s in searches:
            mem["searches"] = [x for x in mem["searches"] if x.get("조건") != s["조건"]] + [{**s, "at": today}]
        mem["searches"] = mem["searches"][-MAX_SEARCHES:]
        for lid, reason in ctx.rejected.items():
            mem["rejected"].setdefault(lid, {"reason": reason, "at": today})
            mem["liked"].pop(lid, None)
        for lid in ctx.liked:
            if lid not in ctx.rejected:
                mem["liked"].setdefault(lid, {"at": today})
        self.save(owner, mem)
        return mem


def memory_summary(mem: dict, name_of) -> str:
    """AI 에게 알려 줄 장기 기억 요약 (공개용 매물 이름만)."""
    parts = []
    if mem.get("searches"):
        recent = [s["조건"] for s in mem["searches"][-3:]]
        parts.append("최근에 찾은 조건: " + " / ".join(reversed(recent)))
    if mem.get("liked"):
        parts.append("관심 매물: " + ", ".join(name_of(i) for i in list(mem["liked"])[-5:]))
    if mem.get("rejected"):
        items = list(mem["rejected"].items())[-5:]
        parts.append("싫다고 한 매물(검색에서 자동 제외): " + ", ".join(
            name_of(i) + (f"({v.get('reason')})" if isinstance(v, dict) and v.get("reason") else "") for i, v in items))
    return "; ".join(parts)
