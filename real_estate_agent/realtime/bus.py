"""이벤트 버스 (pub-sub) + 이벤트 기록 (SQLite).

- publish(topic, payload, owner): 이벤트를 SQLite 에 기록(번호 부여)하고, 지금 구독 중인 모든 구독자에게 즉시 전달
- subscribe(): 구독자 전용 큐를 돌려줌 (화면 연결 하나 = 구독자 하나)
- since(last_id, owner): 연결이 끊겼다 다시 붙으면 놓친 이벤트를 번호 순서대로 다시 보내기 (SSE Last-Event-ID)

토픽
  listing.added        새 매물
  listing.price_drop   가격 인하
  listing.changed      가격 외 정보 변경 (상태 등)
  listing.removed      매물 빠짐 (계약완료·숨김·파일 삭제)
  alert.match          손님이 찾던 조건에 맞는 새 매물/가격 인하 (owner = 그 손님)
  system.sync          매물 폴더를 다시 읽음 (건수)
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS realtime_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    owner TEXT,                 -- NULL 이면 모두에게 (공개), 있으면 그 방문자에게만
    payload TEXT NOT NULL,
    created TEXT NOT NULL,
    dedup TEXT UNIQUE           -- 같은 알림을 두 번 만들지 않기 위한 키 (멱등성)
);
CREATE INDEX IF NOT EXISTS realtime_events_owner ON realtime_events(owner, id);
"""

MAX_QUEUE = 200     # 구독자 하나가 쌓아 둘 수 있는 이벤트 수 (느린 화면 때문에 메모리가 늘지 않게)
KEEP_EVENTS = 5000  # 기록 보관 개수


@dataclass
class Event:
    id: int
    topic: str
    payload: dict[str, Any]
    owner: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ts: float = field(default_factory=time.time)  # 지연시간 측정용

    def public(self) -> dict[str, Any]:
        """화면/API 로 보낼 형태 (owner 는 보내지 않음)."""
        d = asdict(self)
        d.pop("owner", None)
        return d


class EventBus:
    def __init__(self, conn: sqlite3.Connection, lock: Optional[threading.RLock] = None):
        self.conn = conn
        self.lock = lock or threading.RLock()
        self._subs: list[tuple[queue.Queue, Optional[str], bool]] = []  # (큐, owner, 전체 보기)
        self._subs_lock = threading.Lock()
        self.dropped = 0
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------------ 발행
    def publish(self, topic: str, payload: dict[str, Any], owner: Optional[str] = None,
                dedup: Optional[str] = None) -> Optional[Event]:
        """기록 + 전달. dedup 키가 이미 있으면 (같은 알림) 아무것도 하지 않고 None."""
        created = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO realtime_events (topic, owner, payload, created, dedup) VALUES (?,?,?,?,?)",
                    (topic, owner, json.dumps(payload, ensure_ascii=False, default=str), created, dedup))
            except sqlite3.IntegrityError:
                return None
            self.conn.commit()
            event = Event(cur.lastrowid, topic, payload, owner, created)
            if event.id % 500 == 0:
                self.conn.execute("DELETE FROM realtime_events WHERE id <= ?", (event.id - KEEP_EVENTS,))
                self.conn.commit()
        with self._subs_lock:
            subs = list(self._subs)
        for q, sub_owner, see_all in subs:
            if see_all or event.owner is None or event.owner == sub_owner:
                try:
                    q.put_nowait(event)
                except queue.Full:  # 화면이 너무 느리면 버림 (다시 연결하면 기록에서 받아 감)
                    self.dropped += 1
        return event

    # ------------------------------------------------------------------ 구독
    def subscribe(self, owner: Optional[str] = None, see_all: bool = False) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        with self._subs_lock:
            self._subs.append((q, owner, see_all))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            self._subs = [s for s in self._subs if s[0] is not q]

    @property
    def subscribers(self) -> int:
        with self._subs_lock:
            return len(self._subs)

    # ------------------------------------------------------------------ 기록 조회
    def since(self, last_id: int, owner: Optional[str] = None, see_all: bool = False,
              limit: int = 100) -> list[Event]:
        """놓친 이벤트 (다시 연결했을 때)."""
        sql = "SELECT * FROM realtime_events WHERE id > ?"
        args: list[Any] = [last_id]
        if not see_all:
            sql += " AND (owner IS NULL OR owner = ?)"
            args.append(owner)
        sql += " ORDER BY id LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [Event(r["id"], r["topic"], json.loads(r["payload"]), r["owner"], r["created"]) for r in rows]

    def recent(self, limit: int = 20, owner: Optional[str] = None, see_all: bool = False,
               topics: Optional[tuple[str, ...]] = None) -> list[Event]:
        sql = "SELECT * FROM realtime_events WHERE 1=1"
        args: list[Any] = []
        if not see_all:
            sql += " AND (owner IS NULL OR owner = ?)"
            args.append(owner)
        if topics:
            sql += f" AND topic IN ({','.join('?' * len(topics))})"
            args.extend(topics)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [Event(r["id"], r["topic"], json.loads(r["payload"]), r["owner"], r["created"]) for r in rows]

    def last_id(self) -> int:
        with self.lock:
            row = self.conn.execute("SELECT MAX(id) AS m FROM realtime_events").fetchone()
        return int(row["m"] or 0)

    def counts(self) -> dict[str, int]:
        with self.lock:
            rows = self.conn.execute("SELECT topic, COUNT(*) AS n FROM realtime_events GROUP BY topic").fetchall()
        return {r["topic"]: r["n"] for r in rows}
