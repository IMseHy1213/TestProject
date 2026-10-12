"""SQLite 저장소: 정제된 매물, 손님 프로필, 알림/피드백 이력."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import CustomerRequirement, LandListing
from .preference import PreferenceModel

OFFICE_MODEL = "__office__"  # 사무실 전체 피드백으로 학습한 공통 모델

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS customers (
    name TEXT PRIMARY KEY,
    requirement TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    customer TEXT NOT NULL,
    listing_id TEXT NOT NULL,
    score REAL NOT NULL,
    message TEXT,
    created TEXT NOT NULL,
    PRIMARY KEY (customer, listing_id)
);
CREATE TABLE IF NOT EXISTS preferences (
    name TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer TEXT NOT NULL,
    listing_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    created TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fingerprint(l: LandListing) -> str:
    key = f"{l.address}|{l.area_m2}|{l.price_manwon}|{l.status}|{l.jimok}|{l.zoning}|{l.road_access}"
    return hashlib.sha1(key.encode()).hexdigest()


class Storage:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # 웹 서버는 여러 스레드에서 접근하므로 check_same_thread=False (동시 접근은 웹 앱이 잠금으로 보호)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------------- listings ----------------
    def upsert_listings(self, listings: Iterable[LandListing]) -> dict[str, list[LandListing]]:
        """반환: {'new': [...], 'changed': [...], 'removed_ids': [...]}"""
        now = _now()
        new, changed = [], []
        current_ids = set()
        cur = self.conn.cursor()
        existing = {r["listing_id"]: r["fingerprint"] for r in cur.execute("SELECT listing_id, fingerprint FROM listings")}
        for l in listings:
            current_ids.add(l.listing_id)
            fp = fingerprint(l)
            data = json.dumps(l.to_dict(), ensure_ascii=False, default=str)
            if l.listing_id not in existing:
                cur.execute("INSERT INTO listings VALUES (?,?,?,?,?,1)", (l.listing_id, data, fp, now, now))
                new.append(l)
            else:
                if existing[l.listing_id] != fp:
                    changed.append(l)
                cur.execute("UPDATE listings SET data=?, fingerprint=?, last_seen=?, active=1 WHERE listing_id=?",
                            (data, fp, now, l.listing_id))
        removed = [i for i in existing if i not in current_ids]
        cur.executemany("UPDATE listings SET active=0 WHERE listing_id=?", [(i,) for i in removed])
        self.conn.commit()
        return {"new": new, "changed": changed, "removed_ids": removed}

    def active_listings(self) -> list[LandListing]:
        rows = self.conn.execute("SELECT data FROM listings WHERE active=1").fetchall()
        return [LandListing.from_dict(json.loads(r["data"])) for r in rows]

    # ---------------- customers ----------------
    def save_customer(self, req: CustomerRequirement) -> None:
        if not req.customer_name:
            raise ValueError("손님 이름이 필요합니다")
        now = _now()
        self.conn.execute(
            "INSERT INTO customers VALUES (?,?,1,?,?) ON CONFLICT(name) DO UPDATE SET requirement=excluded.requirement,"
            " active=1, updated=excluded.updated",
            (req.customer_name, json.dumps(req.to_dict(), ensure_ascii=False), now, now),
        )
        self.conn.commit()

    def customers(self, active_only: bool = True) -> list[CustomerRequirement]:
        q = "SELECT requirement FROM customers" + (" WHERE active=1" if active_only else "") + " ORDER BY updated DESC"
        return [CustomerRequirement.from_dict(json.loads(r["requirement"])) for r in self.conn.execute(q)]

    def get_customer(self, name: str) -> CustomerRequirement | None:
        r = self.conn.execute("SELECT requirement FROM customers WHERE name=?", (name,)).fetchone()
        return CustomerRequirement.from_dict(json.loads(r["requirement"])) if r else None

    def deactivate_customer(self, name: str) -> None:
        self.conn.execute("UPDATE customers SET active=0 WHERE name=?", (name,))
        self.conn.commit()

    # ---------------- notifications / feedback ----------------
    def was_notified(self, customer: str, listing_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM notifications WHERE customer=? AND listing_id=?",
                                 (customer, listing_id)).fetchone() is not None

    def add_notification(self, customer: str, listing_id: str, score: float, message: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO notifications VALUES (?,?,?,?,?)",
                          (customer, listing_id, score, message, _now()))
        self.conn.commit()

    def notifications(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM notifications ORDER BY created DESC LIMIT ?", (limit,)).fetchall()

    def add_feedback(self, customer: str, listing_id: str, action: str, reason: str = "") -> None:
        self.conn.execute("INSERT INTO feedback (customer, listing_id, action, reason, created) VALUES (?,?,?,?,?)",
                          (customer or "(익명)", listing_id, action, reason, _now()))
        self.conn.commit()

    def rejected_ids(self, customer: str) -> set[str]:
        """이전에 거절한 매물은 다음 상담 때 다시 추천하지 않음."""
        if not customer:
            return set()
        rows = self.conn.execute("SELECT listing_id FROM feedback WHERE customer=? AND action='reject'", (customer,))
        return {r["listing_id"] for r in rows}

    # ---------------- 학습 모델 ----------------
    def load_preference(self, name: str) -> PreferenceModel:
        r = self.conn.execute("SELECT data FROM preferences WHERE name=?", (name,)).fetchone()
        return PreferenceModel.from_dict(json.loads(r["data"])) if r else PreferenceModel()

    def save_preference(self, name: str, model: PreferenceModel) -> None:
        self.conn.execute(
            "INSERT INTO preferences VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET data=excluded.data,"
            " updated=excluded.updated",
            (name, json.dumps(model.to_dict(), ensure_ascii=False), _now()),
        )
        self.conn.commit()
