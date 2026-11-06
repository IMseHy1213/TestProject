"""v1.2 답변 평가(👍/👎) 저장과 온라인 평가 요약.

손님이 답 아래 👍 도움됐어요 / 👎 아쉬워요(+이유)를 누르면:
  - 평가 표(agent_ratings)에 저장: 어느 방식·누가 답했는지(AI/규칙)·안전장치가 작동했는지와 함께
  - 그 답의 추적(v1.1 trace)에 'feedback.rating' 으로 이어 붙임 → 👎 받은 답이 왜 그랬는지 바로 열어 봄
  - '자동' 방식이 고른 답이면 그 방식의 점수(밴딧)에 반영
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_ratings (
    conv     TEXT NOT NULL,
    turn     TEXT NOT NULL,
    owner    TEXT,
    created  TEXT NOT NULL,
    trace_id TEXT,
    mode     TEXT,
    variant  TEXT,
    policy   TEXT,
    guarded  INTEGER DEFAULT 0,
    rating   INTEGER NOT NULL,
    reason   TEXT,
    question TEXT,
    PRIMARY KEY (conv, turn)
);
CREATE INDEX IF NOT EXISTS idx_agent_ratings_created ON agent_ratings(created);
"""
DOWN_REASONS = ["원하는 매물이 아니에요", "답이 틀렸어요", "질문을 이해 못 했어요", "너무 느려요"]


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class RatingStore:
    def __init__(self, conn, lock=None):
        self.conn, self.lock = conn, lock or threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def save(self, owner: str, cid: str, turn: dict, rating: int, reason: str = "") -> Optional[int]:
        """저장하고, 이전 평가(바꾼 경우)를 돌려줌."""
        from ..observability.tracing import redact

        key = _h(cid)
        with self.lock:
            prev = self.conn.execute("SELECT rating FROM agent_ratings WHERE conv = ? AND turn = ?",
                                     (key, turn["id"])).fetchone()
            self.conn.execute(
                "INSERT OR REPLACE INTO agent_ratings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, turn["id"], _h(owner), datetime.now().isoformat(timespec="seconds"), turn.get("trace_id", ""),
                 turn.get("mode", ""), turn.get("variant", ""), turn.get("policy", ""),
                 int(any(s.get("kind") == "guard" for s in turn.get("steps", []))), rating, reason[:60],
                 redact(turn.get("question", ""))))
            self.conn.commit()
        return prev[0] if prev else None

    def summary(self, days: float = 7) -> dict[str, Any]:
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self.lock or nullcontext():
            rows = self.conn.execute("SELECT mode, variant, policy, guarded, rating, reason FROM agent_ratings "
                                     "WHERE created >= ?", (since,)).fetchall()

        def group(key) -> list[dict]:
            out: dict[str, list[int]] = {}
            for r in rows:
                out.setdefault(key(r), []).append(r[4])
            return [{"name": k, "n": len(v), "up": sum(1 for x in v if x == 1),
                     "rate": round(100 * sum(1 for x in v if x == 1) / len(v), 1)} for k, v in sorted(out.items())]

        reasons: dict[str, int] = {}
        for r in rows:
            if r[4] == 0 and r[5]:
                reasons[r[5]] = reasons.get(r[5], 0) + 1
        total = len(rows)
        return {"days": days, "n": total, "up": sum(1 for r in rows if r[4] == 1),
                "rate": round(100 * sum(1 for r in rows if r[4] == 1) / total, 1) if total else None,
                "by_mode": group(lambda r: r[0] or "-"),
                "by_policy": group(lambda r: "AI" if r[2] not in ("rule", "") else "규칙 기반"),
                "by_guard": group(lambda r: "안전장치 작동" if r[3] else "그대로"),
                "by_auto": group(lambda r: ("자동 → " + r[1].split(":", 1)[1]) if (r[1] or "").startswith("auto:")
                                 else "직접 고름"),
                "reasons": sorted(reasons.items(), key=lambda x: -x[1])}

    def recent_negative(self, limit: int = 10) -> list[dict]:
        with self.lock or nullcontext():
            rows = self.conn.execute("SELECT created, question, reason, mode, policy, trace_id FROM agent_ratings "
                                     "WHERE rating = 0 ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"time": r[0][5:16].replace("T", " "), "question": r[1], "reason": r[2], "mode": r[3],
                 "policy": r[4], "trace_id": r[5]} for r in rows]
