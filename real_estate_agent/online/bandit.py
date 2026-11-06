"""v1.2 '자동' 상담 방식: 어느 방식(상담원 1명 / 상담팀)이 손님을 더 만족시키는지 운영하면서 스스로 배움.

다중 슬롯머신(multi-armed bandit) 문제의 톰슨 샘플링(Thompson sampling):
  - 방식마다 👍·👎 수로 '만족 확률'의 분포 Beta(1+👍, 1+👎) 를 둠
  - 새 대화가 오면 방식마다 분포에서 숫자를 하나씩 뽑아 가장 큰 방식을 고름
    → 처음에는 둘 다 골고루 써 보고(탐색), 👍가 쌓인 쪽을 점점 더 많이 고름(활용)
  - 한 대화 안에서는 같은 방식을 계속 씀 (중간에 바뀌지 않게)
A/B 테스트처럼 반반 나누고 끝날 때까지 기다리지 않고, 결과를 보면서 바로바로 비율을 바꾸는 방식입니다.
"""

from __future__ import annotations

import os
import random
import threading
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS online_bandit (
    experiment TEXT NOT NULL,
    arm        TEXT NOT NULL,
    wins       REAL NOT NULL DEFAULT 0,
    losses     REAL NOT NULL DEFAULT 0,
    pulls      INTEGER NOT NULL DEFAULT 0,
    updated    TEXT,
    PRIMARY KEY (experiment, arm)
);
"""
EXPERIMENT = "consult_mode"


def default_arms() -> list[str]:
    arms = [a.strip() for a in os.getenv("REA_AUTO_MODES", "single,multi").split(",") if a.strip()]
    return [a for a in arms if a in {"single", "multi", "distributed"}] or ["single", "multi"]


class ThompsonBandit:
    def __init__(self, conn, lock=None, arms: Optional[list[str]] = None, experiment: str = EXPERIMENT,
                 rng: Optional[random.Random] = None):
        self.conn, self.lock = conn, lock or threading.RLock()
        self.arms, self.experiment = arms or default_arms(), experiment
        self.rng = rng or random.Random()
        with self.lock:
            self.conn.executescript(SCHEMA)
            for a in self.arms:
                self.conn.execute("INSERT OR IGNORE INTO online_bandit (experiment, arm) VALUES (?, ?)",
                                  (experiment, a))
            self.conn.commit()

    def _rows(self) -> dict[str, dict]:
        with self.lock or nullcontext():
            rows = self.conn.execute("SELECT arm, wins, losses, pulls FROM online_bandit WHERE experiment = ?",
                                     (self.experiment,)).fetchall()
        return {r[0]: {"wins": r[1], "losses": r[2], "pulls": r[3]} for r in rows if r[0] in self.arms}

    def choose(self) -> str:
        """톰슨 샘플링으로 방식 하나 고르기 (고른 횟수 기록)."""
        rows = self._rows()
        samples = {a: self.rng.betavariate(1 + rows[a]["wins"], 1 + rows[a]["losses"]) for a in self.arms}
        arm = max(samples, key=samples.get)
        with self.lock:
            self.conn.execute("UPDATE online_bandit SET pulls = pulls + 1, updated = ? WHERE experiment = ? AND arm = ?",
                              (datetime.now().isoformat(timespec="seconds"), self.experiment, arm))
            self.conn.commit()
        return arm

    def update(self, arm: str, reward: Optional[int], previous: Optional[int] = None) -> None:
        """reward 1(👍) / 0(👎). 손님이 평가를 바꾸면 previous 를 빼고 새 값을 더함."""
        if arm not in self.arms:
            return
        dw = (reward == 1) - (previous == 1)
        dl = (reward == 0) - (previous == 0)
        if not dw and not dl:
            return
        with self.lock:
            self.conn.execute("UPDATE online_bandit SET wins = MAX(0, wins + ?), losses = MAX(0, losses + ?), "
                              "updated = ? WHERE experiment = ? AND arm = ?",
                              (dw, dl, datetime.now().isoformat(timespec="seconds"), self.experiment, arm))
            self.conn.commit()

    def stats(self, samples: int = 4000) -> list[dict]:
        """방식별: 고른 횟수, 👍/👎, 만족도 추정(평균), 95% 구간, '가장 좋은 방식일 확률'."""
        rows = self._rows()
        rng = random.Random(7)
        best = {a: 0 for a in self.arms}
        draws = {a: [] for a in self.arms}
        for _ in range(samples):
            s = {a: rng.betavariate(1 + rows[a]["wins"], 1 + rows[a]["losses"]) for a in self.arms}
            best[max(s, key=s.get)] += 1
            for a, v in s.items():
                draws[a].append(v)
        out = []
        for a in self.arms:
            w, l = rows[a]["wins"], rows[a]["losses"]
            d = sorted(draws[a])
            out.append({"arm": a, "pulls": rows[a]["pulls"], "wins": int(w), "losses": int(l),
                        "rated": int(w + l), "mean": round((1 + w) / (2 + w + l), 3),
                        "low": round(d[int(0.025 * len(d))], 3), "high": round(d[int(0.975 * len(d)) - 1], 3),
                        "p_best": round(best[a] / samples, 3)})
        return out

    def reset(self) -> None:
        with self.lock:
            self.conn.execute("UPDATE online_bandit SET wins = 0, losses = 0, pulls = 0 WHERE experiment = ?",
                              (self.experiment,))
            self.conn.commit()
