"""v1.0 과부하 보호.

- RateLimiter: 한 방문자가 1분에 물을 수 있는 횟수 (기본 20번). 자동 반복 요청이 AI 비용·서버를 다 쓰지 않게.
- Bulkhead(격벽): 동시에 처리하는 상담 수 제한 (기본 4개). 상담이 몰려도 매물 목록·관리자 화면 같은 다른 화면은 계속 열리게.
  자리가 없으면 잠깐(기본 15초) 줄을 서고, 그래도 없으면 '잠시 후 다시' 로 거절.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class RateLimiter:
    def __init__(self, per_minute: int | None = None, window: float = 60.0):
        self.per_minute = int(per_minute if per_minute is not None else _env_num("REA_RATE_PER_MIN", 20))
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """허용되면 0, 아니면 다시 물을 수 있을 때까지 남은 초 (허용될 때만 기록)."""
        if self.per_minute <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(0.1, self.window - (now - q[0]))
            q.append(now)
            if len(self._hits) > 10000:  # 오래된 방문자 정리
                for k in [k for k, v in self._hits.items() if not v][:5000]:
                    self._hits.pop(k, None)
            return 0.0


class Bulkhead:
    def __init__(self, limit: int | None = None, wait: float | None = None):
        self.limit = max(1, int(limit if limit is not None else _env_num("REA_MAX_CONCURRENT_AGENTS", 4)))
        self.wait = wait if wait is not None else _env_num("REA_QUEUE_WAIT", 15)
        self._sem = threading.BoundedSemaphore(self.limit)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0

    def acquire(self) -> bool:
        with self._lock:
            self.waiting += 1
        try:
            ok = self._sem.acquire(timeout=self.wait)
        finally:
            with self._lock:
                self.waiting -= 1
        if ok:
            with self._lock:
                self.active += 1
        return ok

    def release(self) -> None:
        with self._lock:
            self.active -= 1
        self._sem.release()
