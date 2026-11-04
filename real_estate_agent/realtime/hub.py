"""v0.9 실시간 · 이벤트 기반 Agent.

    [생산자]                        [이벤트 버스]             [작업 큐 → 워커]                       [구독자]
    매물 폴더 감시 스레드 ──┐                                 매칭 워커 N개: 새 매물·가격 인하를
    관리자 추가·수정·업로드 ─┴→ 변경 감지 → listing.* 발행 → 방문자들의 장기 기억(찾던 조건)과 대조 → alert.match → 손님 화면(SSE)
                                         (SQLite 기록)                                               관리자 화면, API(SSE/WebSocket)

- 변경 감지: 지난번 매물 목록(가격·상태)과 지금을 비교 → 새 매물 / 가격 인하 / 정보 변경 / 빠짐
- 폴더 감시: 매물 파일들의 수정 시각·크기만 몇 초마다 확인하고, 바뀌었을 때만 매물 폴더를 다시 읽음
- 매칭: 손님별 최근 검색 조건으로 같은 규칙(필수 조건 + 점수)을 적용. 싫다고 한 매물은 제외.
        같은 알림은 한 번만 (owner|매물|종류|가격 키로 중복 방지)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..models import format_price
from .bus import Event, EventBus

log = logging.getLogger(__name__)

WATCH_INTERVAL = float(os.getenv("REA_REALTIME_INTERVAL", "10"))   # 매물 폴더 확인 간격 (초)
ALERT_THRESHOLD = float(os.getenv("REA_ALERT_THRESHOLD", "55"))    # 알림을 보낼 최소 점수
WORKERS = 2


@dataclass
class Stats:
    detected: int = 0          # 감지한 매물 변경 이벤트 수
    alerts: int = 0            # 보낸 맞춤 알림 수
    jobs_done: int = 0
    job_errors: int = 0
    syncs: int = 0
    match_ms: list[float] = field(default_factory=list)  # 변경 감지 → 알림 발행까지 걸린 시간

    def as_dict(self) -> dict[str, Any]:
        ms = sorted(self.match_ms[-200:])
        p = (lambda q: round(ms[min(len(ms) - 1, int(len(ms) * q))], 1)) if ms else (lambda q: None)
        return {"detected": self.detected, "alerts": self.alerts, "jobs_done": self.jobs_done,
                "job_errors": self.job_errors, "syncs": self.syncs, "match_ms_p50": p(0.5), "match_ms_p95": p(0.95)}


def listing_card(agent, lid: str) -> Optional[dict[str, Any]]:
    """이벤트에 실을 매물 정보 (손님 화면과 같은 공개 기준: 시/도+시/군/구, 연락처 없음)."""
    from ..agent import mask_address, public_office_no
    from ..models import format_price
    from ..tool_agent.tools import display_name, tool_id

    l = agent.find_listing(lid)
    if l is None:
        return None
    return {"id": tool_id(l), "office_no": public_office_no(l.office_no) or None, "name": display_name(l),
            "region": mask_address(l.address), "price_manwon": l.price_manwon, "price": format_price(l.price_manwon),
            "area_pyeong": round(l.area_pyeong), "jimok": l.jimok or None, "zoning": l.zoning or None,
            "use_tag": l.use_tag or None,
            "features": [f.lstrip("● ").strip() for f in (l.features or "").split(" / ") if f.strip()][:3]}


class RealtimeHub:
    def __init__(self, agent, lock: Optional[threading.RLock] = None, bus: Optional[EventBus] = None,
                 interval: float = WATCH_INTERVAL, threshold: float = ALERT_THRESHOLD, workers: int = WORKERS):
        self.agent = agent
        self.lock = lock or threading.RLock()
        self.bus = bus or EventBus(agent.storage.conn, self.lock)
        from ..tool_agent.memory import SCHEMA as MEMORY_SCHEMA

        with self.lock:  # 방문자 장기 기억 표 (찾던 조건) - 아직 없으면 만듦
            agent.storage.conn.executescript(MEMORY_SCHEMA)
        self.interval, self.threshold, self.n_workers = interval, threshold, workers
        self.jobs: queue.Queue = queue.Queue()
        self.stats = Stats()
        self._snapshot: dict[str, tuple] = self._take_snapshot()
        self._folder_sig = self._folder_signature()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.on_sync = None  # 웹 앱이 폴더를 다시 읽을 때 쓰는 함수 (오류 처리 포함)

    # ------------------------------------------------------------------ 시작/중지
    def start(self) -> "RealtimeHub":
        if self._threads:
            return self
        self._stop.clear()
        for i in range(self.n_workers):
            t = threading.Thread(target=self._worker, name=f"realtime-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        if self.interval > 0:
            t = threading.Thread(target=self._watch_loop, name="realtime-watcher", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        for _ in range(self.n_workers):
            self.jobs.put(None)
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ------------------------------------------------------------------ 변경 감지 (생산자)
    def _take_snapshot(self) -> dict[str, tuple]:
        return {l.listing_id: (l.price_manwon, round(l.area_m2), l.status) for l in self.agent.listings()}

    def detect_changes(self, source: str = "관리자") -> list[Event]:
        """지난번과 지금의 매물 목록을 비교해 이벤트 발행 + 새 매물·가격 인하는 매칭 작업으로 큐에 넣음.
        (반드시 매물 목록을 바꾼 직후, 잠금 안에서 부름)"""
        detected_at = time.perf_counter()
        before, after = self._snapshot, self._take_snapshot()
        self._snapshot = after
        self._folder_sig = self._folder_signature()  # 관리자 화면이 쓴 파일(웹입력 등)을 폴더 감시가 또 읽지 않게
        events = []
        for lid, now in after.items():
            old = before.get(lid)
            card = listing_card(self.agent, lid)
            if card is None:
                continue
            if old is None:
                topic, extra = "listing.added", {}
            elif now[0] < old[0]:
                topic, extra = "listing.price_drop", {"old_price_manwon": old[0], "old_price": format_price(old[0])}
            elif now != old:
                topic, extra = "listing.changed", {}
            else:
                continue
            ev = self.bus.publish(topic, {"listing": card, "source": source, **extra})
            if ev:
                events.append(ev)
                self.stats.detected += 1
                if topic in {"listing.added", "listing.price_drop"}:
                    self.jobs.put({"event": ev, "listing_id": lid, "detected_at": detected_at})
        for lid in before.keys() - after.keys():
            ev = self.bus.publish("listing.removed", {"listing_id": lid, "source": source})
            if ev:
                events.append(ev)
                self.stats.detected += 1
        return events

    def _folder_signature(self) -> tuple:
        try:
            files = self.agent.collector.iter_files()
        except (FileNotFoundError, OSError):
            return ()
        sig = []
        for p in files:
            try:
                st = Path(p).stat()
                sig.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                continue
        return tuple(sig)

    def check_folder(self) -> list[Event]:
        """매물 폴더 파일이 바뀌었으면 다시 읽고 변경 이벤트 발행."""
        sig = self._folder_signature()
        if sig == self._folder_sig:
            return []
        self._folder_sig = sig
        with self.lock:
            if self.on_sync:
                self.on_sync()
            else:
                self.agent.sync()
            self.stats.syncs += 1
            events = self.detect_changes("매물 폴더")
            if events:
                self.bus.publish("system.sync", {"listings": len(self.agent.listings()), "changes": len(events),
                                                 "source": "매물 폴더"})
        return events

    def _watch_loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check_folder()
            except Exception as e:  # 감시가 멈추지 않도록
                log.warning("매물 폴더 감시 오류: %s", e)

    # ------------------------------------------------------------------ 매칭 (작업 큐 소비자)
    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.match(job)
                self.stats.jobs_done += 1
            except Exception as e:  # 작업 하나가 실패해도 워커는 계속
                self.stats.job_errors += 1
                log.warning("매칭 작업 실패: %s", e)
            finally:
                self.jobs.task_done()

    def owners_with_conditions(self, limit: int = 500) -> list[tuple[str, dict]]:
        with self.lock:
            rows = self.agent.storage.conn.execute(
                "SELECT owner, data FROM agent_memory ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            mem = json.loads(r["data"])
            if mem.get("searches"):
                out.append((r["owner"], mem))
        return out

    def match(self, job: dict) -> list[Event]:
        """새 매물/가격 인하 하나를 방문자들이 찾던 조건과 대조해 맞으면 그 방문자에게 알림."""
        from ..models import CustomerRequirement
        from ..recommender import RecommendOptions, hard_filter, score_listing
        from ..tool_agent.tools import requirement_from_args

        lid = job["listing_id"]
        ev: Event = job["event"]
        kind = "가격 인하" if ev.topic == "listing.price_drop" else "새 매물"
        sent = []
        with self.lock:
            listing = self.agent.find_listing(lid)
            if listing is None:
                return []
            owners = self.owners_with_conditions()
            for owner, mem in owners:
                if lid in mem.get("rejected", {}):
                    continue
                for search in reversed(mem["searches"][-3:]):  # 최근 조건 3개 중 하나라도 맞으면
                    req: CustomerRequirement = requirement_from_args(self.agent, search.get("args") or {})
                    opt = RecommendOptions(market_price_per_m2=self.agent.market_price)
                    if hard_filter(listing, req, opt):
                        continue
                    scored = score_listing(listing, req, opt)
                    if scored.score < self.threshold:
                        continue
                    card = ev.payload["listing"]
                    msg = (f"찾으시던 조건({search['조건']})에 맞는 {kind}: {card['name']} · {card['price']}"
                           + (f" (← {ev.payload['old_price']})" if kind == "가격 인하" else "")
                           + f" · {card['area_pyeong']:,}평")
                    alert = self.bus.publish(
                        "alert.match",
                        {"kind": kind, "condition": search["조건"], "score": round(scored.score, 1),
                         "reasons": scored.reasons[:3], "message": msg, "listing": card,
                         "latency_ms": round((time.perf_counter() - job["detected_at"]) * 1000, 1)},
                        owner=owner, dedup=f"{owner}|{lid}|{kind}|{listing.price_manwon}")
                    if alert:
                        sent.append(alert)
                        self.stats.alerts += 1
                        self.stats.match_ms.append(alert.payload["latency_ms"])
                    break
        return sent

    def drain(self, timeout: float = 5.0) -> None:
        """큐에 쌓인 작업이 끝날 때까지 기다림 (테스트·명령줄용). 워커가 없으면 여기서 직접 처리."""
        if not self.running:
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    return
                if job is not None:
                    self.match(job)
                    self.stats.jobs_done += 1
                self.jobs.task_done()
        deadline = time.time() + timeout
        while self.jobs.unfinished_tasks and time.time() < deadline:
            time.sleep(0.02)
