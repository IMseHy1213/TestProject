"""v1.0 상태 점검: 살아 있는지(/healthz) · 손님을 받을 준비가 됐는지(/readyz).

준비 상태
  ok        모두 정상
  degraded  손님은 받을 수 있지만 일부 기능이 대체 모드 (AI 미연결 → 규칙 기반, 원격 Agent 꺼짐 → 이 서버가 대신 등)
  fail      손님을 받을 수 없음 (DB 오류, 매물 0건) → HTTP 503 (로드밸런서·감시 프로그램이 알아챔)
"""

from __future__ import annotations

import time
from typing import Any, Optional


def readiness(agent, lock=None, hub=None, service=None) -> dict[str, Any]:
    from contextlib import nullcontext

    from .stats import STATS

    checks: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    try:
        with lock or nullcontext():
            agent.storage.conn.execute("SELECT 1").fetchone()
        checks["database"] = {"ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as e:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": type(e).__name__}
    n = len(agent.listings())
    checks["listings"] = {"ok": n > 0, "count": n}
    mode = agent.llm.mode()
    checks["ai"] = {"ok": mode == "ai", "mode": mode, "fallback": "규칙 기반" if mode != "ai" else None,
                    "critical": False}
    if hub is not None:
        checks["realtime"] = {"ok": bool(hub.running), "critical": False}
    try:
        from ..a2a.client import A2AClient
        from ..a2a.distributed import ANALYSIS_URL, SEARCH_URL

        open_ = [u for u in (SEARCH_URL, ANALYSIS_URL) if A2AClient.circuit_open(u)]
        checks["a2a_remotes"] = {"ok": not open_, "circuit_open": open_, "critical": False}
    except Exception:  # noqa: BLE001 - A2A 를 안 쓰는 환경
        pass
    if service is not None:
        b = service.bulkhead
        checks["capacity"] = {"ok": b.active < b.limit, "active": b.active, "limit": b.limit, "waiting": b.waiting,
                              "critical": False}
    critical_fail = [k for k, c in checks.items() if not c["ok"] and c.get("critical", True)]
    soft_fail = [k for k, c in checks.items() if not c["ok"] and not c.get("critical", True)]
    status = "fail" if critical_fail else ("degraded" if soft_fail else "ok")
    for c in checks.values():
        c.pop("critical", None)
    return {"status": status, "checks": checks, "reliability": STATS.as_dict()}


def http_status(report: dict[str, Any]) -> int:
    return 503 if report["status"] == "fail" else 200
