"""관리자 '관측' 화면과 python main.py observe / trace 가 쓰는 조회 (저장된 span 기준, 프로그램을 껐다 켜도 남음)."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import nullcontext
from typing import Any, Optional

ROOTS = ("agent.run", "a2a.serve")


def _rows(conn: sqlite3.Connection, lock, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    try:
        with lock or nullcontext():
            cur = conn.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.OperationalError:  # 아직 span 표가 없음
        return []


def _load(row: dict) -> dict:
    row = dict(row)
    row["attrs"] = json.loads(row.get("attrs") or "{}")
    row["events"] = json.loads(row.get("events") or "[]")
    return row


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * q))], 1)


def summary(conn, lock=None, hours: float = 24) -> dict[str, Any]:
    since = time.time_ns() - int(hours * 3600 * 1e9)
    spans = [_load(r) for r in _rows(conn, lock, "SELECT * FROM obs_spans WHERE start_ns >= ? ORDER BY start_ns",
                                     (since,))]
    roots = [s for s in spans if not s["parent_id"] and s["name"] in ROOTS]
    questions = [s for s in roots if s["name"] == "agent.run"] + \
                [s for s in roots if s["name"] == "a2a.serve"]
    lat = [s["dur_ms"] for s in questions]
    ok = [s for s in questions if s["status"] == "ok"]
    ai = [s for s in questions if s["attrs"].get("policy") not in (None, "rule", "error")]
    llm = [s for s in spans if s["name"].startswith("llm.")]
    tools = [s for s in spans if s["name"].startswith("tool.") and not s["name"].startswith("tool.delegate_")]
    events: dict[str, int] = {}
    for s in spans:
        for e in s["events"]:
            if e["name"].startswith("reliability."):
                k = e["name"].split(".", 1)[1]
                events[k] = events.get(k, 0) + 1
    modes: dict[str, int] = {}
    for s in questions:
        m = s["attrs"].get("mode") or ("a2a" if s["name"] == "a2a.serve" else "single")
        modes[m] = modes.get(m, 0) + 1
    by_tool: dict[str, list] = {}
    for s in tools:
        by_tool.setdefault(s["name"][5:], []).append(s)
    tool_rows = sorted(({"tool": k, "calls": len(v), "errors": sum(1 for x in v if x["status"] == "error"),
                         "p50_ms": _pct([x["dur_ms"] for x in v], 0.5), "p95_ms": _pct([x["dur_ms"] for x in v], 0.95)}
                        for k, v in by_tool.items()), key=lambda r: -r["calls"])
    llm_by: dict[str, dict] = {}
    for s in llm:
        p = s["attrs"].get("provider", "?")
        row = llm_by.setdefault(p, {"provider": p, "calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
                                    "cost_usd": 0.0, "lat": []})
        row["calls"] += 1
        row["errors"] += s["status"] == "error"
        row["input_tokens"] += int(s["attrs"].get("input_tokens") or 0)
        row["output_tokens"] += int(s["attrs"].get("output_tokens") or 0)
        row["cost_usd"] += float(s["attrs"].get("cost_usd") or 0)
        row["lat"].append(s["dur_ms"])
    for row in llm_by.values():
        row["p50_ms"], row["p95_ms"] = _pct(row["lat"], 0.5), _pct(row.pop("lat"), 0.95)
        row["cost_usd"] = round(row["cost_usd"], 4)
    return {
        "hours": hours, "questions": len(questions),
        "success_rate": round(100 * len(ok) / len(questions), 1) if questions else None,
        "ai_share": round(100 * len(ai) / len(questions), 1) if questions else None,
        "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95), "max_ms": round(max(lat), 1) if lat else None,
        "modes": modes, "tools": tool_rows, "llm": list(llm_by.values()),
        "tool_error_rate": round(100 * sum(r["errors"] for r in tool_rows) / max(1, len(tools)), 1) if tools else None,
        "reliability": events,
        "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in llm_by.values()),
        "cost_usd": round(sum(r["cost_usd"] for r in llm_by.values()), 4),
        "spans": len(spans),
    }


def recent(conn, lock=None, limit: int = 30, only_errors: bool = False, slow_ms: float = 0) -> list[dict]:
    rows = _rows(conn, lock, "SELECT * FROM obs_spans WHERE (parent_id IS NULL OR parent_id = '') AND name IN (?, ?) "
                             "AND (? = 0 OR status = 'error') AND dur_ms >= ? ORDER BY start_ns DESC LIMIT ?",
                 (*ROOTS, int(only_errors), slow_ms, limit))
    out = []
    for r in map(_load, rows):
        n = _rows(conn, lock, "SELECT COUNT(*) AS n, COUNT(DISTINCT service) AS services FROM obs_spans "
                              "WHERE trace_id = ?", (r["trace_id"],))[0]
        out.append({"trace_id": r["trace_id"], "time": time.strftime("%m-%d %H:%M:%S", time.localtime(r["start_ns"] / 1e9)),
                    "question": r["attrs"].get("question", ""), "mode": r["attrs"].get("mode", ""),
                    "policy": r["attrs"].get("policy", ""), "stopped": r["attrs"].get("stopped", ""),
                    "ms": round(r["dur_ms"], 1), "status": r["status"], "spans": n["n"], "services": n["services"],
                    "guards": r["attrs"].get("guards") or 0, "name": r["name"]})
    return out


def trace_tree(conn, lock=None, trace_id: str = "") -> list[dict]:
    """한 trace 의 span 들을 부모-자식 순서로 + 폭포(waterfall) 그림용 위치(%)."""
    spans = [_load(r) for r in _rows(conn, lock, "SELECT * FROM obs_spans WHERE trace_id = ? ORDER BY start_ns",
                                     (trace_id,))]
    return layout(spans)


def layout(spans: list[dict]) -> list[dict]:
    if not spans:
        return []
    start = min(s["start_ns"] for s in spans)
    end = max(s["start_ns"] + s["dur_ms"] * 1e6 for s in spans)
    total = max(end - start, 1)
    ids = {s["span_id"] for s in spans}
    children: dict[str, list] = {}
    for s in spans:
        children.setdefault(s["parent_id"] if s["parent_id"] in ids else "", []).append(s)
    out: list[dict] = []

    def walk(parent: str, depth: int) -> None:
        for s in sorted(children.get(parent, []), key=lambda x: x["start_ns"]):
            out.append({**s, "depth": depth, "offset_pct": round(100 * (s["start_ns"] - start) / total, 2),
                        "width_pct": max(0.4, round(100 * s["dur_ms"] * 1e6 / total, 2)),
                        "offset_ms": round((s["start_ns"] - start) / 1e6, 1)})
            walk(s["span_id"], depth + 1)

    walk("", 0)
    return out


def text_waterfall(rows: list[dict], width: int = 40) -> str:
    """터미널용 폭포 그림."""
    lines = []
    for r in rows:
        a = int(r["offset_pct"] / 100 * width)
        b = max(1, int(r["width_pct"] / 100 * width))
        bar = " " * a + "█" * min(b, width - a)
        label = r["attrs"].get("agent") or r["attrs"].get("role") or r["attrs"].get("provider") or ""
        mark = " ✘" if r["status"] == "error" else ""
        name = _fit("  " * r["depth"] + r["name"] + (f" [{label}]" if label else ""), 46)
        lines.append(f"{name} |{bar:<{width}}| {r['dur_ms']:>8.1f}ms [{r['service']}]{mark}")
        for e in r["events"]:
            lines.append(" " * (2 * r["depth"] + 4) + f"· {e['name']}")
    return "\n".join(lines)


def _fit(text: str, width: int) -> str:
    """터미널 칸 수에 맞춰 자르고 채움 (한글은 2칸)."""
    import unicodedata

    out, used = "", 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + w > width:
            break
        out, used = out + ch, used + w
    return out + " " * (width - used)
