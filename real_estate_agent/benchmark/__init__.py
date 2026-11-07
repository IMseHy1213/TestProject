"""v1.3 벤치마크: 정해진 문제지(benchmarks/land_consult_v1.json)로 상담 품질·안전·속도·비용·동시 처리량을 매번 같은 방법으로 측정.

    python main.py benchmark                    → 모든 방식 실행, 표 출력, logs/benchmark_*.md·json 저장
    python main.py benchmark --check            → 기준선(benchmarks/baseline.json)보다 나빠졌으면 실패(종료 코드 1)
    python main.py benchmark --save-baseline    → 지금 결과를 새 기준선으로

정답(채점 기준)은 Agent 코드를 쓰지 않고 매물 원자료를 질문의 뜻대로 직접 걸러서 계산합니다 (oracle).
사무실 규칙만 같이 씀: 용도별 건축 가능 판단(land_use), 용도가 있으면 맹지 제외, 예산 10%·면적 20% 여유.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import PROJECT_ROOT
from ..land_use import check_use
from ..recommender import NOT_FOUND_MESSAGE

DATASET = PROJECT_ROOT / "benchmarks" / "land_consult_v1.json"
BASELINE = PROJECT_ROOT / "benchmarks" / "baseline.json"
CONFIGS = ("single", "multi", "distributed", "mcp")
CONFIG_LABEL = {"single": "상담원 1명", "multi": "상담팀", "distributed": "분산 상담팀 (A2A)", "mcp": "MCP 도구",
                "workflow": "Workflow", "plan_execute": "Plan-and-Execute", "langgraph": "LangGraph"}
BUDGET_TOL, AREA_TOL = 0.10, 0.20
_KEY = {"price_asc": lambda l: l.price_manwon, "area_desc": lambda l: -l.area_pyeong,
        "ppp_asc": lambda l: l.price_per_pyeong}


# ---------------------------------------------------------------------- 정답 (oracle)
def oracle(agent, spec: dict) -> list:
    """질문의 뜻(지역·용도·예산·면적·뺄 지역)으로 원자료를 직접 걸러 정렬."""
    out = []
    for l in agent.listings():
        if l.status in {"계약완료", "보류"}:
            continue
        if spec.get("regions") and not any(r in l.address for r in spec["regions"]):
            continue
        if spec.get("exclude_regions") and any(r in l.address for r in spec["exclude_regions"]):
            continue
        if spec.get("purpose"):
            use = check_use(l, spec["purpose"])
            if (use is not None and not use.ok) or l.is_blind_land:
                continue
        if spec.get("max_price") and l.price_manwon > spec["max_price"] * (1 + BUDGET_TOL):
            continue
        if spec.get("min_area") and l.area_pyeong < spec["min_area"] * (1 - AREA_TOL):
            continue
        if spec.get("max_area") and l.area_pyeong > spec["max_area"] * (1 + AREA_TOL * 2):  # 사무실 규칙: 위쪽은 40% 여유
            continue
        out.append(l)
    if spec.get("sort"):
        out.sort(key=_KEY[spec["sort"]])
    return out[: spec["k"]] if spec.get("k") else out


# ---------------------------------------------------------------------- 채점
def check_turn(agent, result, expect: dict, history_ids: list[list[str]], evidence: list) -> tuple[list[str], dict]:
    """실패한 항목 목록과 검색 지표(정밀도·재현율)."""
    from ..reliability.guardrails import Evidence, check_answer, contact_notice

    fails: list[str] = []
    metrics: dict[str, float] = {}
    got = [agent.find_listing(i) for i in result.listing_ids]
    got = [l for l in got if l is not None]
    ids = [l.listing_id for l in got]
    answer = result.answer or ""
    if not answer.strip():
        fails.append("빈 답변")
    problems = check_answer(answer, Evidence.from_observations(evidence), agent)
    if problems:
        fails.append("근거·개인정보: " + "; ".join(problems))
    if "count" in expect and len(ids) != expect["count"]:
        fails.append(f"매물 {expect['count']}건이어야 함 ({len(ids)}건)")
    if "oracle" in expect:
        gold = oracle(agent, expect["oracle"])
        gold_ids = {l.listing_id for l in gold}
        wrong = [i for i in ids if i not in gold_ids]
        if wrong:
            fails.append(f"조건에 안 맞는 매물 {len(wrong)}건: {', '.join(wrong[:3])}")
        if gold and not ids:
            fails.append(f"맞는 매물 {len(gold)}건이 있는데 못 찾음")
        if not gold and (ids or NOT_FOUND_MESSAGE not in answer and "없습니다" not in answer):
            fails.append("맞는 매물이 없는데 '없다'고 하지 않음")
        if ids:
            metrics["precision"] = (len(ids) - len(wrong)) / len(ids)
        if gold:
            metrics["recall"] = len(set(ids) & gold_ids) / min(len(gold), max(len(ids), 1) if "count" in expect else 5)
    if "gold" in expect:
        spec = expect["gold"]
        gold = oracle(agent, spec)
        key = _KEY[spec["sort"]]
        allowed = {l.listing_id for l in oracle(agent, {k: v for k, v in spec.items() if k not in ("k", "sort")})}
        if [key(l) for l in got] != [key(l) for l in gold] or any(i not in allowed for i in ids):
            fails.append(f"정답 순서와 다름 (정답 {', '.join(l.listing_id for l in gold)} / 받은 {', '.join(ids)})")
        metrics["precision"] = sum(1 for i in ids if i in allowed) / len(ids) if ids else 0.0
        metrics["recall"] = len({key(l) for l in got} & {key(l) for l in gold}) / max(1, len(gold))
    if "gold_first" in expect:
        gold = oracle(agent, expect["gold_first"])
        if not got or _KEY[expect["gold_first"]["sort"]](got[0]) != _KEY[expect["gold_first"]["sort"]](gold[0]):
            fails.append(f"첫 매물이 정답({gold[0].listing_id})이 아님 ({ids[:1]})")
    if "pick" in expect:
        prev = [agent.find_listing(i) for i in (history_ids[-1] if history_ids else [])]
        prev = [l for l in prev if l is not None]
        key = {"area_max": lambda l: -l.area_pyeong, "price_min": lambda l: l.price_manwon,
               "price_max": lambda l: -l.price_manwon}[expect["pick"]]
        want = min(prev, key=key) if prev else None
        if want is None or [key(l) for l in got] != [key(want)]:
            fails.append(f"앞서 보여준 것 중 골라야 함 (정답 {want.listing_id if want else '-'} / 받은 {ids})")
    if expect.get("subset_prev2"):
        pool = set(history_ids[-2]) if len(history_ids) >= 2 else set()
        if not ids or any(i not in pool for i in ids):
            fails.append("처음 보여준 매물 안에서 골라야 함")
    if "buildable" in expect:
        b = expect["buildable"]
        use = check_use(agent.find_listing(b["listing"]), b["purpose"])
        want = "가능" if use and use.ok else "어려"
        if want not in answer or (want == "가능" and "어려" in answer.split("가능")[0][-6:]):
            fails.append(f"건축 판단이 '{want}' 이어야 함")
    for want in expect.get("includes", []):
        if want not in ids:
            fails.append(f"매물 {want} 가 카드에 있어야 함")
    if expect.get("not_found") and (ids or "없습니다" not in answer):
        fails.append("없다고 답해야 함")
    for s in expect.get("answer_has", []):
        if s not in answer:
            fails.append(f"답에 '{s}' 가 있어야 함")
    for s in expect.get("answer_not", []):
        if s in answer:
            fails.append(f"답에 '{s}' 가 없어야 함")
    if expect.get("contact_notice") and contact_notice() not in answer:
        fails.append("연락처·지번 안내 문장이 없음")
    return fails, metrics


# ---------------------------------------------------------------------- 실행 방식 (설정)
class _Runners:
    """방식별 Agent 만들기. 분산(A2A)은 같은 프로세스 안의 A2A 서버(HTTP 앱)로, MCP 는 같은 프로세스 MCP 서버로."""

    def __init__(self, agent):
        self.agent = agent
        self._a2a = None

    def make(self, config: str):
        from ..tool_agent import AgentContext, ToolAgent
        from ..tool_agent.multi import MultiAgent

        ctx = AgentContext()
        if config == "single":
            return ToolAgent(self.agent, ctx=ctx, log_dir=""), None
        if config == "multi":
            return MultiAgent(self.agent, ctx=ctx, log_dir=""), None
        if config == "distributed":
            from ..a2a.distributed import DistributedMultiAgent

            return DistributedMultiAgent(self.agent, clients=self._clients(), ctx=ctx, log_dir=""), None
        if config == "workflow":
            from ..architectures import WorkflowAgent

            return WorkflowAgent(self.agent, ctx=ctx, log_dir=""), None
        if config == "plan_execute":
            from ..architectures import PlanExecuteAgent

            return PlanExecuteAgent(self.agent, ctx=ctx, log_dir=""), None
        if config == "langgraph":
            from ..framework.langgraph_agent import LangGraphAgent

            return LangGraphAgent(self.agent, ctx=ctx, log_dir=""), None
        if config == "mcp":
            from ..mcp_server import create_mcp_server
            from ..tool_agent.mcp_client import McpToolSource

            src = McpToolSource(create_mcp_server(self.agent)).start()
            return ToolAgent(self.agent, ctx=ctx, log_dir="", tool_source=src), src.close
        raise ValueError(config)

    def _clients(self):
        if self._a2a is None:
            from fastapi.testclient import TestClient

            from ..a2a.client import A2AClient
            from ..a2a.server import create_a2a_app
            from ..tool_agent.multi import ANALYST, SEARCHER

            self._a2a = {who: A2AClient("http://testserver/", http=TestClient(create_a2a_app(role, agent=self.agent,
                                                                                         token="")))
                         for who, role in ((SEARCHER, "search"), (ANALYST, "analysis"))}
        return self._a2a


def available_configs() -> list[str]:
    out = ["single", "multi"]
    try:
        import fastapi  # noqa: F401

        out.append("distributed")
    except ImportError:
        pass
    try:
        import mcp  # noqa: F401

        out.append("mcp")
    except ImportError:
        pass
    return out


def load_dataset(path: Optional[Path] = None) -> dict:
    return json.loads(Path(path or DATASET).read_text(encoding="utf-8"))


def _llm_usage(trace_id: str) -> tuple[int, int]:
    from ..observability import TRACER

    spans = [s for s in TRACER.trace(trace_id) if s.name.startswith("llm.")] if trace_id else []
    return len(spans), sum(int(s.attrs.get("input_tokens") or 0) + int(s.attrs.get("output_tokens") or 0)
                           for s in spans)


def run_benchmark(agent, configs: Optional[list[str]] = None, dataset: Optional[dict] = None,
                  progress: Optional[Callable[[str], None]] = None) -> dict:
    from ..reliability.chaos import _observations
    from ..a2a.client import A2AClient

    dataset = dataset or load_dataset()
    configs = configs or available_configs()
    runners = _Runners(agent)
    report: dict[str, Any] = {"dataset": dataset["name"], "cases": len(dataset["cases"]),
                              "turns": sum(len(c["turns"]) for c in dataset["cases"]), "configs": {},
                              "policy": agent.llm.mode(), "when": time.strftime("%Y-%m-%d %H:%M")}
    for config in configs:
        if progress:
            progress(CONFIG_LABEL.get(config, config))
        A2AClient.reset_all()
        rows = []
        for case in dataset["cases"]:
            runner, close = runners.make(config)
            history: list = []
            shown: list[list[str]] = []
            try:
                for i, turn in enumerate(case["turns"]):
                    t0 = time.perf_counter()
                    try:
                        r, history = runner.run(turn["q"], history)
                        fails, m = check_turn(agent, r, turn.get("expect", {}), shown, _observations(runner, history))
                        llm_calls, tokens = _llm_usage(r.trace_id)
                        tools = sum(1 for s in r.steps if s.kind == "tool" and not (s.tool or "").startswith("delegate_"))
                        decisions = getattr(r, "decisions", 0)
                        policy = r.policy
                        shown.append(list(r.listing_ids))
                    except Exception as e:  # noqa: BLE001 - 처리 못 한 오류도 실패로 기록
                        fails, m, llm_calls, tokens, tools, policy = [f"오류 {type(e).__name__}: {e}"], {}, 0, 0, 0, "error"
                        decisions = 0
                        shown.append([])
                    rows.append({"case": case["id"], "category": case["category"], "turn": i + 1, "q": turn["q"],
                                 "ok": not fails, "fails": fails, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                 "tools": tools, "llm_calls": llm_calls, "tokens": tokens, "policy": policy,
                                 "decisions": decisions, **m})
            finally:
                if close:
                    close()
        report["configs"][config] = summarize(rows)
    return report


def summarize(rows: list[dict]) -> dict:
    cases: dict[str, bool] = {}
    cats: dict[str, list[bool]] = {}
    for r in rows:
        cases[r["case"]] = cases.get(r["case"], True) and r["ok"]
        cats.setdefault(r["category"], []).append(r["ok"])
    ms = sorted(r["ms"] for r in rows)
    prec = [r["precision"] for r in rows if "precision" in r]
    rec = [r["recall"] for r in rows if "recall" in r]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {
        "turn_pass": pct([r["ok"] for r in rows]), "case_pass": pct(list(cases.values())),
        "passed_cases": sorted(k for k, v in cases.items() if v), "failed_cases": sorted(k for k, v in cases.items() if not v),
        "by_category": {k: pct(v) for k, v in cats.items()},
        "precision": round(100 * statistics.mean(prec), 1) if prec else None,
        "recall": round(100 * statistics.mean(rec), 1) if rec else None,
        "p50_ms": ms[len(ms) // 2], "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))],
        "tools_per_turn": round(statistics.mean(r["tools"] for r in rows), 2),
        "decisions_per_turn": round(statistics.mean(r.get("decisions", 0) for r in rows), 2),
        "llm_calls_per_turn": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "tokens": sum(r["tokens"] for r in rows),
        "policies": sorted({r["policy"] for r in rows}),
        "failures": [{"case": r["case"], "turn": r["turn"], "q": r["q"], "fails": r["fails"]} for r in rows if not r["ok"]],
    }


# ---------------------------------------------------------------------- 동시 처리량 (부하)
LOAD_QUESTIONS = ["음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "양평에 전원주택 지을 땅 3억 이하로 찾아줘",
                  "여주 공장 부지 보여줘", "넓은 땅 순서로 5개 보여줘", "2억 이하 땅 있어?"]


def run_load(agent, levels: tuple[int, ...] = (1, 4, 8, 16), requests_per_level: int = 32,
             ai_delay: float = 0.0, limit: Optional[int] = None) -> list[dict]:
    """동시 손님 수를 늘려 가며 처리량(초당 답변 수)·응답 시간을 잼.
    ai_delay > 0: AI 응답이 그만큼 걸린다고 가정한 가짜 AI (실제 GPT 는 네트워크 대기가 대부분)."""
    import random
    import threading

    from ..reliability.chaos import ChaosLLM, _router
    from ..reliability.limits import Bulkhead
    from ..tool_agent import AgentContext, ToolAgent

    ToolAgent(agent, ctx=AgentContext(), log_dir="").run(LOAD_QUESTIONS[0], [])  # 준비 운동 (첫 실행의 불러오기 시간 제외)
    out = []
    for c in levels:
        bulkhead = Bulkhead(limit=limit, wait=120)  # 운영 설정(동시 상담 수 제한) 그대로
        lat: list[float] = []
        errors = [0]
        mu = threading.Lock()
        per = max(1, requests_per_level // c)

        def visitor(v: int) -> None:
            ctx = AgentContext()
            llm = _router(ChaosLLM(agent, ctx, "llm_slow", 1.0, random.Random(v), ai_delay)) if ai_delay else None
            runner = ToolAgent(agent, llm=llm, ctx=ctx, log_dir="")
            for k in range(per):
                t0 = time.perf_counter()
                if not bulkhead.acquire():
                    with mu:
                        errors[0] += 1
                    continue
                try:
                    runner.run(LOAD_QUESTIONS[(v + k) % len(LOAD_QUESTIONS)], [])
                except Exception:  # noqa: BLE001
                    with mu:
                        errors[0] += 1
                finally:
                    bulkhead.release()
                with mu:
                    lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=visitor, args=(v,)) for v in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        lat.sort()
        out.append({"concurrency": c, "requests": len(lat), "errors": errors[0], "seconds": round(wall, 2),
                    "throughput": round(len(lat) / wall, 2), "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
                    "p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1) if lat else None,
                    "limit": bulkhead.limit, "ai_delay": ai_delay})
    return out


# ---------------------------------------------------------------------- 기준선 비교
def compare_baseline(report: dict, baseline: dict) -> list[str]:
    """기준선보다 나빠진 점 (통과하던 문제가 실패, 통과율 하락)."""
    out = []
    for config, now in report["configs"].items():
        base = (baseline.get("configs") or {}).get(config)
        if not base:
            continue
        newly = sorted(set(base.get("passed_cases", [])) & set(now["failed_cases"]))
        if newly:
            out.append(f"[{CONFIG_LABEL.get(config, config)}] 통과하던 문제가 실패: {', '.join(newly)}")
        if now["turn_pass"] is not None and base.get("turn_pass") is not None and now["turn_pass"] < base["turn_pass"]:
            out.append(f"[{CONFIG_LABEL.get(config, config)}] 질문 통과율 {base['turn_pass']}% → {now['turn_pass']}%")
    return out


def baseline_of(report: dict) -> dict:
    return {"dataset": report["dataset"], "when": report["when"], "policy": report["policy"],
            "configs": {k: {key: v[key] for key in ("turn_pass", "case_pass", "passed_cases", "failed_cases",
                                                    "precision", "recall")}
                        for k, v in report["configs"].items()}}


def markdown(report: dict, load: Optional[list[dict]] = None, load_ai: Optional[list[dict]] = None,
             regressions: Optional[list[str]] = None) -> str:
    cfgs = report["configs"]
    names = list(cfgs)
    lines = [f"# v1.3 벤치마크 — {report['dataset']}", "",
             f"문제 {report['cases']}개 (질문 {report['turns']}개) · 답을 만든 쪽: {report['policy']} · {report['when']}", "",
             "| 항목 | " + " | ".join(CONFIG_LABEL.get(n, n) for n in names) + " |", "|---|" + "---|" * len(names)]
    rows = [("문제 통과율", "case_pass", "%"), ("질문 통과율", "turn_pass", "%"), ("정밀도 (조건에 맞는 매물 비율)", "precision", "%"),
            ("재현율 (있는 매물을 찾은 비율)", "recall", "%"), ("응답 시간 p50", "p50_ms", "ms"), ("응답 시간 p95", "p95_ms", "ms"),
            ("질문당 도구 호출", "tools_per_turn", ""), ("질문당 AI 호출", "llm_calls_per_turn", ""), ("AI 토큰 합계", "tokens", "")]
    for label, key, unit in rows:
        lines.append(f"| {label} | " + " | ".join(f"{cfgs[n][key]}{unit}" if cfgs[n][key] is not None else "-"
                                                for n in names) + " |")
    cats = sorted({c for n in names for c in cfgs[n]["by_category"]})
    lines += ["", "## 분야별 통과율", "", "| 분야 | " + " | ".join(CONFIG_LABEL.get(n, n) for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(f"{cfgs[n]['by_category'].get(c, '-')}%" for n in names) + " |")
    fails = cfgs[names[0]]["failures"]
    lines += ["", f"## 틀린 문제 ({CONFIG_LABEL.get(names[0], names[0])})", ""]
    lines += [f"- {f['case']}-{f['turn']} \"{f['q']}\" → {'; '.join(f['fails'])}" for f in fails] or ["- 없음"]
    ai_title = f"AI 응답 {load_ai[0]['ai_delay']}초 가정 (가짜 AI)" if load_ai else ""
    for title, rows_ in (("규칙 기반 (AI 없음)", load), (ai_title, load_ai)):
        if rows_:
            lines += ["", f"## 동시 처리량 — {title}", "", "| 동시 손님 | 답변 수 | 초당 답변 | p50 | p95 | 실패 | 동시 상담 제한 |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {r['concurrency']} | {r['requests']} | {r['throughput']} | {r['p50_ms']}ms | {r['p95_ms']}ms | "
                      f"{r['errors']} | {r['limit']} |" for r in rows_]
    if regressions is not None:
        lines += ["", "## 기준선 비교", ""] + ([f"- ❌ {r}" for r in regressions] or ["- ✅ 기준선보다 나빠진 점 없음"])
    lines += ["", "※ 벤치마크는 연습용 샘플 매물(sample_data)로 돌기 때문에, 보고서에서는 매물번호를 '(샘플 매물)' 로 가립니다 "
                  "(실제 사무실 매물번호가 아님). 원래 번호는 logs 의 .json 에 있습니다."]
    from ..listing_numbers import NUMBER

    return NUMBER.sub("(샘플 매물)", "\n".join(lines))
