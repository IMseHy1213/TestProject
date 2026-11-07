"""v0.4 Architecture Study + v0.5 Framework 비교: 측정하고 보고서(docs/*.md)를 만듦.

    python main.py architecture-study   → docs/v0.4_architecture_study.md, docs/v0.5_framework.md (+ logs 사본)

측정은 v1.3 벤치마크 문제지(정답은 원자료로 따로 계산)와 같은 채점기를 씀. 연습용 샘플 매물 + 임시 DB 로 돌림.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from .architectures import ARCHITECTURES, PlanExecuteAgent, WorkflowAgent
from .benchmark import _Runners, run_benchmark
from .listing_numbers import NUMBER

AI_SECONDS = 1.5  # 예상 응답 시간 계산용 가정: AI 판단 1번에 1.5초 (GPT 응답은 보통 1~3초)

# 결과가 예상과 다를 때(검색 결과 0~1건인데 비교를 요청 등) 구조가 어떻게 버티는지 보는 질문
ADAPTIVITY_PROBES = [
    "음성군 3000평 넘는 땅 2개 비교해줘",          # 결과 0건인데 비교 요청
    "양평에 공장 지을 땅 1억 이하 비교해줘",          # 결과 0건
    "여주에 근린생활 지을 땅 중 싼 거 3개 비교해줘",   # 결과가 적을 수 있음
    "제주도 땅 중 창고 지을 수 있는지 확인해줘",       # 결과 0건인데 건축 확인 요청
]


def _loc(*objs) -> int:
    n = 0
    for o in objs:
        src = inspect.getsource(o)
        n += sum(1 for line in src.splitlines() if line.strip() and not line.strip().startswith(("#", '"""')))
    return n


def code_size() -> dict[str, int]:
    """흐름 제어 코드 줄 수 (도구·정책·안전장치는 공통이라 제외)."""
    from .tool_agent.loop import ToolAgent
    from .tool_agent.multi import MultiAgent, RuleSupervisor

    out = {"workflow": _loc(WorkflowAgent), "single": _loc(ToolAgent._run_steps, ToolAgent._step, ToolAgent._wrap_up),
           "plan_execute": _loc(PlanExecuteAgent), "multi": _loc(MultiAgent, RuleSupervisor)}
    try:
        from .framework.langgraph_agent import LangGraphAgent

        out["langgraph"] = _loc(LangGraphAgent)
    except ImportError:
        pass
    return out


def adaptivity(agent, configs: list[str]) -> dict[str, dict]:
    runners = _Runners(agent)
    out = {}
    for c in configs:
        errors = calls = 0
        for q in ADAPTIVITY_PROBES:
            runner, close = runners.make(c)
            try:
                r, _ = runner.run(q, [])
                tools = [s for s in r.steps if s.kind == "tool" and not (s.tool or "").startswith("delegate_")]
                calls += len(tools)
                errors += sum(1 for s in tools if s.ok is False)
            finally:
                if close:
                    close()
        out[c] = {"tool_calls": calls, "tool_errors": errors}
    return out


def run_study(agent) -> dict[str, Any]:
    from .framework.langgraph_agent import available

    configs = ["workflow", "single", "plan_execute", "multi"] + (["langgraph"] if available() else [])
    t0 = time.perf_counter()
    bench = run_benchmark(agent, configs)
    return {"configs": configs, "bench": bench, "adaptivity": adaptivity(agent, configs), "loc": code_size(),
            "seconds": round(time.perf_counter() - t0, 1), "when": time.strftime("%Y-%m-%d %H:%M")}


def _row(label: str, values: list) -> str:
    return f"| {label} | " + " | ".join(str(v) for v in values) + " |"


def study_markdown(report: dict[str, Any]) -> str:
    cfgs, b = report["configs"], report["bench"]["configs"]
    names = [ARCHITECTURES[c][0] for c in cfgs]
    est = {c: round(b[c]["decisions_per_turn"] * AI_SECONDS + b[c]["p50_ms"] / 1000, 1) for c in cfgs}
    lines = [
        "# v0.4 Architecture Study — 토지 상담 Agent 의 구조 비교", "",
        f"측정: {report['when']} · v1.3 벤치마크 문제지 {report['bench']['cases']}문제(질문 {report['bench']['turns']}개) · "
        f"답을 만든 쪽: {report['bench']['policy']} (규칙 기반 정책으로 판단) · `python main.py architecture-study` 로 다시 만들 수 있음", "",
        "## 1. 비교한 구조", "",
        "모두 **같은 도구**(검색·상세·비교·건축 가능 여부·피드백)와 같은 실행기·안전장치를 쓰고, **흐름 제어만** 다릅니다.", "",
        "| 구조 | 흐름 | 판단(AI 호출) |",
        "|---|---|---|",
        "| ① Workflow | 질문 → 검색 → (비교/건축 확인) → 정해진 형식의 답 | 없음 (코드가 순서를 정함) |",
        "| ② ReAct | 판단 → 도구 → **결과를 보고 다시 판단** … → 답 | 도구를 쓸 때마다 1번 + 답 1번 |",
        "| ③ Plan-and-Execute | 계획 1번 → 계획대로 실행 → 답 1번 | 항상 2번 (중간에 계획을 고치지 않음) |",
        "| ④ Multi-Agent | 총괄 → 검색 Agent / 분석 Agent → 총괄이 정리 | Agent 마다 따로 (가장 많음) |",
        "| ⑤ ReAct on LangGraph | ②와 같은 판단을 LangGraph 그래프로 (v0.5) | ②와 같음 |", "",
        "```mermaid", "flowchart LR",
        "  subgraph W[① Workflow]", "    w1[질문] --> w2[검색] --> w3[비교/확인] --> w4[답]", "  end",
        "  subgraph R[② ReAct]", "    r1[질문] --> r2{판단} -->|도구| r3[실행] --> r2", "    r2 -->|답| r4[답]", "  end",
        "  subgraph P[③ Plan-and-Execute]", "    p1[질문] --> p2[계획] --> p3[실행 1..n] --> p4[답]", "  end", "```", "",
        "## 2. 결과", "",
        "| 항목 | " + " | ".join(names) + " |", "|---|" + "---|" * len(cfgs),
        _row("문제 통과율", [f"**{b[c]['case_pass']}%**" for c in cfgs]),
        _row("질문 통과율", [f"{b[c]['turn_pass']}%" for c in cfgs]),
        _row("정밀도 / 재현율", [f"{b[c]['precision']}% / {b[c]['recall']}%" for c in cfgs]),
        _row("질문당 판단 횟수", [b[c]["decisions_per_turn"] for c in cfgs]),
        _row("질문당 도구 호출", [b[c]["tools_per_turn"] for c in cfgs]),
        _row("응답 시간 p50 (규칙 판단)", [f"{b[c]['p50_ms']}ms" for c in cfgs]),
        _row(f"예상 응답 시간 (AI 판단 1번 {AI_SECONDS}초 가정)", [f"약 {est[c]}초" for c in cfgs]),
        _row("결과가 예상과 다를 때 헛도구 호출(오류)", [report["adaptivity"][c]["tool_errors"] for c in cfgs]),
        _row("흐름 제어 코드 줄 수", [report["loc"].get(c, "-") for c in cfgs]), "",
        "### 분야별 통과율", "", "| 분야 | " + " | ".join(names) + " |", "|---|" + "---|" * len(cfgs)]
    cats = sorted({k for c in cfgs for k in b[c]["by_category"]})
    lines += [_row(k, [f"{b[c]['by_category'].get(k, '-')}%" for c in cfgs]) for k in cats]
    wf = b.get("workflow", {})
    lines += ["", "### 틀린 문제", ""]
    for c, n in zip(cfgs, names):
        lines.append(f"- {n}: " + (", ".join(b[c]["failed_cases"]) or "없음"))
    lines += [
        "", "## 3. 해석", "",
        f"- **① Workflow 는 이어서 묻기에 약함** (이어서 묻기 {wf.get('by_category', {}).get('이어서 묻기', '-')}%): "
        "'그 중에 넓은 건?', '다른 지역에도 있어?', '정말 없어?' 처럼 앞 대화에 따라 할 일이 달라지는 질문을 코드 순서만으로는 처리 못 함. "
        "AI 판단이 없어 가장 싸고 빠르지만, 손님 말의 흐름을 따라가지 못함.",
        "- **② ReAct 는 결과를 보고 다음 행동을 고름**: 검색 결과가 0~1건이면 비교·건축 확인을 건너뜀 → 헛도구 호출 0. "
        "대신 도구를 쓸 때마다 판단(AI 호출)이 늘어남.",
        "- **③ Plan-and-Execute 는 판단이 항상 2번**이라 AI 호출이 많은 질문(검색+비교+확인)에서 ReAct 보다 싸고 빠름. "
        "하지만 계획을 고치지 않아 결과가 예상과 다르면(0건인데 비교) 헛도구 호출·오류가 생기고, "
        "'제일 많은 지역 매물 보여줘'처럼 **중간 결과(어느 지역이 많은지)를 보고 다음 일을 정해야 하는 질문**을 처리하지 못함.",
        "- **④ Multi-Agent 는 판단이 약 2배** (총괄 + 전문 Agent). 정확도는 같고 비용·지연이 늘지만, 역할별로 따로 고치고 "
        "다른 서버로 나눌 수 있음(v0.8 A2A).",
        "- **⑤ LangGraph 판은 ReAct 와 결과가 같음** (같은 판단 규칙), 프레임워크 때문에 몇 ms 더 걸림 → v0.5 보고서 참고.", "",
        "## 4. 결정", "",
        "- **기본 구조: ② ReAct** — 토지 상담은 '그 중에', '다른 지역', '정말 없어?' 같은 이어서 묻기가 많고, 검색 결과에 따라 다음 행동이 "
        "달라져야 함. 정확도와 헛도구 호출에서 가장 안정적.",
        "- **확장 구조: ④ Multi-Agent** — 정확도는 같으면서 역할 분리·분산(A2A)이 필요할 때 선택 (화면의 '상담팀'). "
        "v1.2 '자동' 방식이 손님 만족도로 둘 중 하나를 고름.",
        "- **비용을 줄여야 할 때: ③ Plan-and-Execute** 를 후보로 둠 (AI 호출이 고정 2번). 단, 0건·1건일 때 계획을 건너뛰는 보완이 필요.",
        "- ① Workflow 는 AI 를 못 쓸 때의 최소 기능 수준과 비교하는 기준선으로만 씀.", "",
        "※ 이 측정은 AI 대신 규칙 기반 정책으로 판단한 결과입니다. GPT/Claude 를 연결하면 같은 명령으로 실제 AI 판단 횟수·토큰·지연을 잴 수 있습니다.",
        "※ 문제지는 연습용 샘플 매물 기준이라 매물번호는 '(샘플 매물)' 로 가렸습니다.",
    ]
    return NUMBER.sub("(샘플 매물)", "\n".join(lines))


def framework_markdown(report: dict[str, Any]) -> str:
    b = report["bench"]["configs"]
    if "langgraph" not in b:
        return "# v0.5 Framework\n\nLangGraph 가 설치되지 않아 측정하지 못했습니다 (`pip install langgraph`).\n"
    import langgraph

    s, g = b["single"], b["langgraph"]
    same = s["passed_cases"] == g["passed_cases"]
    try:
        version = __import__("importlib.metadata").metadata.version("langgraph")
    except Exception:  # noqa: BLE001
        version = getattr(langgraph, "__version__", "?")
    lines = [
        "# v0.5 Framework — 직접 만든 Agent vs LangGraph", "",
        f"측정: {report['when']} · LangGraph {version} · v1.3 벤치마크 문제지 · 규칙 기반 판단", "",
        "## 1. 무엇을 바꿨나", "",
        "v0.2 에서 반복문으로 직접 만든 ReAct Agent 를 **LangGraph**(대표적인 에이전트 프레임워크)로 다시 만들었습니다.",
        "공정한 비교를 위해 도구·판단 규칙·안전장치(v1.0)·추적(v1.1)은 똑같이 쓰고 **흐름 제어만** 프레임워크에 맡겼습니다.", "",
        "| | 직접 구현 (v0.2) | LangGraph (v0.5) |", "|---|---|---|",
        "| 흐름 | `for` 반복문: 판단 → 도구 → … | 그래프: `START → 판단·실행 ⟲ → (정리) → END` |",
        "| 대화 상태 저장 | 직접 만든 체크포인트 (SQLite, v0.3) | 프레임워크 checkpointer (대화마다 thread_id) + 직접 만든 체크포인트 |",
        "| 저장된 상태 다시 보기 | 상담 기록 화면 | `get_state_history` 로 단계별 상태 |",
        "| 흐름 그림 | 직접 그림 | 그래프에서 자동 생성 (아래) |",
        f"| 흐름 제어 코드 | {report['loc'].get('single')}줄 | {report['loc'].get('langgraph')}줄 (+ 프레임워크) |",
        "| 추가 의존성 | 없음 | `langgraph`, `langchain-core` 등 |", "",
        "## 2. 측정", "", "| 항목 | 직접 구현 | LangGraph |", "|---|---|---|",
        f"| 문제 통과율 | {s['case_pass']}% | {g['case_pass']}% |",
        f"| 같은 문제를 맞힘 | | {'✅ 완전히 같음' if same else '❌ 다름'} |",
        f"| 응답 시간 p50 / p95 | {s['p50_ms']} / {s['p95_ms']}ms | {g['p50_ms']} / {g['p95_ms']}ms |",
        f"| 질문당 판단 / 도구 호출 | {s['decisions_per_turn']} / {s['tools_per_turn']} | {g['decisions_per_turn']} / {g['tools_per_turn']} |", "",
        f"→ 결과는 같고, 프레임워크가 질문당 약 {round(g['p50_ms'] - s['p50_ms'], 1)}ms 를 더 씀 (상태를 매 단계 직렬화·저장). "
        "AI 응답(1~3초)에 비하면 작은 차이.", "",
        "## 3. LangGraph 가 그린 흐름", "", "```mermaid",
        _mermaid(), "```", "",
        "## 4. 결정", "",
        "- **운영(웹·API)은 직접 구현을 유지**: 결과가 같고 더 빠르며, 의존성이 없고, 우리 체크포인트·추적·안전장치와 이미 맞물려 있음.",
        "- **LangGraph 판은 선택 사항으로 둠**: `python main.py agent --framework langgraph`, 벤치마크 설정 `langgraph`. "
        "흐름이 복잡해지면(사람 승인 단계, 분기 많은 상담 등) 그래프 선언·상태 기록·흐름 그림이 유리하므로 그때 옮길 수 있게 같은 실행기(_step)를 공유하도록 만듦.",
        "- 배운 점: 프레임워크는 '반복문·상태 저장·그림'을 대신해 주지만, 도구 설계·판단 규칙·안전장치·평가는 여전히 직접 만들어야 함.",
    ]
    return "\n".join(lines)


def _mermaid() -> str:
    import tempfile
    from pathlib import Path

    from .agent import RealEstateAgent
    from .config import Settings
    from .framework.langgraph_agent import LangGraphAgent

    with tempfile.TemporaryDirectory() as tmp:
        a = RealEstateAgent(Settings(data_dir=Path(tmp), db_path=Path(tmp) / "m.db", log_dir=Path(tmp),
                                     llm_providers=["rule"]))
        text = LangGraphAgent(a, log_dir="").mermaid()
        a.storage.conn.close()
    lines = text.strip().splitlines()
    if lines and lines[0] == "---":  # 앞의 설정 블록은 빼고 그래프만
        lines = lines[lines.index("---", 1) + 1:]
    return "\n".join(lines)
