"""명령줄 인터페이스.

  python main.py                              웹 서비스 실행 (브라우저가 자동으로 열림)
  python main.py web --host 0.0.0.0           같은 와이파이의 휴대폰·다른 PC 에서도 접속
  python main.py sync                         매물 폴더 수집·정제 (오류/중복 보고서)
  python main.py recommend "찾는 조건"            터미널에서 추천 (대화형: 다음/피드백)
  python main.py chat                         대화형 상담 모드
  python main.py customers                    저장된 손님 목록
  python main.py proactive                    저장된 손님에게 맞는 매물 선제 탐색
  python main.py watch                        폴더 감시 + 새 매물 선제 알림 (무한 반복)
  python main.py export --out 매물정제.xlsx     정제된 매물을 엑셀로 내보내기
  python main.py template                     매물 입력 양식(엑셀) 만들기
  python main.py setup-key                    OpenAI API 키 저장 (.env)
  python main.py llm-check                    GPT/Claude 연결 점검
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .agent import ProactiveAlert, RealEstateAgent, SyncReport
from .config import get_settings
from .models import CustomerRequirement, ScoredListing, format_price
from .recommender import EXHAUSTED_MESSAGE, NOT_FOUND_MESSAGE, RecommendationSession

LINE = "─" * 78

HELP = """명령:  [Enter]/n 다음 순위 매물 · p 이전 · t 상위 5개 · d 상세 · l 마음에 듦
       그냥 말로 입력해도 됩니다:
         '마음에 안 들어요', '너무 비싸요', '채광이 별로'  -> 다음 순위 매물 (사유를 학습)
         '좀 더 싼 걸로', '여주 쪽도', '300평 이상으로'   -> 조건을 바꿔 다시 추천
       w 학습한 손님 취향 · c 현재 조건 · s 손님 조건 저장(새 매물 선제 알림) · q 종료"""



def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def print_listing(rank: int, s: ScoredListing, total: int, detail: bool = False) -> None:
    l = s.listing
    print(LINE)
    print(f" 추천 {rank}순위 / {total}건   점수 {s.score:.1f}점   [{l.listing_id}]  출처: {l.source}")
    print(f" 소재지 : {l.address}")
    print(f" 지목   : {l.jimok or '-'}     용도지역: {l.zoning or '-'}")
    print(f" 면적   : {l.area_m2:,.0f}㎡ ({l.area_pyeong:,.0f}평)")
    print(f" 가격   : {format_price(l.price_manwon)}  (평당 {l.price_per_pyeong:,.0f}만원)")
    print(f" 도로   : {l.road_access or '-'}")
    if l.description:
        print(f" 특이사항: {l.description}")
    if s.reasons:
        print(" ✔ " + " / ".join(s.reasons))
    if s.penalties:
        print(" ⚠ " + " / ".join(s.penalties))
    if detail:
        print(f" 등록일 : {l.registered_date or '-'}   상태: {l.status}   거래: {l.deal_type}")
        print(f" 연락처 : {l.owner_contact or '-'}")
        if l.lat:
            print(f" 좌표   : {l.lat:.6f}, {l.lon:.6f}")
        if l.extra.get("duplicate_sources"):
            print(f" 중복 등록 출처: {', '.join(l.extra['duplicate_sources'])}")
        if l.extra.get("price_history"):
            print(f" 가격 변동 기록: {', '.join(format_price(p) for p in l.extra['price_history'])}")
        print(f" 원본 파일: {l.source_file}")


def show_current(agent: RealEstateAgent, session: RecommendationSession, explain: bool = True) -> None:
    cur = session.current()
    if cur is None:
        print(LINE)
        print(f" {EXHAUSTED_MESSAGE if session.cursor else NOT_FOUND_MESSAGE}")
        print("  ('f <피드백>' 으로 조건을 바꾸거나 새로 검색해 보세요)")
        if session.filter_stats:
            print(" (제외 사유: " + ", ".join(f"{k} {v}건" for k, v in session.filter_stats.items()) + ")")
        return
    if session.ai_changes and session.opt.pref is not None and session.opt.pref.ai_summary:
        print(f" 🤖 [AI 취향 분석] {session.opt.pref.ai_summary}  (반영: {', '.join(session.ai_changes)})")
        session.ai_changes = []
    if session.last_learning:
        print(f" 🧠 [학습] {', '.join(session.last_learning)}"
              + (" → 순위를 다시 매겼습니다" if session.reordered else ""))
        session.last_learning = []
    print_listing(session.cursor + 1, cur, session.total)
    if explain:
        print(" 💬 " + agent.explain(session.req, cur))


def run_session(agent: RealEstateAgent, req: CustomerRequirement, interactive: bool = True) -> None:
    print(f"\n[이해한 조건] {req.describe()}   (분석: {agent.llm.last_used})")
    for note in req.notes:
        print(f"[안내] {note}")
    session = agent.recommend(req)
    if session.opt.pref is not None and session.opt.pref.n_feedback:
        print(f"[학습된 취향 적용] {session.opt.pref.describe(session.opt.prior)}")
    if session.relaxed:
        print(f"[에이전트] 조건에 맞는 매물이 적어 조건을 완화했습니다: {' → '.join(session.relaxed)}")
    print(f"[후보] {session.total}건")
    show_current(agent, session)
    if not interactive:
        for i, s in enumerate(session.top(5)[1:], start=2):
            print_listing(i, s, session.total)
        return
    print(LINE + "\n" + HELP)
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if cmd in {"q", "quit", "exit", "종료"}:
            return
        if cmd in {"", "n", "다음"}:
            agent.reject(session)
            show_current(agent, session)
        elif cmd.startswith("n "):
            agent.reject(session, cmd[2:].strip())
            show_current(agent, session)
        elif cmd in {"p", "이전"}:
            session.previous()
            show_current(agent, session, explain=False)
        elif cmd == "t":
            for i, s in enumerate(session.top(5), start=session.cursor + 1):
                print(f" {i:>2}. {s.score:5.1f}점  {s.listing.summary()}")
        elif cmd == "d":
            if session.current():
                print_listing(session.cursor + 1, session.current(), session.total, detail=True)
        elif cmd == "l":
            agent.like(session)
            print(" 👍 관심 매물로 기록했습니다.")
        elif cmd.startswith("f "):
            new_req = agent.refine(session, cmd[2:].strip())
            print(f"[조건 수정] {new_req.describe()}   (분석: {agent.llm.last_used})")
            if session.relaxed:
                print(f"[에이전트] 조건 완화: {' → '.join(session.relaxed)}")
            print(f"[후보] {session.total}건")
            show_current(agent, session)
        elif cmd == "c":
            print(" " + session.req.describe())
        elif cmd == "w":
            print(" " + session.opt.pref.describe(session.opt.prior).replace("\n", "\n "))
        elif cmd == "s":
            if not session.req.customer_name:
                name = input(" 손님 이름: ").strip()
                if not name:
                    continue
                session.req.customer_name = name
            agent.save_customer(session)
            print(f" 💾 '{session.req.customer_name}' 님 조건 저장 - 새 매물이 들어오면 먼저 알려드립니다.")
        elif len(cmd) >= 2:
            # '마음에 안 들어요' -> 다음 순위 / '좀 더 싼 걸로' -> 조건 변경
            if agent.respond(session, cmd) == "refine":
                print(f"[조건 수정] {session.req.describe()}   (분석: {agent.llm.last_used})")
                print(f"[후보] {session.total}건")
            show_current(agent, session)
        else:
            print(HELP)


def print_alert(a: ProactiveAlert) -> None:
    print(LINE)
    print(f" 🔔 [선제 제안] {a.customer} 님께 맞는 매물  ({a.scored.score:.1f}점)")
    print(f"    {a.scored.listing.summary()}")
    print(f"    💬 {a.message}")


def print_sync(report: SyncReport, verbose: bool = False) -> None:
    print(report.summary())
    if verbose:
        print(report.clean.report())


def run_tool_agent(agent: RealEstateAgent, question: str, max_steps: int = 6, use_mcp: bool = False,
                   multi: bool = False, distributed: bool = False) -> int:
    """v0.2 Tool-use Agent 를 터미널에서: 도구 호출 과정(행동→관찰)과 답변을 출력."""
    from .tool_agent import ToolAgent
    from .tool_agent.memory import LongTermMemory

    # 터미널 상담도 장기 기억 사용 (지난번에 찾던 조건, 싫다고 한 매물)
    source = None
    if use_mcp:  # 우리 Agent 가 MCP host/client 가 되어 MCP 서버의 도구를 씀
        from .tool_agent.mcp_client import McpToolSource, stdio_server_params

        source = McpToolSource(stdio_server_params(agent.settings)).start()
        print(f"MCP 서버에 연결: 도구 {[t.name for t in source.tools()]}")
    if distributed:  # v0.8: 검색·분석 Agent 가 다른 서버(A2A). 꺼져 있으면 여기서 대신 처리
        from .a2a.distributed import DistributedMultiAgent

        runner = DistributedMultiAgent(agent, max_steps=max_steps, memory=LongTermMemory(agent.storage.conn),
                                       owner="terminal")
        for who, c in runner.clients.items():
            print(f"  {who}: {c.url}")
    elif multi:  # v0.7: 총괄 Agent 가 검색 Agent · 분석 Agent 에게 나눠 맡김
        from .tool_agent.multi import MultiAgent

        runner = MultiAgent(agent, max_steps=max_steps, memory=LongTermMemory(agent.storage.conn), owner="terminal")
    else:
        runner = ToolAgent(agent, max_steps=max_steps, memory=LongTermMemory(agent.storage.conn), owner="terminal",
                           tool_source=source)
    history: list = []
    interactive = not question
    if interactive:
        print("\nAI 상담 (빈 줄 또는 q 로 종료). 예) 음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    while True:
        q = question or input("\n질문> ").strip()
        if not q or q.lower() in {"q", "quit", "exit"}:
            return 0
        result, history = runner.run(q, history)
        print()
        for s in result.steps:
            who = f"{s.agent} " if s.agent else ""
            if s.kind == "tool":
                mark = "✔" if s.ok else "✘"
                print(f"  {who}[{s.n}] {mark} {s.tool}({s.args}) → {s.summary}  ({s.ms}ms, {s.policy})")
            else:
                print(f"  {who}[{s.n}] {s.kind}: {s.summary}  ({s.policy})")
        print(f"\n{result.answer}")
        print(f"\n  (도구 {result.tool_calls}회, 오류 {result.tool_errors}회, {result.total_ms / 1000:.1f}초, {result.policy})")
        if not interactive:
            return 0
        question = ""


def run_a2a_call(url: str, question: str, context_id: str = "") -> int:
    """v0.8: 아무 A2A Agent 에게 질문 (Agent Card 확인 → message/stream 으로 진행 상황을 받으며 답 기다림)."""
    import os

    from .a2a import protocol as p
    from .a2a.client import A2AClient

    client = A2AClient(url, os.getenv("REA_A2A_TOKEN") or None)
    try:
        card = client.card()
    except p.A2AUnavailable as e:
        print(f"[오류] {e}\n  먼저 다른 터미널에서: python main.py a2a-serve --role consult")
        return 1
    print(f"Agent: {card.get('name')} (A2A {card.get('protocolVersion')}) · 기술: "
          + ", ".join(s.get("name", "") for s in card.get("skills", [])))
    try:
        for ev in client.stream(question, context_id=context_id or None):
            kind = ev.get("kind")
            if kind == "task":
                print(f"  task {ev['id'][:8]} 접수 (context {ev['contextId'][:8]})")
            elif kind == "status-update":
                msg = p.get_text(((ev.get("status") or {}).get("message") or {}).get("parts"))
                if not ev.get("final") and msg:
                    print("  · " + msg)
                elif ev.get("final"):
                    print(f"  상태: {ev['status']['state']}")
            elif kind == "artifact-update":
                print("\n" + p.get_text(ev["artifact"]["parts"]))
    except (p.A2AError, p.A2AUnavailable) as e:
        print(f"[오류] {e}")
        return 1
    return 0


def run_a2a_demo(settings) -> int:  # noqa: C901
    """v0.8 발표용: 검색·분석 Agent 서버를 따로 띄우고(프로세스 2개) 분산 상담 → 분석 서버를 끄고 장애 대처까지 보여줌."""
    import subprocess
    import time as _time

    from .a2a.client import A2AClient
    from .a2a.distributed import DistributedMultiAgent
    from .config import PROJECT_ROOT
    from .tool_agent import AgentContext
    from .tool_agent.multi import ANALYST, SEARCHER

    ports = {SEARCHER: ("search", 8111), ANALYST: ("analysis", 8112)}
    procs = {}
    print("① 검색 Agent · 분석 Agent 를 각각 다른 프로세스(서버)로 시작")
    for who, (role, port) in ports.items():
        procs[who] = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "--data-dir", str(settings.data_dir), "--db",
             str(settings.db_path), "--no-llm", "a2a-serve", "--role", role, "--port", str(port)],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clients = {who: A2AClient(f"http://127.0.0.1:{port}/", timeout=30) for who, (_, port) in ports.items()}
    try:
        for who, c in clients.items():
            for _ in range(80):
                try:
                    card = c.card()
                    print(f"   {who}: {c.url}  → Agent Card '{card['name']}', 기술 {[s['id'] for s in card['skills']]}")
                    break
                except Exception:  # noqa: BLE001 - 서버가 뜰 때까지 기다림
                    _time.sleep(0.25)
            else:
                print(f"[오류] {who} 서버가 시작되지 않았습니다")
                return 1
        agent = RealEstateAgent(settings)
        agent.sync()
        from . import observability

        observability.setup(settings, "demo-supervisor", json_logs=False)  # v1.1: 세 프로세스의 span 이 같은 DB 로
        runner = DistributedMultiAgent(agent, clients=clients, ctx=AgentContext(), log_dir="")
        history: list = []
        traces: list[str] = []

        def ask(q: str) -> None:
            nonlocal history
            r, history = runner.run(q, history)
            traces.append(r.trace_id)
            print(f"\n손님> {q}")
            for s in r.steps:
                if s.kind in {"tool", "remote", "fallback"} and s.summary and not s.summary.startswith("AI("):
                    print(f"   [{s.agent}] " + (f"{s.tool} → " if s.tool else f"{s.kind}: ") + s.summary[:110])
            print("답변> " + r.answer.replace("\n", "\n      "))

        print("\n② 분산 상담 (총괄 Agent 는 이 프로세스, 검색·분석은 A2A 로 다른 서버에)")
        for q in ("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘", "그 중에 넓은 건?",
                  "음성군 말고 다른 지역에도 있어?"):
            ask(q)
        print("\n③ 분석 Agent 서버를 끔 (장애 상황)")
        procs[ANALYST].terminate()
        procs[ANALYST].wait(5)
        ask("여주 공장 부지 2개 비교해줘")
        print("\n→ 분석 서버가 없어도 이 서버의 분석 Agent 가 대신 처리해 상담이 이어졌습니다 (trace 의 '전환').")
        print("\n④ v1.1 분산 추적: 첫 질문이 세 프로세스(총괄·검색·분석)를 거친 과정 (traceparent 로 한 trace 로 이어짐)\n")
        observability.TRACER.flush(3)
        _time.sleep(1.2)  # 검색 서버가 자기 span 을 저장할 시간
        from .observability.dashboard import text_waterfall, trace_tree

        print(text_waterfall(trace_tree(agent.storage.conn, None, traces[0])))
        print(f"\n   python main.py trace {traces[0]}  (또는 관리자 화면 → 📈 관측)")
        return 0
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()


def run_reliability_test(agent: RealEstateAgent, seeds: int = 3, deadline: float = 2.0, delay: float = 3.0) -> int:
    """v1.0: 장애 주입 시험 - 고장 종류별 성공률·응답 시간 표 출력, logs 에 저장."""
    from datetime import datetime

    from .reliability.chaos import chaos_markdown, run_chaos

    print(f"장애 주입 시험: 같은 대화 7개 질문 × {seeds}회, 시간 예산 {deadline}초")
    report = run_chaos(agent, seeds=seeds, deadline=deadline, delay=delay, progress=lambda s: print("  · " + s))
    md = chaos_markdown(report)
    print("\n" + md)
    out = Path(agent.settings.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"reliability_{stamp}.md").write_text(md, encoding="utf-8")
    (out / f"reliability_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out / f'reliability_{stamp}.md'}")
    return 0 if all(r["success"] == 100.0 for r in report["rows"] if r["guards"]) else 1


def run_trace(agent: RealEstateAgent, trace_id: str = "") -> int:
    """v1.1: trace 하나를 터미널 폭포 그림으로 (비우면 가장 최근 질문)."""
    from .observability.dashboard import recent, text_waterfall, trace_tree

    conn = agent.storage.conn
    if not trace_id or trace_id == "last":
        last = recent(conn, None, 1)
        if not last:
            print("아직 기록된 질문이 없습니다. 먼저 python main.py agent \"질문\" 또는 웹 AI 상담을 써 보세요.")
            return 1
        trace_id = last[0]["trace_id"]
    rows = trace_tree(conn, None, trace_id)
    if not rows:
        print(f"trace {trace_id} 를 찾을 수 없습니다.")
        return 1
    root = rows[0]
    print(f"trace {trace_id}  ·  {root['attrs'].get('question', '')}  ·  "
          f"{max(r['offset_ms'] + r['dur_ms'] for r in rows):,.1f}ms  ·  span {len(rows)}개\n")
    print(text_waterfall(rows))
    return 0


def run_observe(agent: RealEstateAgent, hours: float = 24) -> int:
    """v1.1: 최근 상담 지표 요약 (관리자 '관측' 화면과 같은 숫자)."""
    from .observability.dashboard import recent, summary

    s = summary(agent.storage.conn, None, hours)
    print(f"최근 {hours:g}시간: 질문 {s['questions']}건 · 성공률 {s['success_rate']}% · 응답 p50 {s['p50_ms']}ms / "
          f"p95 {s['p95_ms']}ms · AI 비율 {s['ai_share']}% · 토큰 {s['tokens']:,} · 도구 오류율 {s['tool_error_rate']}%")
    if s["reliability"]:
        print("안전장치: " + ", ".join(f"{k} {v}" for k, v in s["reliability"].items()))
    for t in s["tools"]:
        print(f"  tool {t['tool']:<18} {t['calls']:>4}회  오류 {t['errors']}  p50 {t['p50_ms']}ms  p95 {t['p95_ms']}ms")
    for r in s["llm"]:
        print(f"  AI   {r['provider']:<18} {r['calls']:>4}회  실패 {r['errors']}  p95 {r['p95_ms']}ms  "
              f"토큰 {r['input_tokens']:,}/{r['output_tokens']:,}")
    print("\n최근 질문:")
    for t in recent(agent.storage.conn, None, 10):
        print(f"  {t['time']}  {t['ms']:>8,.0f}ms  {t['mode']:<11} {t['policy']:<9} {t['trace_id'][:12]}  {t['question']}")
    return 0


def run_online_sim(agent: RealEstateAgent, p_single: float = 0.65, p_multi: float = 0.75) -> int:
    """v1.2: 온라인 학습 시뮬레이션 (추천 순서 학습 + '자동' 상담 방식 밴딧)."""
    from datetime import datetime

    from .online.simulate import markdown, simulate_bandit, simulate_learning

    md = markdown(simulate_learning(agent), simulate_bandit({"single": p_single, "multi": p_multi}))
    print("\n" + md)
    out = Path(agent.settings.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"online_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n저장: {path}")
    return 0


def run_benchmark_cli(agent: RealEstateAgent, configs: str = "", load: bool = True, check: bool = False,
                      save_baseline: bool = False, ai_delay: float = 0.3) -> int:
    """v1.3: 문제지로 방식별 품질·속도 측정 + 동시 처리량 + 기준선 비교."""
    import logging as _logging
    from datetime import datetime

    from .benchmark import (BASELINE, baseline_of, compare_baseline, markdown, run_benchmark, run_load)

    _logging.getLogger("real_estate_agent").setLevel(_logging.ERROR)
    chosen = [c.strip() for c in configs.split(",") if c.strip()] or None
    print("벤치마크 실행 중 …")
    report = run_benchmark(agent, chosen, progress=lambda s: print("  · " + s))
    load_rule = load_ai = None
    if load:
        print("  · 동시 처리량 (규칙 기반)")
        load_rule = run_load(agent, requests_per_level=32)
        if ai_delay > 0:
            print(f"  · 동시 처리량 (AI 응답 {ai_delay}초 가정)")
            load_ai = run_load(agent, requests_per_level=16, ai_delay=ai_delay)
            load_ai += run_load(agent, levels=(16,), requests_per_level=16, ai_delay=ai_delay, limit=16)
    regressions = None
    if BASELINE.exists():
        regressions = compare_baseline(report, json.loads(BASELINE.read_text(encoding="utf-8")))
    md = markdown(report, load_rule, load_ai, regressions)
    print("\n" + md)
    out = Path(agent.settings.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"benchmark_{stamp}.md").write_text(md, encoding="utf-8")
    (out / f"benchmark_{stamp}.json").write_text(json.dumps({"report": report, "load": load_rule, "load_ai": load_ai},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out / f'benchmark_{stamp}.md'}")
    if save_baseline:
        BASELINE.write_text(json.dumps(baseline_of(report), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"기준선 저장: {BASELINE}")
    if check and regressions:
        print("\n❌ 기준선보다 나빠졌습니다.")
        return 1
    return 0


def run_clean_practice(settings, yes: bool = False, dry_run: bool = False) -> int:
    """실제 운영 DB 에 섞인 연습용 샘플 매물 기록만 지움 (지우기 전 백업)."""
    import sqlite3

    from .config import is_sample_data
    from .practice_cleanup import clean, scan, summary

    if is_sample_data(settings.data_dir):
        print("연습용(샘플) 저장 파일은 정리할 필요가 없습니다. 실제 운영 DB 를 정리하려면 --data-dir 없이 실행하세요.")
        return 1
    db = Path(settings.db_path)
    if not db.exists():
        print(f"실제 운영 DB 가 아직 없습니다: {db}")
        return 0
    conn = sqlite3.connect(str(db), timeout=10)
    try:
        plan = scan(conn)
        rows = summary(plan)
        print(f"실제 운영 DB: {db}")
        if not any(n for _, n in rows):
            print("✅ 연습 기록이 없습니다. 정리할 것이 없습니다.")
            return 0
        print("\n지울 연습 기록 (연습용 샘플 매물이 나오는 기록만):")
        for label, n in rows:
            if n:
                print(f"  · {label}: {n}")
        if plan["protected"]:
            print(f"  (실제 매물에도 있는 번호 {len(plan['protected'])}개는 건드리지 않음)")
        print("지우지 않는 것: 실제 매물, 실제 매물이 나온 상담, 손님 목록, 관리자 비밀번호, 설정")
        if dry_run:
            print("\n(--dry-run: 아무것도 지우지 않았습니다)")
            return 0
        if not yes and input("\n지울까요? 지우기 전에 DB 전체를 data/backup/ 에 복사해 둡니다. (y/N) ").strip().lower() not in {"y", "yes", "ㅛ"}:
            print("취소했습니다.")
            return 0
        result = clean(conn, db)
        print(f"\n🧹 정리했습니다. 백업: {result['backup']}")
        print("되돌리려면 프로그램을 끄고 백업 파일을 data/agent.db 로 복사하세요.")
        return 0
    finally:
        conn.close()


def run_compare_agents(agent: RealEstateAgent) -> int:
    """v0.7: 같은 대화를 단일 Agent / 멀티 Agent 로 돌려 비교표를 출력하고 logs 에 저장."""
    from datetime import datetime

    from .tool_agent.multi import compare_agents, compare_markdown, report_json

    report = compare_agents(agent)
    md = compare_markdown(report)
    print("\n" + md)
    out = Path(agent.settings.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"compare_agents_{stamp}.md").write_text(md, encoding="utf-8")
    (out / f"compare_agents_{stamp}.json").write_text(report_json(report), encoding="utf-8")
    print(f"\n저장: {out / f'compare_agents_{stamp}.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="real-estate-agent", description="Real Estate Recommendation Agent (토지)")
    p.add_argument("--data-dir", help="매물 폴더 경로 (기본: 프로젝트의 '매물' 폴더)")
    p.add_argument("--db", help="SQLite DB 경로")
    p.add_argument("--no-llm", action="store_true", help="LLM 없이 규칙 기반으로만 동작")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("sync", help="매물 수집·정제")
    s.add_argument("--report", action="store_true", help="오류/중복 상세 출력")

    r = sub.add_parser("recommend", help="찾는 조건으로 매물 추천")
    r.add_argument("inquiry", help="찾는 땅 조건 (예: 음성군 채광 좋은 2억원대 땅)")
    r.add_argument("--batch", action="store_true", help="대화 없이 상위 5개만 출력")

    sub.add_parser("chat", help="대화형 상담")

    ag = sub.add_parser("agent", help="AI 상담 (v0.2 Tool-use Agent: 도구를 골라 쓰며 답함, 실행 과정 출력)")
    ag.add_argument("question", nargs="?", default="", help="질문 (비우면 대화 모드)")
    ag.add_argument("--max-steps", type=int, default=6)
    ag.add_argument("--mcp", action="store_true", help="도구를 MCP 서버(python main.py mcp)에 연결해서 사용")
    ag.add_argument("--multi", action="store_true", help="v0.7 멀티 Agent (총괄 → 검색 Agent · 분석 Agent)")
    ag.add_argument("--distributed", action="store_true",
                    help="v0.8 분산 상담팀: 검색·분석 Agent 를 A2A 서버로 (a2a-serve 로 먼저 띄우기)")
    a2 = sub.add_parser("a2a-serve", help="v0.8 A2A Agent 서버 (search :8101 / analysis :8102 / consult :8100)")
    a2.add_argument("--role", choices=["search", "analysis", "consult"], default="consult")
    a2.add_argument("--host", default="127.0.0.1")
    a2.add_argument("--port", type=int, default=0, help="기본: search 8101, analysis 8102, consult 8100")
    ac = sub.add_parser("a2a-call", help="v0.8 A2A Agent 에게 질문 (Agent Card 확인 + 스트리밍)")
    ac.add_argument("question")
    ac.add_argument("--url", default="http://127.0.0.1:8100/")
    ac.add_argument("--context", default="", help="이어서 묻기: 같은 contextId")
    sub.add_parser("a2a-demo", help="v0.8 발표용: 검색·분석 Agent 서버를 띄워 분산 상담 + 장애 대처 시연")
    tr = sub.add_parser("trace", help="v1.1 질문 하나의 처리 과정을 시간 막대로 (기본: 가장 최근)")
    tr.add_argument("trace_id", nargs="?", default="last")
    cp = sub.add_parser("clean-practice", help="실제 운영 DB 에 섞인 연습용 샘플 매물 기록만 지움 (지우기 전 백업)")
    cp.add_argument("--yes", action="store_true", help="묻지 않고 바로 정리")
    cp.add_argument("--dry-run", action="store_true", help="무엇을 지울지만 보여줌")
    bm = sub.add_parser("benchmark", help="v1.3 벤치마크 (문제지로 품질·속도·처리량 측정, 기준선 비교)")
    bm.add_argument("--configs", default="", help="쉼표로: single,multi,distributed,mcp (기본: 모두)")
    bm.add_argument("--no-load", action="store_true", help="동시 처리량 측정은 건너뜀")
    bm.add_argument("--ai-delay", type=float, default=0.3, help="동시 처리량: AI 응답 시간 가정 (초, 0=건너뜀)")
    bm.add_argument("--check", action="store_true", help="기준선보다 나빠졌으면 실패(종료 코드 1)")
    bm.add_argument("--save-baseline", action="store_true", help="이번 결과를 benchmarks/baseline.json 으로 저장")
    os_ = sub.add_parser("online-sim", help="v1.2 온라인 학습 시뮬레이션 (반응으로 순서 학습 + '자동' 방식 밴딧)")
    os_.add_argument("--p-single", type=float, default=0.65, help="가정: 상담원 1명 만족 확률")
    os_.add_argument("--p-multi", type=float, default=0.75, help="가정: 상담팀 만족 확률")
    ob = sub.add_parser("observe", help="v1.1 최근 상담 지표 요약 (질문 수·성공률·응답 시간·AI 토큰·도구)")
    ob.add_argument("--hours", type=float, default=24)
    sub.add_parser("compare-agents", help="v0.7 단일 Agent vs 멀티 Agent 비교 (결과는 logs 폴더에 저장)")
    rt = sub.add_parser("reliability-test", help="v1.0 장애 주입 시험 (AI 오류·지연·환각·도구 장애 등에서 성공률)")
    rt.add_argument("--seeds", type=int, default=3, help="같은 대화를 몇 번 돌릴지")
    rt.add_argument("--deadline", type=float, default=2.0, help="질문 하나의 시간 예산 (초)")
    rt.add_argument("--delay", type=float, default=3.0, help="'AI 지연' 고장에서 멈추는 시간 (초)")

    mc = sub.add_parser("mcp", help="MCP 서버 (Claude Desktop 등에서 매물 도구 사용). 기본 stdio")
    mc.add_argument("--http", action="store_true", help="Streamable HTTP 로 실행 (http://127.0.0.1:8002/mcp)")
    mc.add_argument("--host", default="127.0.0.1")
    mc.add_argument("--port", type=int, default=8002)

    c = sub.add_parser("customers", help="저장된 손님 목록")
    c.add_argument("--remove", help="해당 손님 선제 알림 중지")

    pr = sub.add_parser("proactive", help="저장된 손님 조건으로 전체 매물 선제 탐색")
    pr.add_argument("--threshold", type=float)

    w = sub.add_parser("watch", help="폴더 감시 + 선제 알림")
    w.add_argument("--interval", type=int, help="초 단위 (기본 600)")

    wb = sub.add_parser("web", help="웹 서비스 실행 (브라우저로 사용, 기본 명령)")
    wb.add_argument("--host", default="127.0.0.1", help="0.0.0.0 이면 같은 와이파이의 다른 기기에서도 접속")
    wb.add_argument("--port", type=int, default=8000)
    wb.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 끄기")

    ap = sub.add_parser("api", help="JSON API 서버 (FastAPI · async · 스트리밍). 문서: http://127.0.0.1:8001/docs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)

    tp = sub.add_parser("template", help="매물 입력 양식(엑셀) 만들기")
    tp.add_argument("--out", default="", help="저장 위치 (기본: 매물 폴더 안)")

    sub.add_parser("setup-key", help="OpenAI API 키를 .env 에 저장 (입력 내용은 화면에 안 보임)")
    sub.add_parser("llm-check", help="GPT/Claude 연결·모델 점검")
    sub.add_parser("set-admin-password", help="웹 관리자 화면 비밀번호 정하기/바꾸기 (입력 내용은 화면에 안 보임)")

    e = sub.add_parser("export", help="정제된 매물 엑셀/CSV 내보내기")
    e.add_argument("--out", default="정제된_토지매물.xlsx")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command == "template":
        from .template import write_template

        out = Path(args.out) if args.out else get_settings(data_dir=args.data_dir).data_dir / "토지매물_입력양식.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 1
        while out.exists() and not args.out:  # 이미 작성 중인 양식은 덮어쓰지 않음
            n += 1
            out = out.with_name(f"토지매물_입력양식_{n}.xlsx")
        path = write_template(out)
        print(f"매물 입력 양식을 만들었습니다: {path.resolve()}")
        print("엑셀로 열어 '매물입력' 시트에 적고 저장한 뒤, 웹 관리자 화면의 '매물 폴더 다시 읽기'를 누르세요.")
        return 0
    if args.command == "setup-key":
        from .setup_tools import setup_key

        return setup_key()
    settings = get_settings(data_dir=args.data_dir, db_path=args.db)
    if args.command == "mcp":  # 표준 출력은 MCP 통신 전용 -> 아무것도 print 하지 않음
        from .mcp_server import main as mcp_main

        return mcp_main(settings, http=args.http, host=args.host, port=args.port)
    if args.command == "clean-practice":
        return run_clean_practice(settings, args.yes, args.dry_run)
    if args.command == "a2a-call":
        return run_a2a_call(args.url, args.question, args.context)
    if args.command == "a2a-serve":
        if args.no_llm:
            settings.llm_providers = ["rule"]
        from .a2a.server import main as a2a_main

        return a2a_main(args.role, settings, host=args.host, port=args.port or None)
    if args.command == "a2a-demo":
        settings.llm_providers = ["rule"] if args.no_llm else settings.llm_providers
        return run_a2a_demo(settings)
    if args.command == "set-admin-password":
        from .admin_auth import set_password_cli

        return set_password_cli(settings)
    if args.no_llm:
        settings.llm_providers = ["rule"]
    cmd = args.command or "web"  # 아무 명령 없이 실행하면 웹 서비스
    if cmd == "api":
        try:
            import uvicorn

            from .api import create_api
        except ImportError:
            print("API 서버에는 fastapi, uvicorn 패키지가 필요합니다. 터미널에서:  pip install fastapi uvicorn")
            return 1
        print(f"JSON API: http://{args.host}:{args.port}/docs  (끄려면 Ctrl+C)")
        uvicorn.run(create_api(settings, start_realtime=True), host=args.host, port=args.port, log_level="warning")
        return 0
    if cmd == "web":
        from .web import run

        run(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8000),
            open_browser=not getattr(args, "no_browser", False), settings=settings)
        return 0
    agent = RealEstateAgent(settings)

    print(f"Real Estate Recommendation Agent  |  매물 폴더: {settings.data_dir}")
    if cmd == "llm-check":
        from .setup_tools import llm_check

        return llm_check(agent)
    print(agent.llm.status())
    if cmd in {"trace", "observe"}:
        return run_trace(agent, args.trace_id) if cmd == "trace" else run_observe(agent, args.hours)
    if cmd in {"agent", "compare-agents", "reliability-test"}:  # v1.1: 터미널 상담도 추적 기록 (관측 화면에서 보임)
        from . import observability

        observability.setup(settings, "cli", json_logs=cmd == "agent")
    if cmd == "agent":
        agent.sync()
        return run_tool_agent(agent, args.question, args.max_steps, use_mcp=args.mcp, multi=args.multi,
                              distributed=args.distributed)
    if cmd == "compare-agents":
        agent.sync()
        return run_compare_agents(agent)
    if cmd == "benchmark":  # 문제지에 맞는 샘플 매물 + 임시 DB 로 (사무실 DB·관측 기록을 건드리지 않음)
        import tempfile

        from . import observability
        from .benchmark import load_dataset
        from .config import PROJECT_ROOT, Settings

        observability.setup(settings, "benchmark", sqlite=False, json_logs=False)
        with tempfile.TemporaryDirectory() as tmp:
            bench = RealEstateAgent(Settings(data_dir=PROJECT_ROOT / load_dataset()["data_dir"],
                                             db_path=Path(tmp) / "benchmark.db", log_dir=settings.log_dir,
                                             llm_providers=settings.llm_providers))
            bench.sync()
            return run_benchmark_cli(bench, args.configs, not args.no_load, args.check, args.save_baseline,
                                     args.ai_delay)
    if cmd == "online-sim":
        agent.sync()
        return run_online_sim(agent, args.p_single, args.p_multi)
    if cmd == "reliability-test":
        agent.sync()
        return run_reliability_test(agent, args.seeds, args.deadline, args.delay)

    try:
        if cmd in {"sync", "recommend", "chat", "proactive", "export"}:
            report = agent.sync()
            print_sync(report, verbose=(cmd == "sync" and args.report))
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        if cmd == "sync" or not agent.storage.active_listings():
            return 1
        print("  -> 이전에 저장된 매물 데이터로 계속합니다.")

    if cmd == "recommend":
        req = agent.understand(args.inquiry)
        run_session(agent, req, interactive=not args.batch)
    elif cmd == "chat":
        print("\n찾는 땅 조건을 입력하세요. (예: 음성군 채광이 좋은 2억원대 땅)  종료: q")
        while True:
            try:
                text = input("\n찾는 조건> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in {"q", "quit", "exit", "종료"}:
                break
            if not text:
                continue
            run_session(agent, agent.understand(text))
    elif cmd == "customers":
        if args.remove:
            agent.storage.deactivate_customer(args.remove)
            print(f"'{args.remove}' 님 선제 알림을 중지했습니다.")
        for req in agent.storage.customers():
            print(f" - {req.customer_name}: {req.describe()}")
    elif cmd == "proactive":
        alerts = agent.proactive_scan(threshold=args.threshold, use_llm=True)
        print(f"\n새로 제안할 매물 {len(alerts)}건")
        for a in alerts:
            print_alert(a)
    elif cmd == "watch":
        print(f"폴더 감시 시작 (간격 {args.interval or settings.watch_interval_sec}초, 종료: Ctrl+C)")
        try:
            agent.watch(print_alert, lambda r: print_sync(r), interval=args.interval)
        except KeyboardInterrupt:
            print("\n감시 종료")
    elif cmd == "export":
        export(agent, args.out)
    return 0


def export(agent: RealEstateAgent, out: str) -> None:
    import pandas as pd

    rows = []
    for l in agent.listings():
        rows.append({
            "매물번호": l.listing_id, "소재지": l.address, "지목": l.jimok, "용도지역": l.zoning,
            "면적(㎡)": l.area_m2, "면적(평)": round(l.area_pyeong, 1), "매매가(만원)": l.price_manwon,
            "평당가(만원)": round(l.price_per_pyeong), "도로": l.road_access, "상태": l.status,
            "등록일": l.registered_date, "특이사항": l.description, "출처": l.source,
            "중복출처": ",".join(l.extra.get("duplicate_sources", [])), "원본파일": l.source_file,
        })
    df = pd.DataFrame(rows)
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(out, index=False)
    print(f"{len(rows)}건을 '{out}' 로 내보냈습니다.")


if __name__ == "__main__":
    sys.exit(main())
