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

    tp = sub.add_parser("template", help="매물 입력 양식(엑셀) 만들기")
    tp.add_argument("--out", default="", help="저장 위치 (기본: 매물 폴더 안)")

    sub.add_parser("setup-key", help="OpenAI API 키를 .env 에 저장 (입력 내용은 화면에 안 보임)")
    sub.add_parser("llm-check", help="GPT/Claude 연결·모델 점검")

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
    if args.no_llm:
        settings.llm_providers = ["rule"]
    cmd = args.command or "web"  # 아무 명령 없이 실행하면 웹 서비스
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
