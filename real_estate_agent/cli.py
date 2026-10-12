"""명령줄 인터페이스.

  python main.py sync                         매물 폴더 수집·정제 (오류/중복 보고서)
  python main.py recommend "문의 내용" --name 홍길동   손님 문의로 추천 (대화형: 다음/피드백)
  python main.py chat                         대화형 상담 모드
  python main.py customers                    저장된 손님 목록
  python main.py proactive                    저장된 손님에게 맞는 매물 선제 탐색
  python main.py watch                        폴더 감시 + 새 매물 선제 알림 (무한 반복)
  python main.py export --out 매물정제.xlsx     정제된 매물을 엑셀로 내보내기
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Optional

from .agent import ProactiveAlert, RealEstateAgent, SyncReport
from .config import get_settings
from .models import CustomerRequirement, ScoredListing, format_price
from .recommender import RecommendationSession
from .requirement_parser import parse_rule_based

LINE = "─" * 78

HELP = """명령:  [Enter]/n 다음 순위 매물 · p 이전 · t 상위 5개 · d 상세 · l 마음에 듦
       그냥 말로 입력해도 됩니다:
         '마음에 안 들어요', '너무 비싸요', '채광이 별로'  -> 다음 순위 매물 (사유를 학습)
         '좀 더 싼 걸로', '여주 쪽도', '300평 이상으로'   -> 조건을 바꿔 다시 추천
       w 학습한 손님 취향 · c 현재 조건 · s 손님 조건 저장(새 매물 선제 알림) · q 종료"""

# '이 매물이 별로' 라는 평가 (-> 다음 순위)
_DISLIKE = r"(별로|싫|마음에\s*안|맘에\s*안|안\s*좋|아니|비싸|좁|작아|멀|어둡|그늘|불편|애매|글쎄|패스|다른\s*(거|것|매물))"
# 조건을 바꾸려는 말인지 (아니면 단순히 '이건 별로' 인지) 판단하는 단서
_CONDITION_CUES = r"(더\s|좀\s*더|으로|로\s*(해|바꿔|보여)|바꿔|말고|쪽|이상|이하|이내|평|억|만원|지역|도로\s*(있|접|붙))"


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
        print(" 더 이상 조건에 맞는 매물이 없습니다. 'f <피드백>' 으로 조건을 바꿔보세요.")
        if session.filter_stats:
            print(" (제외 사유: " + ", ".join(f"{k} {v}건" for k, v in session.filter_stats.items()) + ")")
        return
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
            dislike_only = re.search(_DISLIKE, cmd) and not re.search(_CONDITION_CUES, cmd)
            if not dislike_only and (re.search(_CONDITION_CUES, cmd)
                                     or parse_rule_based(cmd).describe() != "(조건 없음)"):
                cmd = "f " + cmd
                new_req = agent.refine(session, cmd[2:])
                print(f"[조건 수정] {new_req.describe()}   (분석: {agent.llm.last_used})")
                print(f"[후보] {session.total}건")
            else:  # '마음에 안 들어요', '너무 비싸요' -> 다음 순위
                agent.reject(session, cmd)
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
    p.add_argument("--data-dir", help="매물 폴더 경로 (기본: E:\\부동산매물백업\\매물폴더\\토지)")
    p.add_argument("--db", help="SQLite DB 경로")
    p.add_argument("--no-llm", action="store_true", help="LLM 없이 규칙 기반으로만 동작")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("sync", help="매물 수집·정제")
    s.add_argument("--report", action="store_true", help="오류/중복 상세 출력")

    r = sub.add_parser("recommend", help="손님 문의로 매물 추천")
    r.add_argument("inquiry", help="손님 문의 내용")
    r.add_argument("--name", default="", help="손님 이름")
    r.add_argument("--save", action="store_true", help="손님 조건 저장 (선제 알림 대상)")
    r.add_argument("--batch", action="store_true", help="대화 없이 상위 5개만 출력")

    sub.add_parser("chat", help="대화형 상담")

    c = sub.add_parser("customers", help="저장된 손님 목록")
    c.add_argument("--remove", help="해당 손님 선제 알림 중지")

    pr = sub.add_parser("proactive", help="저장된 손님 조건으로 전체 매물 선제 탐색")
    pr.add_argument("--threshold", type=float)

    w = sub.add_parser("watch", help="폴더 감시 + 선제 알림")
    w.add_argument("--interval", type=int, help="초 단위 (기본 600)")

    e = sub.add_parser("export", help="정제된 매물 엑셀/CSV 내보내기")
    e.add_argument("--out", default="정제된_토지매물.xlsx")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings(data_dir=args.data_dir, db_path=args.db)
    if args.no_llm:
        settings.llm_providers = ["rule"]
    agent = RealEstateAgent(settings)
    cmd = args.command or "chat"

    print(f"Real Estate Recommendation Agent  |  매물 폴더: {settings.data_dir}")
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
        req = agent.understand(args.inquiry, args.name)
        if args.save and req.customer_name:
            agent.storage.save_customer(req)
        run_session(agent, req, interactive=not args.batch)
    elif cmd == "chat":
        print("\n손님 문의를 입력하세요. (예: 양평 쪽 전원주택 지을 땅 200평 정도, 3억 이하, 맹지 제외)  종료: q")
        while True:
            try:
                text = input("\n문의> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in {"q", "quit", "exit", "종료"}:
                break
            if not text:
                continue
            name = input("손님 이름(선택)> ").strip()
            run_session(agent, agent.understand(text, name))
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
