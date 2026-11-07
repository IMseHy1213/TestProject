"""버전 태그 v0.1 ~ v1.3 을 순서대로 한 번에 붙이고 GitHub 에 올림.

    python scripts/tag_versions.py            # 붙이고 올리기
    python scripts/tag_versions.py --dry-run  # 무엇을 할지만 보여줌

- 먼저 GitHub 에서 최신 커밋·태그를 받아옴 (git fetch) → 'Failed to resolve ... as a valid ref' 오류가 나지 않음
- 이미 있는 태그(v0.2, v0.3, v0.9 등)는 건너뜀 (덮어쓰지 않음)
- 새 태그만 모아 `git push` 한 번으로 올림
"""

from __future__ import annotations

import argparse
import subprocess
import sys

BRANCH = "claude/real-estate-agent-dev-7stcnv"

# (태그, 커밋 또는 'grep:커밋 제목 시작', 설명) - 커리큘럼 순서
VERSIONS = [
    ("v0.1", "1afce9a", "v0.1 API: GPT/Claude API 연동 (분당 한도·오류 처리)"),
    ("v0.2", "5e9a5cf", "v0.2 Tool Agent: 도구를 골라 쓰는 ReAct Agent"),
    ("v0.3", "1b3b6c7", "v0.3 Stateful: 대화 저장·체크포인트·장기 기억 + FastAPI"),
    ("v0.4", "grep:v0.4 Architecture Study", "v0.4 Architecture Study: Workflow·ReAct·Plan-and-Execute·Multi 비교"),
    ("v0.5", "grep:v0.4 Architecture Study", "v0.5 Framework: LangGraph 판과 직접 구현 비교"),
    ("v0.6", "7576d08", "v0.6 MCP: 매물 도구를 MCP 서버로 공개"),
    ("v0.7", "7576d08", "v0.7 Multi-Agent: 총괄 → 검색·분석 Agent"),
    ("v0.8", "c60fd6f", "v0.8 Distributed/A2A: Agent 를 다른 서버로"),
    ("v0.9", "2d6497d", "v0.9 Realtime: 새 매물·가격 인하 실시간 알림"),
    ("v1.0", "8d5e868", "v1.0 Reliable: 안전장치·시간 예산·장애 주입 시험"),
    ("v1.1", "13b2331", "v1.1 Observable: 추적·지표·관측 화면"),
    ("v1.2", "f1437c7", "v1.2 Online: 손님 반응으로 바로 배우기·자동 상담 방식"),
    ("v1.3", "4739b29", "v1.3 Benchmark: 문제지 채점·동시 처리량·회귀 검사 (+ 연습 기록 분리·정리)"),
]


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 실패:\n{r.stderr.strip()}")
    return r.stdout.strip()


def resolve(ref: str) -> str:
    if ref.startswith("grep:"):
        found = git("log", "--all", "--format=%h", "--fixed-strings", f"--grep={ref[5:]}", "-n", "1", check=False)
        if not found:
            raise RuntimeError(f"'{ref[5:]}' 로 시작하는 커밋을 찾지 못했습니다 (git pull 을 먼저 해 보세요)")
        return found
    if subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], capture_output=True).returncode != 0:
        raise RuntimeError(f"커밋 {ref} 가 이 PC 에 없습니다 (git fetch origin 을 먼저 해 보세요)")
    return ref


def plan(existing: set[str]) -> list[tuple[str, str, str]]:
    out = []
    for tag, ref, message in VERSIONS:
        if tag in existing:
            continue
        out.append((tag, resolve(ref), message))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="버전 태그 v0.1 ~ v1.3 한 번에 붙이기")
    ap.add_argument("--dry-run", action="store_true", help="무엇을 할지만 보여줌")
    ap.add_argument("--no-push", action="store_true", help="이 PC 에만 붙이고 올리지 않음")
    args = ap.parse_args(argv)
    try:
        print("① GitHub 에서 최신 커밋·태그 받아오기 …")
        git("fetch", "origin", BRANCH, "--tags", check=False)
        local = set(git("tag", "--list").split())
        remote = {line.split("refs/tags/")[1].removesuffix("^{}")
                  for line in git("ls-remote", "--tags", "origin", check=False).splitlines() if "refs/tags/" in line}
        existing = local | remote
        todo = plan(existing)
    except RuntimeError as e:
        print(f"[오류] {e}")
        return 1
    for tag, _, _ in VERSIONS:
        if tag in existing:
            print(f"   {tag}: 이미 있음 → 건너뜀")
    if not todo:
        print("✅ 붙일 새 태그가 없습니다. (모두 이미 있음)")
        return 0
    print("② 새로 붙일 태그 (순서대로):")
    for tag, commit, message in todo:
        print(f"   {tag} → {commit}  {message}")
    if args.dry_run:
        print("(--dry-run: 아무것도 바꾸지 않았습니다)")
        return 0
    try:
        for tag, commit, message in todo:
            git("tag", "-a", tag, commit, "-m", message)
        if args.no_push:
            print("✅ 이 PC 에 태그를 붙였습니다 (--no-push).")
            return 0
        print("③ GitHub 에 한 번에 올리기 …")
        git("push", "origin", *[t for t, _, _ in todo])
    except RuntimeError as e:
        print(f"[오류] {e}")
        return 1
    print(f"✅ 태그 {len(todo)}개를 붙이고 올렸습니다: {', '.join(t for t, _, _ in todo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
