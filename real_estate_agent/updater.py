"""최신 코드 받기 (python main.py update) + 지금 돌고 있는 프로그램 버전.

git pull 이 'Your local changes … would be overwritten by merge … Aborting' 으로 멈추면 파일이 하나도 바뀌지 않아
예전 프로그램이 그대로 실행됩니다. 이 명령은 그런 경우를 알아서 처리합니다:
  1) GitHub 에서 최신 코드를 받아 옴 (git fetch)
  2) 이 PC 에서 바뀐 파일이 있으면 목록을 보여 주고, 허락하면 '보관함(git stash)' 에 옮겨 둠 (지우지 않음)
     my_api_key.py, .env, 매물/, data/, logs/ 는 git 이 다루지 않는 파일이라 그대로 남음
  3) 최신 코드로 바꿈 (git merge --ff-only: 이 PC 의 기록을 덮어쓰거나 합치지 않음)
  4) 무엇이 바뀌었는지 보여 주고, 프로그램을 껐다 켜라고 안내
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRANCH = "claude/real-estate-agent-dev-7stcnv"


def _git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=check)


def version() -> dict:
    """지금 폴더의 코드 버전: {'commit': '76c0ccc', 'date': '2026-11-23', 'subject': …} (git 이 없으면 빈 값)."""
    try:
        r = _git("log", "-1", "--format=%h|%cd|%s", "--date=format:%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return {"commit": "", "date": "", "subject": ""}
    if r.returncode != 0 or "|" not in r.stdout:
        return {"commit": "", "date": "", "subject": ""}
    commit, date, subject = r.stdout.strip().split("|", 2)
    return {"commit": commit, "date": date, "subject": subject}


def version_text() -> str:
    v = version()
    return f"{v['commit']} ({v['date']})" if v["commit"] else "알 수 없음 (git 폴더가 아님)"


def _changed_files() -> list[str]:
    """git 이 관리하는 파일 중 이 PC 에서 바뀐 것 (새로 만든 파일은 제외)."""
    r = _git("status", "--porcelain", "--untracked-files=no")
    return [line[3:] for line in r.stdout.splitlines() if line.strip()]


def run_update(yes: bool = False, branch: str = BRANCH, ask=input) -> int:
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        print("❌ 이 폴더는 git 폴더가 아닙니다. GitHub 에서 받은(clone) 프로젝트 폴더에서 실행하세요.")
        return 1
    before = version()
    print(f"지금 코드: {before['commit']} ({before['date']}) {before['subject']}")
    cur = _git("branch", "--show-current").stdout.strip()
    print("① GitHub 에서 최신 코드 받는 중 …")
    f = _git("fetch", "origin", branch)
    if f.returncode != 0:
        print("❌ 받아오기 실패 (인터넷 연결이나 GitHub 로그인 확인):\n" + (f.stderr.strip() or f.stdout.strip()))
        return 1
    remote = f"origin/{branch}"
    behind = _git("rev-list", "--count", f"HEAD..{remote}").stdout.strip() or "0"
    if cur != branch:
        print(f"   지금 가지(branch)는 '{cur or '(없음)'}' 입니다. 작업 가지 '{branch}' 로 바꿉니다.")
    elif behind == "0":
        print("✅ 이미 최신입니다. 화면이 예전 그대로면 프로그램을 껐다가 다시 켜고, 브라우저에서 Ctrl+F5 를 누르세요.")
        return 0
    else:
        print(f"   새로 받을 변경: {behind}개")

    changed = _changed_files()
    if changed:
        print("② 이 PC 에서 바뀐 파일이 있어 그대로는 최신 코드로 바꿀 수 없습니다 (git pull 이 멈추던 이유):")
        for name in changed[:20]:
            print(f"     · {name}")
        print("   이 파일들을 '보관함(git stash)' 에 옮겨 두고 최신 코드로 바꿉니다. 지우지 않으니 나중에 되살릴 수 있습니다.")
        print("   (my_api_key.py · .env · 매물 폴더 · data · logs 는 해당 없음 - 그대로 남습니다)")
        if not yes and ask("   진행할까요? (y/N) ").strip().lower() not in {"y", "yes", "ㅛ", "예", "네"}:
            print("취소했습니다. 아무것도 바꾸지 않았습니다.")
            return 1
        s = _git("stash", "push", "-m", "update 전 자동 보관 (python main.py update)")
        if s.returncode != 0:
            print("❌ 보관 실패:\n" + s.stderr.strip())
            return 1
        print("   ✔ 보관함에 옮겼습니다. 되살리려면: git stash list → git stash pop")

    print("③ 최신 코드로 바꾸는 중 …")
    if cur != branch:
        sw = _git("switch", branch) if _git("rev-parse", "--verify", "--quiet", branch).returncode == 0 \
            else _git("switch", "-c", branch, "--track", remote)
        if sw.returncode != 0:
            print("❌ 가지 바꾸기 실패:\n" + sw.stderr.strip())
            return 1
    m = _git("merge", "--ff-only", remote)
    if m.returncode != 0:
        print("❌ 최신 코드로 바꾸지 못했습니다 (이 PC 에만 있는 커밋이 있음):\n" + (m.stderr.strip() or m.stdout.strip()))
        print(f"   도움이 필요하면 이 메시지를 그대로 알려 주세요. (지금 코드: {version_text()})")
        return 1
    after = version()
    print(f"✅ 최신 코드로 바꿨습니다: {before['commit']} → {after['commit']} ({after['date']})")
    log = _git("log", "--format=   · %s", f"{before['commit']}..{after['commit']}").stdout.strip()
    if log:
        print("   이번에 들어온 변경:\n" + "\n".join(log.splitlines()[:15]))
    req = _git("diff", "--name-only", f"{before['commit']}..{after['commit']}", "--", "requirements.txt").stdout.strip()
    if req:
        print("   requirements.txt 가 바뀌었습니다 → 먼저 실행:  pip install -r requirements.txt")
    print("④ 켜져 있는 프로그램을 끄고(터미널에서 Ctrl+C) 다시 켜세요. 브라우저는 Ctrl+F5 로 새로고침.")
    return 0


def port_in_use(host: str, port: int) -> bool:
    """다른 프로그램(예: 예전에 켜 둔 이 프로그램)이 이미 이 포트를 쓰고 있는지."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1" if host in {"0.0.0.0", ""} else host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - 명령줄 진입점
    return run_update(yes="--yes" in (argv or sys.argv[1:]))
