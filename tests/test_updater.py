"""git pull 이 '이 PC 에서 바뀐 파일' 때문에 멈춰 예전 프로그램이 돌던 문제 -> python main.py update 가 처리."""

import socket
import subprocess
from pathlib import Path

import pytest

import real_estate_agent.updater as updater


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """GitHub(origin) + 부모님 PC(clone). origin 에 새 커밋이 생기고, PC 에서는 같은 파일을 고쳐 둔 상황."""
    origin, dev, pc = tmp_path / "origin.git", tmp_path / "dev", tmp_path / "pc"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(dev)], check=True)
    for r in (dev,):
        git(r, "config", "user.email", "t@example.com")
        git(r, "config", "user.name", "t")
    git(dev, "switch", "-q", "-c", "work")
    (dev / "my_api_key.example.py").write_text("OPENAI_API_KEY = ''\n", encoding="utf-8")
    (dev / "app.py").write_text("form = False\n", encoding="utf-8")
    git(dev, "add", ".")
    git(dev, "commit", "-q", "-m", "11-17 version")
    git(dev, "push", "-q", "origin", "work")
    subprocess.run(["git", "clone", "-q", "-b", "work", str(origin), str(pc)], check=True)
    git(pc, "config", "user.email", "t@example.com")
    git(pc, "config", "user.name", "t")
    # GitHub 쪽 새 코드 (11-23): 같은 파일을 바꿈
    (dev / "my_api_key.example.py").write_text("OPENAI_API_KEY = ''\nSMS_TO = ''\n", encoding="utf-8")
    (dev / "app.py").write_text("form = True\n", encoding="utf-8")
    git(dev, "commit", "-q", "-am", "11-23 매물 접수 form")
    git(dev, "push", "-q", "origin", "work")
    # 부모님 PC 에서 고쳐 둔 파일 + git 이 다루지 않는 키 파일
    (pc / "my_api_key.example.py").write_text("OPENAI_API_KEY = 'edited on pc'\n", encoding="utf-8")
    (pc / ".gitignore").exists()
    monkeypatch.setattr(updater, "PROJECT_ROOT", pc)
    return pc


def test_plain_pull_aborts_like_on_the_pc(repos):
    r = subprocess.run(["git", "pull", "origin", "work"], cwd=repos, capture_output=True, text=True)
    assert r.returncode != 0 and "would be overwritten" in r.stderr
    assert (repos / "app.py").read_text() == "form = False\n"  # 아무것도 안 바뀜 = 예전 프로그램


def test_update_stashes_local_changes_and_gets_latest(repos, capsys):
    assert updater.run_update(yes=True, branch="work") == 0
    out = capsys.readouterr().out
    assert (repos / "app.py").read_text() == "form = True\n"
    assert "my_api_key.example.py" in out and "보관함" in out and "11-23 매물 접수 form" in out
    assert "edited on pc" in git(repos, "stash", "show", "-p")  # 고친 내용은 지우지 않고 보관
    assert updater.run_update(yes=True, branch="work") == 0
    assert "이미 최신" in capsys.readouterr().out


def test_update_asks_before_touching_local_changes(repos, capsys):
    assert updater.run_update(branch="work", ask=lambda _: "n") == 1
    assert (repos / "app.py").read_text() == "form = False\n" and "edited on pc" in (repos / "my_api_key.example.py").read_text()


def test_update_switches_from_other_branch(repos):
    git(repos, "checkout", "-q", "-b", "main")
    git(repos, "checkout", "-q", "--", ".")
    assert updater.run_update(yes=True, branch="work") == 0
    assert git(repos, "branch", "--show-current") == "work" and (repos / "app.py").read_text() == "form = True\n"


def test_version_and_port_check(tmp_path, monkeypatch):
    v = updater.version()
    assert len(v["commit"]) >= 7 and v["date"][:2] == "20"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert updater.port_in_use("127.0.0.1", port)
    finally:
        s.close()
    assert not updater.port_in_use("127.0.0.1", port)


def test_web_run_stops_when_old_copy_still_runs(capsys):
    """예전에 켜 둔 프로그램이 8000번을 쓰고 있으면 새 프로그램이 그 사실을 알려 주고 멈춤 (예전 화면이 보이는 원인)."""
    from real_estate_agent.web.app import run

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        run(host="127.0.0.1", port=s.getsockname()[1], open_browser=False)
    finally:
        s.close()
    assert "이미 다른 프로그램이 쓰고 있습니다" in capsys.readouterr().out
