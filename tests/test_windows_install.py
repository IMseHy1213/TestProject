"""사무실 PC(윈도 10·11) 설치 파일: 설치하기.bat · 켜기.bat · 최신으로 바꾸기.bat + windows-setup."""

import re
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from real_estate_agent import updater, windows_setup

ROOT = Path(__file__).resolve().parent.parent
BATS = ["설치하기.bat", "켜기.bat", "최신으로 바꾸기.bat"]


def bat_text(name):
    raw = (ROOT / name).read_bytes()
    return raw, raw.decode("cp949")


@pytest.mark.parametrize("name", BATS)
def test_bat_encoding_and_line_endings(name):
    """한국어 윈도 cmd 가 읽는 CP949 + CRLF. 첫 줄에서 코드페이지를 949 로 고정 (UTF-8 베타 설정 PC 대비)."""
    raw, text = bat_text(name)
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")  # 모든 줄이 CRLF
    assert not raw.startswith(b"\xef\xbb\xbf")  # BOM 이 있으면 첫 줄 @echo off 가 깨짐
    lines = text.split("\r\n")
    assert lines[0] == "@echo off" and lines[1] == "chcp 949 >nul"
    assert 'cd /d "%~dp0"' in text  # 바로가기·다른 폴더에서 눌러도 프로젝트 폴더에서 실행


@pytest.mark.parametrize("name", BATS)
def test_bat_labels_and_blocks(name):
    """goto/call 대상 라벨이 모두 있고, 괄호 블록이 한 줄 안에서 짝이 맞음 (따옴표 밖 기준)."""
    _, text = bat_text(name)
    labels = {m.group(1).lower() for m in re.finditer(r"^:(\w+)", text, re.M)}
    for target in re.findall(r"\b(?:goto|call) :?(\w+)", text, re.I):
        if target.lower() != "eof":
            assert target.lower() in labels, f"{name}: 없는 라벨 {target}"
    for line in text.split("\r\n"):
        if line.lower().startswith(("rem ", "echo", "title")):
            continue
        outside = re.sub(r'"[^"]*"', "", line)
        assert outside.count("(") == outside.count(")"), f"{name}: 괄호 짝 {line}"


def test_install_bat_steps():
    _, text = bat_text("설치하기.bat")
    assert "sys.version_info>=(3,10)" in text and "Python.Python.3.12" in text  # 파이썬 없으면 winget 으로
    assert "python.org/downloads" in text  # winget 이 없는 윈도 10 은 홈페이지 안내
    assert "rmdir /s /q \".venv\"" in text  # 노트북에서 복사해 온 .venv 는 다시 만듦
    assert "pip install -r requirements.txt" in text and "Git.Git" in text
    assert "main.py windows-setup" in text


def test_start_bat_uses_venv_and_reports_already_running():
    _, text = bat_text("켜기.bat")
    assert '".venv\\Scripts\\python.exe" main.py mobile %*' in text  # 자동 실행 바로가기는 --no-browser 를 넘김
    assert "if errorlevel 3 if not errorlevel 4 goto already" in text


def test_gitattributes_keeps_bat_bytes():
    assert "*.bat -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_make_ico_wraps_png(tmp_path):
    ico = windows_setup.make_ico(windows_setup.ICON_PNG, tmp_path / "a.ico")
    data = ico.read_bytes()
    assert struct.unpack("<HHH", data[:6]) == (0, 1, 1)
    w, h, _, _, planes, bpp, size, offset = struct.unpack("<BBBBHHII", data[6:22])
    assert (w, h, planes, bpp, offset) == (192, 192, 1, 32, 22) and data[22:30] == b"\x89PNG\r\n\x1a\n"
    assert size == len(data) - 22


def test_shortcut_script_desktop_and_autostart(tmp_path):
    project = Path("C:/Users/사장/매물 추천's")
    on = windows_setup.shortcut_script(project, tmp_path / "i.ico", True)
    assert "GetFolderPath('Desktop')" in on and "GetFolderPath('Startup')" in on  # OneDrive 바탕화면도
    assert "매물 추천 켜기.lnk" in on and "매물 추천 최신으로 바꾸기.lnk" in on
    assert "'--no-browser' 7" in on  # 자동 실행은 최소화 + 브라우저 안 염
    assert "매물 추천''s" in on  # 경로의 작은따옴표 이스케이프
    off = windows_setup.shortcut_script(project, tmp_path / "i.ico", False)
    assert "Remove-Item" in off and "'--no-browser'" not in off
    assert "Startup" in off and "Remove-Item" not in windows_setup.shortcut_script(project, tmp_path / "i.ico", None)


def test_windows_setup_run(tmp_path, monkeypatch):
    seen = []
    answers = iter(["y", "n"])  # 자동 실행: 예, 절전 끄기: 아니오
    ok = windows_setup.run(ask=lambda q: next(answers), project=tmp_path, platform="win32",
                           powershell=lambda s: seen.append(s) or SimpleNamespace(returncode=0, stdout="  OK", stderr=""))
    assert ok == 0 and "'--no-browser' 7" in seen[0] and (tmp_path / "data" / "매물추천.ico").exists()
    assert windows_setup.run(platform="linux") == 1
    fail = windows_setup.run(autostart="off", project=tmp_path, platform="win32",
                             powershell=lambda s: SimpleNamespace(returncode=1, stdout="", stderr="blocked"))
    assert fail == 1


def test_update_without_git_or_from_zip(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    assert updater.run_update(yes=True) == 1 and "Git 이 설치되어 있지 않아" in capsys.readouterr().out

    calls = []
    monkeypatch.setattr(updater.shutil, "which", lambda name: "git")
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    (tmp_path / "main.py").write_text("")

    def fake_git(*args, check=False):
        calls.append(args)
        rc = 128 if args[0] == "rev-parse" else 0
        return subprocess.CompletedProcess(args, rc, stdout="abc|2026-11-25 10:00|msg" if args[0] == "log" else "", stderr="")

    monkeypatch.setattr(updater, "_git", fake_git)
    assert updater.run_update(ask=lambda q: "n") == 1 and len(calls) == 1  # 묻고 취소하면 아무것도 안 함
    assert updater.run_update(ask=lambda q: "y") == 0
    steps = [c[0] for c in calls[1:]]
    assert steps[:5] == ["rev-parse", "init", "remote", "fetch", "checkout"]
    assert ("remote", "add", "origin", updater.REPO_URL) in calls
    assert ("checkout", "-f", "-B", updater.BRANCH, "--track", f"origin/{updater.BRANCH}") in calls
