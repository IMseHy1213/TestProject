"""사무실 PC(윈도 10·11) 설치 마무리: 바탕화면 아이콘, (선택) 켤 때 자동 실행, (선택) 절전 끄기.

    설치하기.bat 이 마지막에 자동으로 부름:  python main.py windows-setup
    자동 실행만 끄기:                          python main.py windows-setup --autostart off

- 바로가기(.lnk)는 윈도에 기본으로 있는 PowerShell 로 만듭니다 (추가 설치 없음).
- 아이콘 파일은 웹 화면 아이콘(PNG)을 그대로 담은 .ico 로 data/ 에 만듭니다 (윈도 Vista 이후 지원 형식).
- 바탕화면이 OneDrive 로 옮겨진 PC 도 실제 바탕화면 위치를 윈도에 물어서 씁니다.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICON_PNG = PROJECT_ROOT / "real_estate_agent" / "web" / "static" / "icons" / "icon-192.png"
START_BAT = "켜기.bat"
UPDATE_BAT = "최신으로 바꾸기.bat"
SHORTCUT_START = "매물 추천 켜기"
SHORTCUT_UPDATE = "매물 추천 최신으로 바꾸기"
SHORTCUT_AUTOSTART = "매물 추천 (자동 실행)"
YES = {"y", "yes", "ㅛ", "예", "네", "응"}


def make_ico(png: Path, out: Path) -> Path:
    """PNG 한 장을 그대로 담은 .ico (윈도 바로가기 아이콘용)."""
    data = png.read_bytes()
    width = height = 0
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
    w = 0 if width >= 256 else width  # .ico 규칙: 256 이상은 0 으로 적음
    h = 0 if height >= 256 else height
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(data), 6 + 16)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + entry + data)
    return out


def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def shortcut_script(project: Path, icon: Path, autostart: Optional[bool]) -> str:
    """바로가기를 만드는(또는 자동 실행을 지우는) PowerShell 스크립트."""
    start, update = project / START_BAT, project / UPDATE_BAT
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$ws = New-Object -ComObject WScript.Shell",
        "$desk = [Environment]::GetFolderPath('Desktop')",
        "$startup = [Environment]::GetFolderPath('Startup')",
        "function Make($file, $target, $arg, $style, $desc) {",
        "  $s = $ws.CreateShortcut($file)",
        "  $s.TargetPath = $target",
        "  $s.Arguments = $arg",
        f"  $s.WorkingDirectory = {_ps_quote(project)}",
        f"  $s.IconLocation = {_ps_quote(icon)}",
        "  $s.WindowStyle = $style",
        "  $s.Description = $desc",
        "  $s.Save()",
        "  Write-Output ('  OK ' + $file)",
        "}",
        f"Make (Join-Path $desk {_ps_quote(SHORTCUT_START + '.lnk')}) {_ps_quote(start)} '' 1 "
        f"{_ps_quote('토지·공장 매물 추천 프로그램 켜기')}",
        f"Make (Join-Path $desk {_ps_quote(SHORTCUT_UPDATE + '.lnk')}) {_ps_quote(update)} '' 1 "
        f"{_ps_quote('GitHub 에서 최신 코드 받기')}",
    ]
    auto = f"(Join-Path $startup {_ps_quote(SHORTCUT_AUTOSTART + '.lnk')})"
    if autostart is True:  # 컴퓨터를 켜면 작은 창(최소화)으로 켜고 브라우저는 열지 않음
        lines.append(f"Make {auto} {_ps_quote(start)} '--no-browser' 7 {_ps_quote('컴퓨터를 켜면 매물 추천 프로그램 자동 실행')}")
    elif autostart is False:
        lines.append(f"if (Test-Path {auto}) {{ Remove-Item {auto}; Write-Output '  자동 실행을 껐습니다' }}")
    return "\n".join(lines) + "\n"


def run_powershell(script: str) -> subprocess.CompletedProcess:
    # 한글이 깨지지 않게 UTF-8(BOM) 파일로 저장해 실행 (윈도 PowerShell 5.1 은 BOM 이 있어야 UTF-8 로 읽음)
    fd, path = tempfile.mkstemp(suffix=".ps1")
    os.close(fd)
    try:
        Path(path).write_text(script, encoding="utf-8-sig")
        return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _ask(ask: Callable[[str], str], question: str) -> bool:
    try:
        return ask(question).strip().lower() in YES
    except EOFError:
        return False


def status_lines(project: Path = PROJECT_ROOT) -> list[str]:
    """옮겨 와야 하는 파일(GitHub 에 없는 것)이 있는지."""
    from .config import Settings

    s = Settings()
    data_dir = Path(s.data_dir)
    files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".xlsx", ".xlsm", ".xls", ".csv"}
             and not f.name.startswith("~$")] if data_dir.exists() else []
    out = [f"  매물 폴더: {data_dir}  → 엑셀·CSV {len(files)}개"
           + ("" if files else "   ⚠ 비어 있음: 노트북의 '매물' 폴더를 USB 로 이 폴더에 복사하세요")]
    keyfile = project / "my_api_key.py"
    out.append("  AI 키 파일(my_api_key.py): " + ("있음" if keyfile.exists() else
               "없음 - 처음 켤 때 자동으로 만들어집니다. 키는 관리자 화면 'AI 연결' 에서 넣어도 됩니다"))
    pw = Path(s.db_path).parent / "admin_password.json"
    out.append("  관리자 비밀번호: " + ("정해져 있음" if pw.exists() or os.getenv("REA_ADMIN_PASSWORD") else
               "아직 없음 - 처음 관리자 화면에 들어가면 정하라고 나옵니다"))
    return out


def run(autostart: str = "ask", ask: Callable[[str], str] = input, project: Path = PROJECT_ROOT,
        powershell: Callable[[str], subprocess.CompletedProcess] = run_powershell, platform: str = sys.platform) -> int:
    if not platform.startswith("win"):
        print("이 명령은 윈도 PC(10·11)에서만 씁니다.")
        return 1
    print("\n바탕화면 아이콘 만드는 중 …")
    icon = make_ico(ICON_PNG, project / "data" / "매물추천.ico")
    if autostart == "ask":
        on = _ask(ask, "\n컴퓨터를 켜면 프로그램이 자동으로 켜지게 할까요? (사무실 PC 추천) [y/N]: ")
    else:
        on = autostart == "on"
    r = powershell(shortcut_script(project, icon, on))
    if r.returncode != 0:
        print("❌ 바로가기를 만들지 못했습니다:\n" + (r.stderr or r.stdout).strip()[-800:])
        print(f"   대신 이 폴더의 '{START_BAT}' 를 마우스 오른쪽 → 보내기 → 바탕 화면에 바로 가기 만들기 를 쓰세요.")
        return 1
    print((r.stdout or "").rstrip())
    print(f"✅ 바탕화면에 '{SHORTCUT_START}', '{SHORTCUT_UPDATE}' 아이콘을 만들었습니다.")
    if on:
        print("✅ 컴퓨터를 켜면 작은 창으로 자동 실행됩니다 (끄려면: python main.py windows-setup --autostart off)")

    if autostart == "ask" and _ask(ask, "\n전원이 꽂혀 있을 때 PC 가 절전 모드로 들어가지 않게 할까요?\n"
                                        "(절전 모드가 되면 휴대폰에서 접속이 안 됩니다. 화면 꺼짐은 그대로) [y/N]: "):
        p = subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print("✅ 절전 모드를 껐습니다 (전원 연결 시)." if p.returncode == 0 else
              "⚠ 절전 설정을 바꾸지 못했습니다. 설정 → 시스템 → 전원 및 절전 → 절전 모드: '안 함' 으로 바꿔 주세요.")

    print("\n지금 이 PC 상태:")
    for line in status_lines(project):
        print(line)
    return 0
