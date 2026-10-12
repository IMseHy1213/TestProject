"""Real Estate Recommendation Agent 실행 파일.

VS Code: 이 파일을 열고 F5 (또는 오른쪽 위 ▷ 버튼)
터미널  : python main.py chat
"""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# 실행 위치(작업 폴더)와 상관없이 real_estate_agent 패키지를 찾을 수 있도록
sys.path.insert(0, str(PROJECT_DIR))

REQUIRED = {"pandas": "pandas", "openpyxl": "openpyxl", "dotenv": "python-dotenv", "requests": "requests"}


def missing_packages() -> list[str]:
    importlib.invalidate_caches()
    return [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]


def check_packages() -> bool:
    missing = missing_packages()
    if not missing:
        return True
    requirements = PROJECT_DIR / "requirements.txt"
    print("[설치 필요] 다음 패키지가 지금 사용 중인 파이썬에 설치되어 있지 않습니다: " + ", ".join(missing))
    print(f"  사용 중인 파이썬: {sys.executable}")

    if sys.stdin.isatty():
        answer = input("  지금 자동으로 설치할까요? (1~3분 걸립니다) [Y/n]: ").strip().lower()
        if answer in {"", "y", "yes", "ㅛ", "예", "네"}:
            print("  설치 중... (처음 한 번만 필요합니다)\n")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
            if result.returncode == 0 and not missing_packages():
                print("\n  ✅ 설치 완료! 프로그램을 시작합니다.\n")
                return True
            print("\n  ❌ 자동 설치에 실패했습니다. 위의 오류 메시지를 확인해 주세요.")

    print("  직접 설치하려면 VS Code 터미널(Ctrl+`)에 아래 한 줄을 그대로 붙여넣으세요:")
    print(f'    & "{sys.executable}" -m pip install -r "{requirements}"')
    return False


if __name__ == "__main__":
    if not check_packages():
        sys.exit(1)
    from real_estate_agent.cli import main

    sys.exit(main())
