"""Real Estate Recommendation Agent 실행 파일.

VS Code: 이 파일을 열고 F5 (또는 오른쪽 위 ▷ 버튼)
터미널  : python main.py chat
"""

import importlib.util
import sys
from pathlib import Path

# 실행 위치(작업 폴더)와 상관없이 real_estate_agent 패키지를 찾을 수 있도록
sys.path.insert(0, str(Path(__file__).resolve().parent))

REQUIRED = {"pandas": "pandas", "openpyxl": "openpyxl", "dotenv": "python-dotenv", "requests": "requests"}


def check_packages() -> bool:
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return True
    print("[설치 필요] 다음 패키지가 지금 사용 중인 파이썬에 설치되어 있지 않습니다: " + ", ".join(missing))
    print(f"  사용 중인 파이썬: {sys.executable}")
    print("  VS Code 터미널(Ctrl+`)에서 프로젝트 폴더로 이동한 뒤 아래 명령을 실행하세요:")
    print(f'    "{sys.executable}" -m pip install -r requirements.txt')
    return False


if __name__ == "__main__":
    if not check_packages():
        sys.exit(1)
    from real_estate_agent.cli import main

    sys.exit(main())
