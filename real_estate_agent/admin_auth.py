"""관리자 비밀번호 관리.

비밀번호를 정하는 방법 (둘 중 하나)
  1) 웹 화면: 처음 /admin 에 들어가면 '관리자 비밀번호 정하기' 화면이 나옴 (서버를 켠 PC 에서만)
  2) VS Code 실행 목록 '⓪ 관리자 비밀번호 정하기' 또는  python main.py set-admin-password

비밀번호는 그대로 저장하지 않고 '해시(복원할 수 없는 값)' 로만 data/admin_password.json 에 저장합니다.
(data/ 폴더는 .gitignore 에 있어 GitHub 에 올라가지 않음)
.env 에 REA_ADMIN_PASSWORD 를 적어 두었다면 그 값이 우선합니다.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

MIN_LENGTH = 8
MAX_FAILURES = 5          # 이 횟수만큼 틀리면
LOCK_SECONDS = 5 * 60     # 이 시간 동안 로그인 잠금


class AdminAuth:
    def __init__(self, path: Path, env_password: Optional[str] = None):
        self.path = Path(path)
        self.env_password = os.getenv("REA_ADMIN_PASSWORD", "") if env_password is None else env_password
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 상태
    @property
    def source(self) -> str:
        """'env' (.env 의 REA_ADMIN_PASSWORD) / 'file' (프로그램에서 정한 비밀번호) / '' (아직 없음)"""
        if self.env_password:
            return "env"
        return "file" if self._stored_hash() else ""

    @property
    def configured(self) -> bool:
        return bool(self.source)

    def _stored_hash(self) -> str:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return str(data.get("hash", ""))
        except (OSError, ValueError, AttributeError):
            return ""

    # ------------------------------------------------------------------ 확인·변경
    def check(self, password: str) -> bool:
        if self.env_password:
            return hmac.compare_digest(password.encode(), self.env_password.encode())
        stored = self._stored_hash()
        return bool(stored) and check_password_hash(stored, password)

    def set_password(self, new: str, confirm: Optional[str] = None) -> None:
        """새 비밀번호 저장. 규칙에 맞지 않으면 ValueError (화면에 그대로 보여줄 안내 문장)."""
        if self.env_password:
            raise ValueError(".env 의 REA_ADMIN_PASSWORD 로 비밀번호가 정해져 있습니다. "
                             "바꾸려면 .env 에서 그 줄을 고치거나 지운 뒤 서버를 다시 켜세요.")
        problem = password_problem(new, confirm)
        if problem:
            raise ValueError(problem)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"hash": generate_password_hash(new),
                                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------ 무차별 대입 방지
    def locked_for(self, who: str) -> int:
        """잠겨 있으면 남은 초, 아니면 0."""
        with self._lock:
            recent = [t for t in self._failures.get(who, []) if time.time() - t < LOCK_SECONDS]
            self._failures[who] = recent
            if len(recent) >= MAX_FAILURES:
                return int(LOCK_SECONDS - (time.time() - recent[0])) + 1
            return 0

    def record_failure(self, who: str) -> None:
        with self._lock:
            self._failures.setdefault(who, []).append(time.time())

    def clear_failures(self, who: str) -> None:
        with self._lock:
            self._failures.pop(who, None)


def password_problem(new: str, confirm: Optional[str] = None) -> str:
    if len(new) < MIN_LENGTH:
        return f"비밀번호는 {MIN_LENGTH}자 이상으로 정해 주세요."
    if len(new) > 200:
        return "비밀번호가 너무 깁니다 (200자 이하)."
    if new.strip() != new:
        return "비밀번호 앞뒤에 띄어쓰기를 넣을 수 없습니다."
    if confirm is not None and not hmac.compare_digest(new.encode(), confirm.encode()):
        return "두 번 입력한 비밀번호가 서로 다릅니다."
    return ""


def auth_file(settings) -> Path:
    """비밀번호 해시 파일 위치: DB 와 같은 폴더 (기본: 프로젝트/data/)."""
    return Path(settings.db_path).parent / "admin_password.json"


def set_password_cli(settings) -> int:
    """터미널에서 관리자 비밀번호 정하기 / 바꾸기 (입력 글자는 화면에 안 보임)."""
    import getpass

    auth = AdminAuth(auth_file(settings))
    if auth.source == "env":
        print(".env 의 REA_ADMIN_PASSWORD 로 이미 비밀번호가 정해져 있습니다. .env 에서 바꾸세요.")
        return 1
    print("관리자 비밀번호를 " + ("바꿉니다." if auth.configured else "정합니다.")
          + f" ({MIN_LENGTH}자 이상, 입력하는 글자는 화면에 보이지 않습니다)")
    if auth.configured and not auth.check(getpass.getpass("  지금 비밀번호: ")):
        print("지금 비밀번호가 맞지 않습니다.")
        return 1
    new = getpass.getpass("  새 비밀번호: ")
    confirm = getpass.getpass("  새 비밀번호 한 번 더: ")
    try:
        auth.set_password(new, confirm)
    except ValueError as e:
        print(f"저장하지 못했습니다: {e}")
        return 1
    print(f"저장했습니다. 웹 화면 오른쪽 위 '관리자' 에서 로그인하세요. (저장 위치: {auth.path})")
    return 0
