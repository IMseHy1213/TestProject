"""환경 설정.

모든 값은 환경변수 또는 프로젝트 루트의 `.env` 파일로 바꿀 수 있습니다.
(`.env.example` 참고)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 매물 자료(.xlsx) 폴더: 프로젝트 폴더 안의 '매물' 폴더
DEFAULT_DATA_DIR = PROJECT_ROOT / "매물"
# 예전 버전의 기본값. .env 에 이 값이 남아 있어도 프로젝트의 '매물' 폴더를 사용
LEGACY_DATA_DIRS = {r"E:\부동산매물백업\매물폴더\토지", "E:/부동산매물백업/매물폴더/토지"}

KEY_FILE = PROJECT_ROOT / "my_api_key.py"  # 코드에 직접 넣는 API 키 (GitHub 에 올라가지 않음)
KEY_FILE_EXAMPLE = PROJECT_ROOT / "my_api_key.example.py"
KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_MODEL", "ANTHROPIC_API_KEY")


def load_keys() -> list[str]:
    """API 키를 읽어 환경변수에 넣음. 우선순위: my_api_key.py (코드에 넣은 키) > .env > 시스템 환경변수.
    반환: 키를 찾은 곳 목록 (화면 안내용)."""
    found = []
    try:  # python-dotenv 는 선택 사항. 실행 위치와 상관없이 프로젝트 폴더의 .env 를 읽음
        from dotenv import load_dotenv

        if load_dotenv(PROJECT_ROOT / ".env", override=True):
            found.append(".env")
    except ImportError:  # pragma: no cover
        pass
    if not KEY_FILE.exists() and KEY_FILE_EXAMPLE.exists():
        try:  # 처음 실행이면 키를 붙여넣을 빈 파일을 만들어 둠
            KEY_FILE.write_text(KEY_FILE_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    values = read_key_file()
    for name, value in values.items():
        if value:
            os.environ[name] = value
    if values.get("OPENAI_API_KEY"):
        found.append("my_api_key.py")
    return found


def read_key_file() -> dict[str, str]:
    """my_api_key.py 를 '실행하지 않고' 읽음 (NAME = "값" 줄만). 잘못 적어도 프로그램이 멈추지 않도록."""
    import re

    out: dict[str, str] = {}
    try:
        text = KEY_FILE.read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for name in KEY_NAMES:
        m = re.search(rf'^\s*{name}\s*=\s*["\']([^"\']*)["\']', text, re.M)
        if m and m.group(1).strip():
            out[name] = m.group(1).strip()
    return out


KEY_SOURCES = load_keys()


def _data_dir_from_env() -> Path:
    value = os.getenv("REA_DATA_DIR", "").strip()
    if not value or value in LEGACY_DATA_DIRS:
        return DEFAULT_DATA_DIR
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path  # 상대 경로는 프로젝트 폴더 기준


def _env_list(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: _data_dir_from_env())
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("REA_DB_PATH", str(PROJECT_ROOT / "data" / "agent.db")))
    )
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("REA_LOG_DIR", str(PROJECT_ROOT / "logs"))))

    # LLM: 앞에서부터 순서대로 시도하고, 모두 실패하면 규칙 기반(rule)으로 동작합니다.
    llm_providers: list[str] = field(default_factory=lambda: _env_list("LLM_PROVIDERS", "openai,anthropic,rule"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))
    # OpenAI 모델 ID는 계정에서 실제 사용 가능한 이름으로 바꿔주세요 (예: GPT-5.6 Luna).
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))

    # 공공데이터 / GIS (선택)
    data_go_kr_key: str = field(default_factory=lambda: os.getenv("DATA_GO_KR_SERVICE_KEY", ""))
    vworld_key: str = field(default_factory=lambda: os.getenv("VWORLD_API_KEY", ""))

    # 선제 추천: 이 점수 이상인 신규 매물만 알림
    proactive_threshold: float = field(default_factory=lambda: float(os.getenv("REA_PROACTIVE_THRESHOLD", "70")))
    watch_interval_sec: int = field(default_factory=lambda: int(os.getenv("REA_WATCH_INTERVAL", "600")))


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, Path(v) if k in {"data_dir", "db_path", "log_dir"} else v)
    return s
