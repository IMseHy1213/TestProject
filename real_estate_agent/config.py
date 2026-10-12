"""환경 설정.

모든 값은 환경변수 또는 프로젝트 루트의 `.env` 파일로 바꿀 수 있습니다.
(`.env.example` 참고)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 부모님 부동산의 토지 매물 백업 폴더 (Windows)
DEFAULT_DATA_DIR = r"E:\부동산매물백업\매물폴더\토지"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # python-dotenv 는 선택 사항. 실행 위치와 상관없이 프로젝트 폴더의 .env 를 읽음
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_list(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("REA_DATA_DIR", DEFAULT_DATA_DIR)))
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
