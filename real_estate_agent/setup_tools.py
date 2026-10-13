"""API 키 설정(setup-key)과 LLM 연결 점검(llm-check).

API 키는 프로젝트 폴더의 .env 파일에만 저장합니다. (.gitignore 에 들어 있어 GitHub 에 올라가지 않음)
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from .config import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"
EXAMPLE_PATH = PROJECT_ROOT / ".env.example"


def update_env_file(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """.env 의 해당 항목만 바꾸고 나머지 줄(주석 포함)은 그대로 둠. 없으면 .env.example 을 복사해 시작."""
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    elif EXAMPLE_PATH.exists() and path == ENV_PATH:
        lines = EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    remaining = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    lines += [f"{k}={v}" for k, v in remaining.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask(key: str) -> str:
    return key[:7] + "…" + key[-4:] if len(key) > 12 else "***"


def setup_key() -> int:
    from .config import KEY_FILE

    print("OpenAI API 키를 저장합니다. (입력하는 글자는 화면에 보이지 않습니다)")
    print(f"  저장 위치: {KEY_FILE}  (GitHub 에 올라가지 않는 파일)")
    key = getpass.getpass("  OpenAI API 키 붙여넣기 후 Enter: ").strip()
    if not key:
        print("입력이 없어 취소했습니다.")
        return 1
    model = input("  사용할 GPT 모델 이름 (그냥 Enter = gpt-5.6-luna): ").strip() or "gpt-5.6-luna"
    try:
        save_key_to_code_file(key, model if model != "gpt-5.6-luna" else "")
    except ValueError as e:
        print(f"저장하지 못했습니다: {e}")
        return 1
    print(f"저장했습니다: {KEY_FILE.name} 의 OPENAI_API_KEY={mask(key)}, OPENAI_MODEL={model}")
    print("이제 'python main.py llm-check' 로 연결을 확인해 보세요.")
    return 0


def llm_check(agent) -> int:
    """설정된 LLM 에 실제로 요청을 보내 연결·모델·응답을 점검."""
    from .llm.openai_client import OpenAILLM

    router = agent.llm
    print(router.status())
    if not router.available:
        print("\n사용 가능한 LLM 이 없습니다. 'python main.py setup-key' 로 OpenAI 키를 저장하세요.")
        return 1
    ok = True
    for client in list(router.clients):
        print(f"\n[{client.name}] 모델: {getattr(client, 'model', '-')}")
        if isinstance(client, OpenAILLM):
            try:
                models = client.available_models()
                gpts = [m for m in models if m.startswith("gpt-5")]
                print(f"  계정에서 쓸 수 있는 모델 {len(models)}개 (GPT-5 계열: {', '.join(gpts[:12]) or '없음'})")
                if client.model not in models:
                    print(f"  ⚠ '{client.model}' 은 목록에 없습니다. 첫 요청 때 비슷한 모델로 자동 교체됩니다.")
            except Exception as e:
                print(f"  모델 목록 조회 실패: {e}")
        try:
            data = client.complete_json(
                "토지 매물 문의에서 조건을 뽑는 도우미입니다. regions(목록), budget_min, budget_max(만원), "
                "keywords(목록) 를 가진 JSON 으로 답하세요.",
                "음성군 채광이 좋은 2억원대 땅",
            )
            print(f"  ✅ 응답 성공 (실제 사용 모델: {getattr(client, 'model', '-')}): {data}")
        except Exception as e:
            ok = False
            print(f"  ❌ 실패: {e}")
    return 0 if ok else 1


def save_key_to_code_file(key: str, model: str = "") -> None:
    """관리자 화면에서 입력한 키를 my_api_key.py 에 저장 (GitHub 에 올라가지 않는 파일)."""
    import re as _re

    from .config import KEY_FILE, KEY_FILE_EXAMPLE

    key = key.strip()
    if not _re.fullmatch(r"sk-[A-Za-z0-9_\-]{20,}", key):
        raise ValueError("OpenAI API 키 형식이 아닙니다 ('sk-' 로 시작하는 긴 문자열)")
    if model and not _re.fullmatch(r"[A-Za-z0-9._\-]{2,60}", model.strip()):
        raise ValueError("모델 이름 형식이 올바르지 않습니다")
    text = (KEY_FILE.read_text(encoding="utf-8") if KEY_FILE.exists()
            else KEY_FILE_EXAMPLE.read_text(encoding="utf-8") if KEY_FILE_EXAMPLE.exists()
            else 'OPENAI_API_KEY = ""\nOPENAI_MODEL = ""\n')
    for name, value in (("OPENAI_API_KEY", key), ("OPENAI_MODEL", model.strip())):
        if name == "OPENAI_MODEL" and not value:
            continue
        line = f'{name} = "{value}"'
        if _re.search(rf"^\s*{name}\s*=.*$", text, _re.M):
            text = _re.sub(rf"^\s*{name}\s*=.*$", lambda _m: line, text, count=1, flags=_re.M)
        else:
            text += f"\n{line}\n"
    KEY_FILE.write_text(text, encoding="utf-8")
    os.environ["OPENAI_API_KEY"] = key
    if model.strip():
        os.environ["OPENAI_MODEL"] = model.strip()
