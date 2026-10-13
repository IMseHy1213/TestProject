from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMUnavailable(Exception):
    """API 키 미설정, 패키지 미설치, 호출 실패 등.

    permanent=True   : 키 없음/잘못된 키/모델 없음 등 -> 라우터가 해당 LLM 을 이번 실행 동안 비활성화
    retry_after=초   : 시간 초과/일시적 연결 끊김/요청 한도 등 -> 그 시간 동안만 쉬었다가 자동으로 다시 사용
    """

    def __init__(self, message: str = "", permanent: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.permanent = permanent
        self.retry_after = retry_after


class LLMClient(ABC):
    name = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        ...

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict[str, Any]:
        text = self.complete(system + "\n\n반드시 JSON 객체 하나만 출력하세요. 다른 설명은 쓰지 마세요.", user,
                             max_tokens)
        return extract_json(text)


def extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMUnavailable(f"JSON 파싱 실패: {e}") from e
    if not isinstance(data, dict):
        raise LLMUnavailable("JSON 객체가 아님")
    return data
