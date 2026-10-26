from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMUnavailable(Exception):
    """API 키 미설정, 패키지 미설치, 호출 실패 등.

    permanent=True   : 키 없음/잘못된 키/모델 없음 등 -> 라우터가 해당 LLM 을 이번 실행 동안 비활성화
    retry_after=초   : 시간 초과/일시적 연결 끊김/요청 한도 등 -> 그 시간 동안만 쉬었다가 자동으로 다시 사용
    kind             : 화면 안내용 원인 ('rate_limit' 분당 한도, 'quota' 크레딧 부족, 'timeout', 'connection', 'server')
    """

    def __init__(self, message: str = "", permanent: bool = False, retry_after: float | None = None,
                 kind: str = ""):
        super().__init__(message)
        self.permanent = permanent
        self.retry_after = retry_after
        self.kind = kind


@dataclass
class ToolCall:
    """모델이 요청한 도구 호출 하나 (GPT/Claude 공통 형태)."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str = ""  # 인자 JSON 을 해석하지 못했을 때


@dataclass
class ToolTurn:
    """도구를 쓸 수 있는 대화에서 모델의 한 번 응답: 도구 호출들 또는 최종 답변(text)."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)  # 제공자별 원본 (같은 제공자에게 되돌려 보낼 때 사용)
    provider: str = ""


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    """도구 인자 (JSON 문자열 또는 dict) -> (dict, 오류 메시지)."""
    if isinstance(raw, dict):
        return raw, ""
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError) as e:
        return {}, f"인자 JSON 해석 실패: {e}"
    return (value, "") if isinstance(value, dict) else ({}, "인자는 JSON 객체여야 합니다")


class LLMClient(ABC):
    name = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        ...

    def tool_step(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 1500) -> ToolTurn:
        """도구를 쓸 수 있는 대화 한 단계.

        messages: 공통 형식
          {"role": "user", "content": str}
          {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": {...}}
          {"role": "tool", "call_id": str, "name": str, "content": str, "is_error": bool}
        tools: [{"name", "description", "parameters"(JSON schema)}] (비어 있으면 도구 없이 답만)
        """
        raise LLMUnavailable(f"{self.name}: 도구 호출(Function calling)을 지원하지 않습니다")

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
