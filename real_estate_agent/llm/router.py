"""여러 LLM 을 우선순위대로 시도하는 라우터.

LLM_PROVIDERS=anthropic,openai,rule  ->  Claude 실패 시 GPT, 그것도 실패하면 규칙 기반.
'rule' 은 LLM 없이 동작하는 모드로, 이 경우 호출부가 규칙 기반 로직을 사용합니다.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .base import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, providers: list[str], anthropic_model: str = "claude-sonnet-5",
                 openai_model: str = "gpt-5.6-luna"):
        self.clients: list[LLMClient] = []
        self.unavailable: dict[str, str] = {}
        factories: dict[str, Callable[[], LLMClient]] = {
            "anthropic": lambda: _make_anthropic(anthropic_model),
            "claude": lambda: _make_anthropic(anthropic_model),
            "openai": lambda: _make_openai(openai_model),
            "gpt": lambda: _make_openai(openai_model),
        }
        for p in providers:
            p = p.lower()
            if p == "rule":
                break
            if p not in factories:
                self.unavailable[p] = "지원하지 않는 provider"
                continue
            try:
                self.clients.append(factories[p]())
            except LLMUnavailable as e:
                self.unavailable[p] = str(e)
        self.last_used: str = "rule"
        self.cooldown_until: dict[str, float] = {}  # 일시적 오류로 잠깐 쉬는 LLM -> 다시 쓸 시각
        self.last_error: dict[str, str] = {}

    @property
    def available(self) -> bool:
        """설정된 LLM 이 있는지 (잠깐 쉬는 중인 것 포함). 키 오류 등으로 완전히 빠진 것은 제외."""
        return bool(self.clients)

    def _cooling(self, c: LLMClient) -> float:
        """잠깐 쉬는 중이면 남은 초, 아니면 0."""
        return max(0.0, self.cooldown_until.get(c.name, 0.0) - time.time())

    @property
    def ready(self) -> bool:
        """지금 바로 쓸 수 있는 LLM 이 있는지."""
        return any(self._cooling(c) == 0 for c in self.clients)

    def mode(self) -> str:
        """화면 표시용: 'ai' (사용 중) / 'cooling' (일시적 오류로 잠시 규칙 기반) / 'off' (미연결)"""
        if not self.clients:
            return "off"
        return "ai" if self.ready else "cooling"

    def seconds_until_retry(self) -> int:
        waits = [self._cooling(c) for c in self.clients]
        return int(min(waits)) + 1 if waits else 0

    def status(self) -> str:
        parts = []
        for c in self.clients:
            label = f"{c.name}({getattr(c, 'model', '')})"
            wait = self._cooling(c)
            if wait:
                label += f" - 일시 오류로 {int(wait) + 1}초 후 재시도 ({self.last_error.get(c.name, '')})"
            parts.append(label)
        active = ", ".join(parts) or "없음"
        off = "; ".join(f"{k}: {v}" for k, v in self.unavailable.items())
        return f"사용 가능 LLM: {active}" + (f" | 비활성: {off}" if off else "") + " | 최종 대체: 규칙 기반"

    def _try(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        for c in list(self.clients):
            if self._cooling(c):  # 일시적 오류 후 쉬는 중 -> 기다리지 않고 다음으로
                continue
            try:
                result = getattr(c, fn_name)(*args, **kwargs)
                self.last_used = c.name
                self.cooldown_until.pop(c.name, None)
                return result
            except LLMUnavailable as e:
                log.warning("%s 실패: %s -> 다음 LLM 시도", c.name, e)
                last_err = e
                self.last_error[c.name] = str(e)
                if e.permanent:  # 키 오류 등: 이번 실행 동안 사용 안 함
                    self.clients.remove(c)
                    self.unavailable[c.name] = str(e)
                elif e.retry_after:  # 시간 초과 등: 잠깐만 쉬고 자동으로 다시 사용
                    self.cooldown_until[c.name] = time.time() + e.retry_after
        self.last_used = "rule"
        raise LLMUnavailable(str(last_err) if last_err else "사용 가능한 LLM 없음")

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        return self._try("complete", system, user, max_tokens)

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict[str, Any]:
        return self._try("complete_json", system, user, max_tokens)


def _make_anthropic(model: str) -> LLMClient:
    from .anthropic_client import AnthropicLLM

    return AnthropicLLM(model)


def _make_openai(model: str) -> LLMClient:
    from .openai_client import OpenAILLM

    return OpenAILLM(model)
