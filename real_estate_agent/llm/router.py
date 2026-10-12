"""여러 LLM 을 우선순위대로 시도하는 라우터.

LLM_PROVIDERS=anthropic,openai,rule  ->  Claude 실패 시 GPT, 그것도 실패하면 규칙 기반.
'rule' 은 LLM 없이 동작하는 모드로, 이 경우 호출부가 규칙 기반 로직을 사용합니다.
"""

from __future__ import annotations

import logging
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

    @property
    def available(self) -> bool:
        return bool(self.clients)

    def status(self) -> str:
        active = ", ".join(f"{c.name}({getattr(c, 'model', '')})" for c in self.clients) or "없음"
        off = "; ".join(f"{k}: {v}" for k, v in self.unavailable.items())
        return f"사용 가능 LLM: {active}" + (f" | 비활성: {off}" if off else "") + " | 최종 대체: 규칙 기반"

    def _try(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        for c in list(self.clients):
            try:
                result = getattr(c, fn_name)(*args, **kwargs)
                self.last_used = c.name
                return result
            except LLMUnavailable as e:
                log.warning("%s 실패: %s -> 다음 LLM 시도", c.name, e)
                last_err = e
                if e.permanent:
                    self.clients.remove(c)
                    self.unavailable[c.name] = str(e)
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
