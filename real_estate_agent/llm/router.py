"""여러 LLM(GPT, Claude)을 함께 쓰는 라우터.

LLM_STRATEGY=balance (기본) : GPT 와 Claude 를 골고루 번갈아 사용. 최근 1분 동안 덜 쓴 쪽에 먼저 보내고,
                              한쪽이 실패하거나 쉬는 중이면 다른 쪽이 바로 받음. (분당 한도가 두 배로 늘어나는 효과)
LLM_STRATEGY=order          : LLM_PROVIDERS 에 적은 순서대로 (앞의 것이 실패할 때만 다음 것).
둘 다 안 되면 규칙 기반. 'rule' 은 LLM 없이 동작하는 모드로, 이 경우 호출부가 규칙 기반 로직을 사용합니다.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
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
        self.last_kind: dict[str, str] = {}
        # 분당 사용 한도(429) 를 넘지 않도록: 꼭 필요하지 않은 호출(설명 문장, 취향 학습)은
        # 최근 1분 동안 이 횟수를 넘으면 AI 대신 규칙 기반으로 처리. 검색 문장 이해는 항상 AI 로 시도.
        try:
            self.optional_per_min = max(0, int(os.getenv("LLM_OPTIONAL_PER_MIN", "10")))
        except ValueError:
            self.optional_per_min = 10
        self._recent_calls: dict[str, deque[float]] = {}  # LLM 이름 -> 최근 호출 시각
        self.skipped_optional = 0
        self.strategy = os.getenv("LLM_STRATEGY", "balance").strip().lower() or "balance"
        self._turn = 0  # 사용량이 같을 때 번갈아 고르기 위한 순번
        self.usage: dict[str, int] = {}  # 이번 실행에서 LLM 별로 성공한 횟수 (관리자 화면 표시)

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

    def off_reason(self) -> str:
        """AI 를 못 쓰는 이유 (화면 안내용, 키 값은 포함하지 않음)."""
        if self.clients:
            return ""
        reasons = [v for k, v in self.unavailable.items() if k in {"openai", "gpt"}] or list(self.unavailable.values())
        return reasons[0] if reasons else "사용할 AI 가 설정되지 않았습니다 (LLM_PROVIDERS 확인)"

    def cooling_reason(self) -> str:
        """잠시 쉬는 이유 (화면 안내용)."""
        kinds = [self.last_kind.get(c.name, "") for c in self.clients if self._cooling(c)]
        if "quota" in kinds:
            return "OpenAI 크레딧(잔액)이 부족합니다 - 관리자가 결제 설정을 확인해야 합니다"
        if "rate_limit" in kinds:
            return "짧은 시간에 AI 요청이 몰려 OpenAI 분당 사용 한도에 걸렸습니다"
        if "timeout" in kinds:
            return "AI 응답이 늦어지고 있습니다"
        if "connection" in kinds:
            return "AI 서버에 연결할 수 없습니다 (인터넷 확인)"
        if "server" in kinds:
            return "AI 서버에 일시적인 오류가 있습니다"
        return "AI 응답이 지연되고 있습니다"

    def _recent(self, c: LLMClient) -> int:
        """최근 1분 동안 이 LLM 에 보낸 요청 수."""
        q = self._recent_calls.setdefault(c.name, deque())
        now = time.time()
        while q and now - q[0] > 60:
            q.popleft()
        return len(q)

    def _order(self) -> list[LLMClient]:
        """이번 요청을 보낼 순서. balance: 최근 1분 동안 덜 쓴 LLM 먼저, 같으면 번갈아."""
        clients = list(self.clients)
        if self.strategy != "balance" or len(clients) < 2:
            return clients
        self._turn += 1
        n = len(clients)
        return sorted(clients, key=lambda c: (self._recent(c), (clients.index(c) - self._turn) % n))

    def names(self) -> str:
        """화면 표시용: 지금 연결된 AI 이름 (예: 'GPT + Claude')."""
        label = {"openai": "GPT", "anthropic": "Claude"}
        return " + ".join(label.get(c.name, c.name) for c in self.clients) or "없음"

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
        mode = ("GPT·Claude 골고루 번갈아 사용" if self.strategy == "balance" and len(self.clients) > 1
                else "적힌 순서대로 사용")
        used = ", ".join(f"{k} {v}회" for k, v in self.usage.items())
        return (f"사용 가능 LLM: {active} ({mode})" + (f" | 이번 실행 사용 횟수: {used}" if used else "")
                + (f" | 비활성: {off}" if off else "") + " | 최종 대체: 규칙 기반")

    def _try(self, fn_name: str, *args: Any, optional: bool = False, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        candidates = self._order()
        if optional:  # 설명·학습 같은 부가 요청: LLM 마다 1분 예산 안에서만
            candidates = [c for c in candidates if self._recent(c) < self.optional_per_min]
            if self.clients and not candidates:
                self.skipped_optional += 1
                self.last_used = "rule"
                raise LLMUnavailable("최근 1분 동안 AI 요청이 많아 이번 설명/학습은 규칙 기반으로 처리")
        for c in candidates:
            if c not in self.clients or self._cooling(c):  # 일시적 오류 후 쉬는 중 -> 기다리지 않고 다음으로
                continue
            self._recent_calls.setdefault(c.name, deque()).append(time.time())
            try:
                result = getattr(c, fn_name)(*args, **kwargs)
                self.last_used = c.name
                self.usage[c.name] = self.usage.get(c.name, 0) + 1
                self.cooldown_until.pop(c.name, None)
                return result
            except LLMUnavailable as e:
                log.warning("%s 실패: %s -> 다음 LLM 시도", c.name, e)
                last_err = e
                self.last_error[c.name] = str(e)
                self.last_kind[c.name] = getattr(e, "kind", "")
                if e.permanent:  # 키 오류 등: 이번 실행 동안 사용 안 함
                    self.clients.remove(c)
                    self.unavailable[c.name] = str(e)
                elif e.retry_after:  # 시간 초과 등: 잠깐만 쉬고 자동으로 다시 사용
                    self.cooldown_until[c.name] = time.time() + e.retry_after
        self.last_used = "rule"
        raise LLMUnavailable(str(last_err) if last_err else "사용 가능한 LLM 없음")

    def complete(self, system: str, user: str, max_tokens: int = 2000, optional: bool = False) -> str:
        """optional=True: 없어도 되는 호출(설명 문장 등) - 분당 예산을 넘으면 바로 규칙 기반으로."""
        return self._try("complete", system, user, max_tokens, optional=optional)

    def complete_json(self, system: str, user: str, max_tokens: int = 2000,
                      optional: bool = False) -> dict[str, Any]:
        return self._try("complete_json", system, user, max_tokens, optional=optional)

    def tool_step(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 1500):
        """도구 호출 대화 한 단계. 대화 기록이 공통 형식이라 중간에 GPT -> Claude 로 바뀌어도 이어서 진행."""
        turn = self._try("tool_step", system, messages, tools, max_tokens)
        turn.provider = self.last_used
        return turn


def _make_anthropic(model: str) -> LLMClient:
    from .anthropic_client import AnthropicLLM

    return AnthropicLLM(model)


def _make_openai(model: str) -> LLMClient:
    from .openai_client import OpenAILLM

    return OpenAILLM(model)
