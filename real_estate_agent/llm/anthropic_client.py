"""Claude (Anthropic) 클라이언트.

기본 모델: claude-sonnet-5  (Sonnet 4.6 을 쓰려면 ANTHROPIC_MODEL=claude-sonnet-4-6)
API 키: 환경변수 ANTHROPIC_API_KEY
"""

from __future__ import annotations

import logging

from .base import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic 패키지가 없습니다 (pip install anthropic)") from e
        self._anthropic = anthropic
        try:
            self.client = anthropic.Anthropic()
        except Exception as e:  # 자격 증명 없음
            raise LLMUnavailable(f"Anthropic 클라이언트 생성 실패: {e}") from e
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                # 요구사항 추출/설명 생성은 단순 작업이라 effort 를 낮춰 비용·지연을 줄임
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": user}],
            )
        except a.AuthenticationError as e:
            raise LLMUnavailable("Anthropic API 키가 올바르지 않습니다", permanent=True) from e
        except a.NotFoundError as e:
            raise LLMUnavailable(f"모델 '{self.model}' 을 찾을 수 없습니다", permanent=True) from e
        except a.RateLimitError as e:
            raise LLMUnavailable("Anthropic 요청 한도 초과") from e
        except a.APIStatusError as e:
            raise LLMUnavailable(f"Anthropic API 오류 {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMUnavailable("Anthropic 서버 연결 실패") from e
        except TypeError as e:  # 자격 증명을 찾지 못한 경우 요청 시점에 발생
            raise LLMUnavailable(f"Anthropic 자격 증명 없음: {e}", permanent=True) from e

        if resp.stop_reason == "refusal":
            raise LLMUnavailable("모델이 요청을 거절했습니다")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text:
            raise LLMUnavailable("빈 응답")
        return text
