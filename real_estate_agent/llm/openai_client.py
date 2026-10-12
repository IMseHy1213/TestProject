"""OpenAI (GPT) 클라이언트.

모델 ID 는 OPENAI_MODEL 환경변수로 지정합니다. (예: GPT-5.6 Luna 의 실제 모델 ID)
API 키: 환경변수 OPENAI_API_KEY
"""

from __future__ import annotations

from .base import LLMClient, LLMUnavailable


class OpenAILLM(LLMClient):
    name = "openai"

    def __init__(self, model: str):
        try:
            import openai
        except ImportError as e:
            raise LLMUnavailable("openai 패키지가 없습니다 (pip install openai)") from e
        self._openai = openai
        try:
            self.client = openai.OpenAI()
        except Exception as e:  # OPENAI_API_KEY 없음
            raise LLMUnavailable(f"OpenAI 클라이언트 생성 실패: {e}") from e
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        try:
            resp = self.client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                max_output_tokens=max_tokens,
            )
        except (self._openai.AuthenticationError, self._openai.NotFoundError) as e:
            raise LLMUnavailable(f"OpenAI 인증/모델 오류 (OPENAI_MODEL 확인): {e}", permanent=True) from e
        except self._openai.OpenAIError as e:
            raise LLMUnavailable(f"OpenAI 호출 실패: {e}") from e
        text = resp.output_text
        if not text:
            raise LLMUnavailable("빈 응답")
        return text
