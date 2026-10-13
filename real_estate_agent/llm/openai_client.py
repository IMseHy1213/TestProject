"""OpenAI (GPT) 클라이언트.

API 키: 환경변수 OPENAI_API_KEY  (.env 파일에 넣으세요. 코드나 GitHub 에는 절대 넣지 마세요)
모델  : 환경변수 OPENAI_MODEL (기본 gpt-5.6-luna)

- 지정한 모델 이름이 계정에 없으면, 계정에서 쓸 수 있는 GPT 모델 목록을 조회해 가장 알맞은 것을
  자동으로 고르고 화면/로그에 알려줍니다. (`python main.py llm-check` 로 미리 확인 가능)
- GPT-5 계열은 답을 쓰기 전에 내부 추론에 토큰을 쓰므로, 출력 토큰 한도를 넉넉히 잡고
  추론 강도는 낮게(low) 요청합니다. 추론 옵션을 지원하지 않는 모델이면 옵션 없이 다시 요청합니다.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .base import LLMClient, LLMUnavailable, extract_json

log = logging.getLogger(__name__)

MIN_OUTPUT_TOKENS = 4000  # 추론 토큰 + 실제 답변


def pick_model(available: list[str], wanted: str) -> Optional[str]:
    """계정에서 쓸 수 있는 모델 중 wanted 와 가장 가까운 채팅용 GPT 모델을 고름."""
    chat = [m for m in available if m.startswith("gpt-")
            and not re.search(r"(audio|realtime|transcribe|tts|image|search|embedding|instruct)", m)]
    if not chat:
        return None
    if wanted in chat:
        return wanted
    wanted_l = wanted.lower()
    # 1) 'luna' 처럼 이름 일부가 같은 모델
    for token in re.split(r"[-_.\s]+", wanted_l):
        if token and not token.isdigit() and token not in {"gpt"}:
            hits = [m for m in chat if token in m.lower()]
            if hits:
                return sorted(hits, key=_version_key, reverse=True)[0]
    # 2) 가장 최신 버전 (날짜가 붙은 스냅샷·mini/nano 보다 기본 모델 우선)
    base = [m for m in chat if not re.search(r"(mini|nano|\d{4}-\d{2}-\d{2})", m)] or chat
    return sorted(base, key=_version_key, reverse=True)[0]


def _version_key(model: str) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", model.split("-20")[0])][:3]
    return tuple(nums + [0] * (3 - len(nums))) + (-len(model),)


class OpenAILLM(LLMClient):
    name = "openai"

    def __init__(self, model: str, client: Any = None):
        try:
            import openai
        except ImportError as e:
            raise LLMUnavailable("openai 패키지가 없습니다 (pip install openai)") from e
        self._openai = openai
        if client is not None:
            self.client = client
        else:
            try:
                self.client = openai.OpenAI(timeout=90.0, max_retries=1)
            except Exception as e:  # OPENAI_API_KEY 없음
                raise LLMUnavailable("OPENAI_API_KEY 가 설정되지 않았습니다 (.env 확인)", permanent=True) from e
        self.model = model
        self.requested_model = model
        self._use_reasoning = True
        self._model_resolved = False

    # ------------------------------------------------------------------ 공개 API
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        return self._call(system, user, max_tokens, json_mode=False)

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict[str, Any]:
        text = self._call(system + "\n\n반드시 JSON 객체 하나만 출력하세요.", user, max_tokens, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return extract_json(text)

    def available_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())

    # ------------------------------------------------------------------ 내부
    def _call(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str:
        o = self._openai
        params: dict[str, Any] = {
            "instructions": system,
            "input": user,
            "max_output_tokens": max(max_tokens, MIN_OUTPUT_TOKENS),
        }
        if json_mode:
            params["text"] = {"format": {"type": "json_object"}}
        for _attempt in range(3):
            if self._use_reasoning:
                params["reasoning"] = {"effort": "low"}
            else:
                params.pop("reasoning", None)
            try:
                resp = self.client.responses.create(model=self.model, **params)
                break
            except o.NotFoundError as e:
                if not self._resolve_model():
                    raise LLMUnavailable(f"OpenAI 모델 '{self.model}' 을 찾을 수 없습니다 (OPENAI_MODEL 확인)",
                                         permanent=True) from e
            except o.BadRequestError as e:
                msg = str(e).lower()
                if self._use_reasoning and "reasoning" in msg:
                    self._use_reasoning = False  # 추론 옵션을 지원하지 않는 모델 (예: gpt-4.1)
                    continue
                if json_mode and ("json" in msg or "format" in msg):
                    json_mode = False
                    params.pop("text", None)
                    continue
                raise LLMUnavailable(f"OpenAI 요청 오류: {e}") from e
            except o.AuthenticationError as e:
                raise LLMUnavailable("OpenAI API 키가 올바르지 않거나 만료되었습니다", permanent=True) from e
            except o.PermissionDeniedError as e:
                raise LLMUnavailable(f"OpenAI 권한 없음 (모델 접근 권한/결제 확인): {e}", permanent=True) from e
            except o.RateLimitError as e:
                raise LLMUnavailable("OpenAI 요청 한도 초과 또는 크레딧 부족 (결제 설정 확인)", retry_after=30) from e
            except o.APITimeoutError as e:  # 응답이 늦음 (APIConnectionError 의 한 종류라 먼저 처리)
                raise LLMUnavailable("OpenAI 응답 시간 초과 - 잠시 후 다시 시도합니다", retry_after=20) from e
            except o.APIConnectionError as e:
                # 인터넷이 잠깐 끊긴 경우: 매번 기다리지 않도록 1분만 쉬고, 그 뒤 자동으로 다시 시도
                raise LLMUnavailable("OpenAI 서버에 연결할 수 없습니다 (인터넷/방화벽 확인)", retry_after=60) from e
            except o.InternalServerError as e:
                raise LLMUnavailable(f"OpenAI 서버 오류 - 잠시 후 다시 시도합니다: {e}", retry_after=20) from e
            except o.OpenAIError as e:
                raise LLMUnavailable(f"OpenAI 호출 실패: {e}") from e
        else:
            raise LLMUnavailable("OpenAI 요청 재시도 실패")

        text = getattr(resp, "output_text", "") or ""
        if not text.strip():
            status = getattr(resp, "status", "")
            raise LLMUnavailable(f"OpenAI 빈 응답 (status={status})")
        return text

    def _resolve_model(self) -> bool:
        """지정 모델이 없을 때 한 번만, 계정에서 쓸 수 있는 모델로 교체."""
        if self._model_resolved:
            return False
        self._model_resolved = True
        try:
            choice = pick_model(self.available_models(), self.requested_model)
        except Exception as e:
            log.warning("OpenAI 모델 목록 조회 실패: %s", e)
            return False
        if not choice or choice == self.model:
            return False
        log.warning("OpenAI 모델 '%s' 이 없어 '%s' 로 대신 사용합니다. (.env 의 OPENAI_MODEL 을 바꾸면 고정됩니다)",
                    self.requested_model, choice)
        self.model = choice
        return True
