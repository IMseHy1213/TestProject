"""Claude (Anthropic) 클라이언트.

기본 모델: claude-sonnet-5  (Sonnet 4.6 을 쓰려면 ANTHROPIC_MODEL=claude-sonnet-4-6)
API 키: 환경변수 ANTHROPIC_API_KEY
"""

from __future__ import annotations

import logging

from .base import LLMClient, LLMUnavailable, ToolCall, ToolTurn

log = logging.getLogger(__name__)


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic 패키지가 없습니다 (pip install anthropic)") from e
        self._anthropic = anthropic
        import os

        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise LLMUnavailable("ANTHROPIC_API_KEY 가 없습니다 (my_api_key.py 또는 관리자 화면에서 Claude 키 입력)",
                                 permanent=True)
        try:
            self.client = anthropic.Anthropic(timeout=90.0, max_retries=1)
        except Exception as e:  # 자격 증명 없음
            raise LLMUnavailable(f"Anthropic 클라이언트 생성 실패: {e}") from e
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._create(system=system, max_tokens=max_tokens, messages=[{"role": "user", "content": user}])
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("모델이 요청을 거절했습니다")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text:
            raise LLMUnavailable("빈 응답")
        return text

    def tool_step(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 1500) -> ToolTurn:
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                               for t in tools]
        resp = self._create(system=system, max_tokens=max_tokens, messages=to_anthropic_messages(messages), **kwargs)
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("모델이 요청을 거절했습니다")
        return from_anthropic_response(resp)

    def _create(self, **kwargs):
        a = self._anthropic
        try:
            return self.client.messages.create(
                model=self.model,
                # 요구사항 추출/설명 생성은 단순 작업이라 effort 를 낮춰 비용·지연을 줄임
                output_config={"effort": "low"},
                **kwargs,
            )
        except a.AuthenticationError as e:
            raise LLMUnavailable("Anthropic API 키가 올바르지 않습니다", permanent=True) from e
        except a.NotFoundError as e:
            raise LLMUnavailable(f"모델 '{self.model}' 을 찾을 수 없습니다", permanent=True) from e
        except a.RateLimitError as e:
            raise LLMUnavailable("Anthropic 분당 요청 한도 초과", retry_after=30, kind="rate_limit") from e
        except a.APIStatusError as e:
            raise LLMUnavailable(f"Anthropic API 오류 {e.status_code}: {e.message}",
                                 retry_after=20 if e.status_code >= 500 else None) from e
        except a.APIConnectionError as e:
            raise LLMUnavailable("Anthropic 서버 연결 실패", retry_after=60, kind="connection") from e
        except TypeError as e:  # 자격 증명을 찾지 못한 경우 요청 시점에 발생
            raise LLMUnavailable(f"Anthropic 자격 증명 없음: {e}", permanent=True) from e


# ---------------------------------------------------------------------- 공통 대화 형식 <-> Messages API
def _block(b) -> dict:
    if isinstance(b, dict):
        return b
    if hasattr(b, "model_dump"):
        return b.model_dump(exclude_none=True)
    return {k: v for k, v in vars(b).items() if not k.startswith("_")}


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            if out and out[-1]["role"] == "user":  # 도구 결과 바로 뒤의 새 질문 -> 같은 메시지에 이어 붙임
                prev = out[-1]["content"]
                blocks = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
                out[-1]["content"] = blocks + [{"type": "text", "text": m["content"]}]
            else:
                out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            raw = (m.get("raw") or {}).get("anthropic")
            if raw:  # Claude 가 만든 응답이면 그대로
                out.append({"role": "assistant", "content": raw})
                continue
            content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                        for c in m.get("tool_calls") or []]
            if content:
                out.append({"role": "assistant", "content": content})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["call_id"], "content": m["content"]}
            if m.get("is_error"):
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)  # 연속된 도구 결과는 한 메시지로
            else:
                out.append({"role": "user", "content": [block]})
    return out


def from_anthropic_response(resp) -> ToolTurn:
    calls, texts = [], []
    for b in resp.content:
        if b.type == "tool_use":
            calls.append(ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {})))
        elif b.type == "text":
            texts.append(b.text)
    if not calls and not "".join(texts).strip():
        raise LLMUnavailable("빈 응답")
    return ToolTurn(text="".join(texts), tool_calls=calls, raw={"anthropic": [_block(b) for b in resp.content]},
                    provider="anthropic")
