"""OpenAI 연동 테스트 (실제 API 를 부르지 않고 가짜 클라이언트로 검사)."""

from types import SimpleNamespace

import httpx2 as httpx  # openai SDK 가 쓰는 HTTP 라이브러리
import openai
import pytest

from real_estate_agent.llm.base import LLMUnavailable
from real_estate_agent.llm.openai_client import OpenAILLM, pick_model
from real_estate_agent.setup_tools import update_env_file

REQ = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _err(cls, status, msg):
    return cls(msg, response=httpx.Response(status, request=REQ), body=None)


class FakeResponses:
    def __init__(self, script):
        self.script = list(script)  # 순서대로 돌려줄 응답 또는 예외
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(output_text=item, status="completed")


class FakeClient:
    def __init__(self, script, models=("gpt-4.1", "gpt-5", "gpt-5.1", "gpt-5.1-mini", "gpt-5-2025-08-07")):
        self.responses = FakeResponses(script)
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in models])


def test_pick_model():
    models = ["gpt-4.1", "gpt-5", "gpt-5.1", "gpt-5.1-mini", "gpt-5-2025-08-07", "gpt-4o-audio-preview", "o3"]
    assert pick_model(models, "gpt-5.1") == "gpt-5.1"
    assert pick_model(models, "gpt-5.6-luna") == "gpt-5.1"  # 없으면 최신 기본 모델
    assert pick_model(models + ["gpt-5.6-luna-2026-09-01"], "gpt-5.6-luna") == "gpt-5.6-luna-2026-09-01"
    assert pick_model(["o3", "dall-e-3"], "gpt-5") is None


def test_json_request_and_parameters():
    fake = FakeClient(['{"regions": ["음성군"], "budget_max": 29999}'])
    llm = OpenAILLM("gpt-5.1", client=fake)
    assert llm.complete_json("sys", "음성군 2억원대") == {"regions": ["음성군"], "budget_max": 29999}
    call = fake.responses.calls[0]
    assert call["model"] == "gpt-5.1"
    assert call["text"] == {"format": {"type": "json_object"}}
    assert call["reasoning"] == {"effort": "low"}
    assert call["max_output_tokens"] >= 4000  # 추론 토큰 여유


def test_unknown_model_is_replaced():
    fake = FakeClient([_err(openai.NotFoundError, 404, "model not found"), "안녕하세요"])
    llm = OpenAILLM("gpt-5.6-luna", client=fake)
    assert llm.complete("sys", "hi") == "안녕하세요"
    assert llm.model == "gpt-5.1"
    assert [c["model"] for c in fake.responses.calls] == ["gpt-5.6-luna", "gpt-5.1"]


def test_retry_without_reasoning_for_older_models():
    fake = FakeClient([_err(openai.BadRequestError, 400, "Unsupported parameter: 'reasoning'"), "ok"])
    llm = OpenAILLM("gpt-4.1", client=fake)
    assert llm.complete("sys", "hi") == "ok"
    assert "reasoning" not in fake.responses.calls[1]


def test_errors_become_llm_unavailable():
    llm = OpenAILLM("gpt-5", client=FakeClient([_err(openai.AuthenticationError, 401, "bad key")]))
    with pytest.raises(LLMUnavailable) as e:
        llm.complete("s", "u")
    assert e.value.permanent  # 키 오류는 이번 실행 동안 GPT 비활성화
    llm = OpenAILLM("gpt-5", client=FakeClient([""]))
    with pytest.raises(LLMUnavailable):
        llm.complete("s", "u")


def test_update_env_file_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# 주석\nREA_DATA_DIR=E:\\매물\nOPENAI_API_KEY=\n", encoding="utf-8")
    update_env_file({"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-5"}, env)
    text = env.read_text(encoding="utf-8")
    assert "# 주석" in text and "REA_DATA_DIR=E:\\매물" in text
    assert "OPENAI_API_KEY=sk-test" in text and "OPENAI_MODEL=gpt-5" in text


# ---------------- 일시적 오류 후 자동 복구 ----------------
from real_estate_agent.llm.base import LLMClient  # noqa: E402
from real_estate_agent.llm.router import LLMRouter  # noqa: E402


def test_timeout_is_temporary_not_permanent():
    llm = OpenAILLM("gpt-5", client=FakeClient([openai.APITimeoutError(request=REQ)]))
    with pytest.raises(LLMUnavailable) as e:
        llm.complete("s", "u")
    assert not e.value.permanent and e.value.retry_after  # 시간 초과는 잠깐만 쉼
    llm = OpenAILLM("gpt-5", client=FakeClient([openai.APIConnectionError(request=REQ)]))
    with pytest.raises(LLMUnavailable) as e:
        llm.complete("s", "u")
    assert not e.value.permanent and e.value.retry_after == 60


class Flaky(LLMClient):
    name = "openai"
    model = "gpt-test"

    def __init__(self, script):
        self.script = list(script)

    def complete(self, system, user, max_tokens=2000):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_router_recovers_after_cooldown(monkeypatch):
    import real_estate_agent.llm.router as router_mod

    now = [1000.0]
    monkeypatch.setattr(router_mod.time, "time", lambda: now[0])
    r = LLMRouter(["rule"])
    r.clients = [Flaky([LLMUnavailable("timeout", retry_after=20), "다시 연결됨"])]
    assert r.mode() == "ai"
    with pytest.raises(LLMUnavailable):
        r.complete("s", "u")
    assert r.available and r.mode() == "cooling" and r.seconds_until_retry() == 21  # 꺼지지 않고 잠깐 쉼
    with pytest.raises(LLMUnavailable):
        r.complete("s", "u")  # 쉬는 동안은 기다리지 않고 바로 규칙 기반으로
    now[0] += 21
    assert r.mode() == "ai" and r.complete("s", "u") == "다시 연결됨"  # 자동 복구


def test_router_permanent_error_disables():
    r = LLMRouter(["rule"])
    r.clients = [Flaky([LLMUnavailable("bad key", permanent=True)])]
    with pytest.raises(LLMUnavailable):
        r.complete("s", "u")
    assert not r.available and r.mode() == "off" and "openai" in r.unavailable
