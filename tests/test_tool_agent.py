"""v0.2 Tool-use Agent 테스트 (가짜 GPT/Claude 로 Function calling 흐름 검사)."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.llm import LLMClient, LLMRouter, LLMUnavailable, ToolCall, ToolTurn
from real_estate_agent.llm.anthropic_client import from_anthropic_response, to_anthropic_messages
from real_estate_agent.llm.openai_client import from_openai_response, to_openai_input
from real_estate_agent.tool_agent import AgentContext, ToolAgent, ToolRegistry, validate_arguments

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def agent(tmp_path):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / "t.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


class ScriptedLLM(LLMClient):
    """정해진 순서대로 도구 호출/답변을 돌려주는 가짜 모델. 받은 대화 기록도 저장."""
    name = "openai"
    model = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def complete(self, system, user, max_tokens=2000):
        raise LLMUnavailable("not used")

    def tool_step(self, system, messages, tools, max_tokens=1500):
        self.seen.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item


def router(*clients):
    r = LLMRouter(["rule"])
    r.clients = list(clients)
    return r


def call(name, **args):
    return ToolTurn(tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args)])


def test_validate_arguments():
    schema = {"type": "object", "required": ["listing_id"], "properties": {
        "listing_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "purpose": {"type": "string", "enum": ["창고", "공장"]}, "ids": {"type": "array", "items": {"type": "string"}}}}
    assert validate_arguments(schema, {"listing_id": "A", "limit": 3.0, "ids": "a, b"}) == (
        {"listing_id": "A", "limit": 3, "ids": ["a", "b"]}, [])
    _, errors = validate_arguments(schema, {"limit": 50, "purpose": "카페", "color": "red"})
    text = " ".join(errors)
    assert "listing_id" in text and "10 이하" in text and "카페" in text and "모르는 인자 'color'" in text


def test_tools_never_expose_contacts_or_full_lot_number(agent):
    reg = ToolRegistry(agent)
    obs = reg.run("search_listings", {"query": "양평 땅", "limit": 10})
    assert obs["ok"] and obs["result"]["매물"]
    text = json.dumps(obs, ensure_ascii=False)
    assert "010-" not in text and "***" in text  # 연락처 없음, 지번 가림
    detail = reg.run("get_listing", {"listing_id": obs["result"]["매물"][0]["id"]})
    assert detail["ok"] and "010-" not in json.dumps(detail, ensure_ascii=False)


def test_tool_errors_become_observations(agent):
    reg = ToolRegistry(agent)
    assert "없습니다" in reg.run("nope", {})["error"]
    assert "찾을 수 없습니다" in reg.run("get_listing", {"listing_id": "X-없음"})["error"]
    assert "인자 오류" in reg.run("check_land_use", {"listing_id": "T-1007", "purpose": "카페"})["error"]
    assert "2개 이상" in reg.run("compare_listings", {"listing_ids": ["T-1007", "T-1007"]})["error"]


def test_react_loop_with_llm_recovers_from_tool_error(agent):
    fake = ScriptedLLM([
        call("check_land_use", listing_id="T-1007", purpose="카페"),         # 잘못된 인자 -> 오류 관찰
        call("check_land_use", listing_id="T-1007", purpose="전원주택"),      # 고쳐서 다시
        ToolTurn(text="T-1007 매물(1억 9,000만원)은 전원주택 건축이 가능해 보입니다."),
    ])
    ta = ToolAgent(agent, llm=router(fake))
    result, history = ta.run("T-1007 에 전원주택 지을 수 있어?")
    tools = [(s.tool, s.ok) for s in result.steps if s.kind == "tool"]
    assert tools == [("check_land_use", False), ("check_land_use", True)]
    assert result.stopped == "answer" and result.policy == "openai" and "T-1007" in result.listing_ids
    # 두 번째 요청 때 모델은 첫 도구 오류를 관찰로 받았어야 함
    second = fake.seen[1]
    assert second[-1]["role"] == "tool" and second[-1]["is_error"] and "인자 오류" in second[-1]["content"]
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


def test_failover_to_rule_policy_mid_run(agent):
    fake = ScriptedLLM([call("search_listings", query="음성군 창고 땅", limit=3),
                        LLMUnavailable("한도", retry_after=30, kind="rate_limit")])
    result, _ = ToolAgent(agent, llm=router(fake)).run("음성군 창고 땅")
    kinds = [s.kind for s in result.steps]
    assert "fallback" in kinds and result.policy == "rule" and result.stopped == "answer"
    assert re.search(r"매물 \d+건", result.answer)  # 규칙 기반이 이미 받은 검색 결과로 답을 마무리


def test_repeated_calls_and_step_limit(agent):
    same = lambda _m: call("list_regions")  # noqa: E731 - 계속 같은 호출만 하는 모델
    fake = ScriptedLLM([same] * 4 + [ToolTurn(text="정리된 답")])
    result, _ = ToolAgent(agent, llm=router(fake), max_steps=4).run("지역 알려줘")
    errors = [s.summary for s in result.steps if s.kind == "tool" and not s.ok]
    assert any("여러 번" in e for e in errors)  # 반복 호출 차단
    assert result.stopped == "max_steps" and result.answer == "정리된 답"


def test_feedback_tool_excludes_rejected_in_same_conversation(agent):
    ctx = AgentContext()
    reg = ToolRegistry(agent, ctx)
    first = reg.run("search_listings", {"query": "양평 땅", "limit": 1})["result"]["매물"][0]["id"]
    assert reg.run("record_feedback", {"listing_id": first, "action": "reject", "reason": "너무 비싸요"})["ok"]
    again = reg.run("search_listings", {"query": "양평 땅", "limit": 10})["result"]
    assert first not in [x["id"] for x in again["매물"]] and first in again["제외한_매물"]


def test_rule_policy_compare_flow_and_run_log(agent, tmp_path):
    result, _ = ToolAgent(agent).run("양평 땅 2개 비교해줘")
    assert [s.tool for s in result.steps if s.kind == "tool"] == ["search_listings", "compare_listings"]
    assert "가장 싸고 넓은 매물" in result.answer or "최저가" in result.answer and len(result.listing_ids) == 2
    row = json.loads((tmp_path / "logs" / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["tool_calls"] == 2 and row["policy"] == "rule" and row["total_ms"] >= 0


def test_no_result_answer(agent):
    result, _ = ToolAgent(agent).run("부산 해운대 땅")
    assert result.answer == "해당 조건에 맞는 매물이 없습니다."


# ---------------------------------------------------------------- 제공자별 형식 변환
def history_sample():
    return [{"role": "user", "content": "양평 땅"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("call_1", "search_listings", {"query": "양평"})]},
            {"role": "tool", "call_id": "call_1", "name": "search_listings", "content": '{"ok": true}'},
            {"role": "tool", "call_id": "call_2", "name": "list_regions", "content": '{"ok": false}', "is_error": True},
            {"role": "user", "content": "더 싼 거"}]


def test_openai_conversion():
    items = to_openai_input(history_sample())
    assert items[1] == {"type": "function_call", "call_id": "call_1", "name": "search_listings",
                        "arguments": '{"query": "양평"}'}
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'}
    resp = SimpleNamespace(output=[SimpleNamespace(type="reasoning", id="rs_1", summary=[]),
                                   SimpleNamespace(type="function_call", call_id="call_9", name="get_listing",
                                                   arguments='{"listing_id": "T-1"}'),
                                   SimpleNamespace(type="function_call", call_id="call_10", name="x", arguments="{bad")],
                           output_text="", status="completed")
    turn = from_openai_response(resp)
    assert [c.name for c in turn.tool_calls] == ["get_listing", "x"]
    assert turn.tool_calls[0].arguments == {"listing_id": "T-1"} and turn.tool_calls[1].parse_error
    assert len(turn.raw["openai"]) == 3  # 추론 항목까지 그대로 보관 -> 다음 요청에 되돌려 보냄
    again = to_openai_input([{"role": "assistant", "content": "", "tool_calls": turn.tool_calls, "raw": turn.raw}])
    assert again[0]["type"] == "reasoning"


def test_anthropic_conversion():
    msgs = to_anthropic_messages(history_sample())
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "search_listings",
                                     "input": {"query": "양평"}}
    results = msgs[2]["content"]  # 도구 결과 2개 + 새 질문이 한 메시지로
    assert [b["type"] for b in results] == ["tool_result", "tool_result", "text"] and results[1]["is_error"]
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="찾아볼게요", model_dump=lambda **k: {"type": "text"}),
                                    SimpleNamespace(type="tool_use", id="toolu_1", name="search_listings",
                                                    input={"query": "음성"}, model_dump=lambda **k: {"type": "tool_use"})],
                           stop_reason="tool_use")
    turn = from_anthropic_response(resp)
    assert turn.text == "찾아볼게요" and turn.tool_calls[0].arguments == {"query": "음성"}


def test_openai_client_sends_tools(monkeypatch):
    from real_estate_agent.llm.openai_client import OpenAILLM

    calls = []

    class Responses:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(output=[], output_text="답", status="completed")

    llm = OpenAILLM("gpt-5.1", client=SimpleNamespace(responses=Responses()))
    turn = llm.tool_step("sys", [{"role": "user", "content": "q"}],
                         [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}])
    assert turn.text == "답" and calls[0]["tools"][0] == {"type": "function", "name": "t", "description": "d",
                                                         "parameters": {"type": "object", "properties": {}}}


def test_step_limit_falls_back_to_rule_summary_if_model_keeps_calling_tools(agent):
    keep = lambda _m: call("search_listings", query="양평 땅", limit=2)  # noqa: E731
    result, _ = ToolAgent(agent, llm=router(ScriptedLLM([keep] * 5)), max_steps=3).run("양평 땅")
    assert result.stopped == "max_steps" and result.policy == "rule" and re.search(r"매물 \d+건", result.answer)


def test_rule_policy_sorts_by_price_when_asked_for_cheap(agent):
    result, _ = ToolAgent(agent).run("음성군 땅 중 제일 싼 거 3개")
    search = next(s for s in result.steps if s.tool == "search_listings")
    assert search.args.get("sort") == "price_asc" and "가격 낮은 순" in result.answer
    obs = ToolRegistry(agent).run("search_listings", {"query": "음성군 땅", "sort": "price_asc", "limit": 5})
    prices = [x["가격_만원"] for x in obs["result"]["매물"]]
    assert prices == sorted(prices)


def test_public_address_shows_only_province_and_city():
    from real_estate_agent.agent import mask_address, public_office_no

    assert mask_address("음성군 금왕읍 각회리 210-3") == "충청북도 음성군 ***"          # 시/도 채움
    assert mask_address("경기도 양평군 서종면 수입리 210") == "경기도 양평군 ***"
    assert mask_address("충북 음성군 생극면 차곡리 7 외 1필지") == "충청북도 음성군 ***"
    assert mask_address("서울특별시 강남구 테헤란로 152 강남파이낸스센터 21층 2101호") == "서울특별시 강남구 ***"
    assert mask_address("충청북도 청주시 흥덕구 오송읍 오송생명로 123") == "충청북도 청주시 흥덕구 ***"
    assert mask_address("감곡면 문촌리 417-2") == "***"
    assert public_office_no("S.24250409(이재순님)") == "S.24250409" and public_office_no("T-1004") == "T-1004"


def test_cards_and_badges_from_tool_results(agent):
    result, _ = ToolAgent(agent).run("음성군 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    assert len(result.listing_ids) == 2
    badges = [b for i in result.listing_ids for b in result.badges.get(i, [])]
    assert "최저가" in badges and "가장 넓음" in badges and any(b.startswith("창고 ") for b in badges)
    assert "\n1)" not in result.answer and "●" not in result.answer  # 목록·특징을 답변에 늘어놓지 않음
    assert len(result.answer.splitlines()) <= 3  # 짧게 (3줄 이내)


def test_names_in_office_numbers_never_reach_the_ai(tmp_path):
    """매물번호 'S.24250409(이재순님)' 의 이름이 AI 에게 가는 도구 결과·실행 과정에 나오지 않음."""
    from openpyxl import Workbook

    data = tmp_path / "매물"
    data.mkdir()
    wb = Workbook()
    ws = wb.active
    ws.append(["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "특이사항"])
    ws.append(["S.24250409(이재순님)", "충청북도 음성군 금왕읍 각회리 123", "대", "계획관리지역", 84, "1억 3900만", "창고 가능"])
    ws.append(["S.251229(골드)", "충청북도 음성군 생극면 차곡리 7", "임야", "계획관리지역", 605, "3억", "2차선 도로"])
    wb.save(data / "장부.xlsx")
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp_path / "n.db", log_dir=tmp_path / "l", llm_providers=["rule"]))
    a.sync()
    fake = ScriptedLLM([call("search_listings", query="음성군 창고 땅", limit=5),
                        lambda m: call("compare_listings", listing_ids=[x["id"] for x in json.loads(m[-1]["content"])["result"]["매물"]]),
                        call("check_land_use", listing_id="S.24250409", purpose="창고"),
                        ToolTurn(text="S.24250409(1억 3,900만원)가 가장 저렴합니다.")])
    result, history = ToolAgent(a, llm=router(fake)).run("음성군 창고 땅 비교")
    sent = json.dumps([m.get("content") for m in history], ensure_ascii=False)
    trace = json.dumps([vars(s) for s in result.steps], ensure_ascii=False)
    assert "이재순" not in sent and "골드" not in sent and "이재순" not in trace
    assert result.listing_ids == ["S.24250409(이재순님)", "S.251229(골드)"]  # 화면 카드용 실제 id
    assert "최저가" in result.badges["S.24250409(이재순님)"] and "창고 가능" in result.badges["S.24250409(이재순님)"]


def test_follow_up_picks_only_from_listings_just_shown(agent):
    ctx = AgentContext()
    runner = ToolAgent(agent, ctx=ctx)
    first, history = runner.run("음성군에 창고 지을 수 있는 땅 중 싼 거 2개 비교해줘")
    shown = first.listing_ids
    assert len(shown) == 2
    second, history = runner.run("그 중에 넓은 건?", history)
    search = next(s for s in second.steps if s.tool == "search_listings")
    assert search.args == {"use_previous_conditions": True, "sort": "area_desc", "limit": 1}
    widest_shown = max(shown, key=lambda i: agent.find_listing(i).area_m2)
    assert second.listing_ids == [widest_shown]  # 보여준 2건 안에서만
    assert second.answer.startswith("앞서 보여드린 2건 중 가장 넓은 매물은") and len(second.answer.splitlines()) == 1
    # 같은 2건 안에서 계속 고름
    third, _ = runner.run("그 중에 싼 건?", history)
    cheapest_shown = min(shown, key=lambda i: agent.find_listing(i).price_manwon)
    assert third.listing_ids == [cheapest_shown]
    # 음성군 창고 매물 전체로 넓히지 않음 (전체 중 가장 넓은 매물이 보여준 2건 밖에 있어도)
    all_ok = ToolRegistry(agent).run("search_listings", {"query": "음성군 창고 땅", "limit": 10})["result"]["매물"]
    assert len(all_ok) > 2


def test_listing_without_office_number_gets_readable_name(tmp_path):
    from openpyxl import Workbook

    data = tmp_path / "매물" / "음성군" / "음성군 감곡면 오향리 1 외 36필지(물류부지)"
    data.mkdir(parents=True)
    wb = Workbook()
    ws = wb.active
    for r in [["소재지", "충청북도 음성군 감곡면 오향리 1"], ["지목", "전"], ["면적", "15,005평"], ["매매가", "148억 8060만"],
              ["매물 특징", "● 감곡IC-7분 거리"]]:
        ws.append(r)
    wb.save(data / "정보.xlsx")
    a = RealEstateAgent(Settings(data_dir=tmp_path / "매물", db_path=tmp_path / "n.db", log_dir=tmp_path / "l",
                                 llm_providers=["rule"]))
    a.sync()
    result, _ = ToolAgent(a).run("음성군 땅")
    assert "F-" not in result.answer and "음성군 전 15,005평 매물" in result.answer


def test_new_question_is_not_treated_as_follow_up(agent):
    ctx = AgentContext()
    runner = ToolAgent(agent, ctx=ctx)
    _, history = runner.run("음성군 창고 땅")
    result, _ = runner.run("양평 전원주택 부지 넓은 거", history)
    search = next(s for s in result.steps if s.tool == "search_listings")
    assert "use_previous_conditions" not in search.args and "양평" in search.args["query"]
