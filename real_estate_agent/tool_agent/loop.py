"""ReAct 방식 Tool-use Agent (프레임워크 없이 직접 구현, v0.2).

    질문 ─▶ [모델] 생각 → 도구 선택(action)
              ▲                 │
              │ 관찰(observation)│ 도구 실행 (인자 검사·오류 처리)
              └─────────────────┘   … 반복 …  → 최종 답변

- 모델: GPT/Claude 의 Function calling (LLMRouter.tool_step). 중간에 한쪽이 실패하면 다른 쪽이 이어 받음.
- 둘 다 안 되면 RulePolicy 가 같은 도구를 같은 방식(행동→관찰)으로 써서 답을 만듦.
- 안전장치: 최대 단계 수, 같은 호출 반복 감지, 도구 오류는 관찰로 돌려줘 모델이 스스로 고치게 함.
- 매 실행의 단계·지연시간·오류를 trace 로 남기고 logs/agent_runs.jsonl 에 기록 (평가용).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm import LLMUnavailable, ToolCall, ToolTurn
from ..models import format_price
from ..recommender import NOT_FOUND_MESSAGE
from .tools import AgentContext, ToolRegistry, observation_text

log = logging.getLogger(__name__)

MAX_STEPS = 6          # 모델 응답(행동) 최대 횟수
MAX_CALLS_PER_STEP = 4  # 한 번에 요청할 수 있는 도구 호출 수
MAX_HISTORY = 40       # 대화 기록 보관 (메시지 수)
try:  # v1.0: 질문 하나의 시간 예산 (초). 넘으면 AI 를 기다리지 않고 규칙 기반으로 마무리
    DEADLINE = max(1.0, float(os.getenv("REA_AGENT_DEADLINE", "45")))
except ValueError:
    DEADLINE = 45.0
_LLM_POOL = None  # AI 호출을 시간 제한과 함께 기다리기 위한 스레드 묶음


def _llm_pool():
    global _LLM_POOL
    if _LLM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _LLM_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm-call")
    return _LLM_POOL

SYSTEM_PROMPT = """당신은 충북 음성의 토지 전문 공인중개사무소 '상담 에이전트'입니다. 손님 질문에 답하려면 도구를 사용하세요.

규칙
- 매물 정보는 반드시 도구 결과(관찰)에서만 가져오세요. 도구로 확인하지 않은 매물·가격·면적을 지어내지 마세요.
- 매물을 찾을 때는 search_listings, 자세히 볼 때 get_listing, 지을 수 있는지 물으면 check_land_use,
  여러 매물을 비교하면 compare_listings, 손님이 마음에 든다/싫다고 하면 record_feedback 을 쓰세요.
- 도구가 오류를 돌려주면 오류 내용을 읽고 인자를 고쳐 다시 시도하세요. 같은 호출을 반복하지 마세요.
- 가격 인자는 만원 단위입니다 (1억 = 10000).
- 지번 전체와 소유자 연락처는 알 수 없습니다. 물으면 "상담 시 안내해 드립니다"라고 답하세요.
- 조건에 맞는 매물이 없으면 "해당 조건에 맞는 매물이 없습니다."라고 답하세요.
- 손님이 '다른 지역에도 있어?'라고 하면 search_listings 에 use_previous_conditions=true, other_regions=true 를,
  '음성군 말고 다른 지역'처럼 뺄 지역을 말하면 use_previous_conditions=true, exclude_regions=["음성군"] 을 주세요.
  여러 곳을 말하면 모두 넣으세요: '진천군이랑 음성군 말고' -> exclude_regions=["진천군", "음성군"].
  (regions 에 '음성군'을 넣으면 안 됩니다. 이때는 직전에 보여준 매물로 좁히지 않고 전체에서 다시 찾습니다)
- '정말 다른 지역에는 없어?', '진짜 없어?', '확실해?', '다시 찾아봐'는 방금 검색을 다시 확인해 달라는 뜻입니다.
  search_listings(recheck=true) 로 직전 조건(뺀 지역 포함)을 그대로 다시 찾고, 결과 개수와 지역별 개수를 분명히 말하세요.
  (이때 other_regions 를 쓰거나 뺀 지역을 바꾸지 마세요)
- 손님이 '그 중에', '거기서', '그럼 넓은 건?'처럼 직전 결과를 가리키면 search_listings 에
  use_previous_conditions=true 와 sort 를 주세요 (직전에 보여준 매물 안에서만 고릅니다). 하나만 물으면 limit=1.
- 매물을 가리킬 때는 '매물번호', 없으면 '표시이름'을 쓰세요. id(F-… 같은 내부 번호)는 손님에게 말하지 마세요.
- 화면에 매물 카드(지역·가격·면적·지목·특징)가 따로 나오므로, 최종 답변은 한국어 1~3문장의 짧은 요약만 쓰세요.
  매물은 매물번호(없으면 id)로 가리키고 핵심 수치(가격 또는 면적)만 괄호로 덧붙이세요.
  매물 목록을 줄줄이 나열하거나 특징을 그대로 옮겨 쓰지 말고, 표·마크다운(**, #, -)도 쓰지 마세요.
- 주소는 도구가 준 대로(시/도 + 시/군/구)만 말하세요.
- '어느 지역 매물이 제일 많아요?', '지역별로 몇 건이에요?' 처럼 수만 물으면 list_regions 만 쓰고 숫자로만 답하세요 (매물을 검색하지 마세요).
  그다음 '제일 많은 지역 매물 보여줘', '그 지역 매물 보고 싶어요' 라고 하면 그 지역으로 search_listings(regions=[지역]) 하세요.
- 건축 가능 여부는 프로그램의 1차 판단이며 토지이용계획 확인이 필요하다고 덧붙이세요.
"""


@dataclass
class Step:
    n: int
    kind: str                 # 'tool' / 'answer' / 'fallback' / 'error' / 'guard'
    policy: str = ""          # openai / anthropic / rule
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    ok: Optional[bool] = None
    summary: str = ""         # 화면 표시용 한 줄 요약
    ms: float = 0.0
    thought: str = ""         # 모델이 도구 호출과 함께 쓴 말 (있으면)
    agent: str = ""           # v0.7 멀티 Agent: 이 단계를 한 Agent (총괄 / 검색 / 분석)


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step]
    listing_ids: list[str]    # 답변에서 다룬 매물 (화면 카드용)
    policy: str               # 마지막으로 답을 만든 쪽
    stopped: str              # 'answer' / 'max_steps' / 'fallback'
    total_ms: float
    run_id: str = ""
    badges: dict[str, list[str]] = field(default_factory=dict)  # 매물별 표시 (최저가, 창고 가능 …)
    decisions: int = 0        # v0.4: 판단 횟수 (AI 모드라면 AI 호출 수)
    trace_id: str = ""        # v1.1: 이 질문의 추적 id (관리자 '관측' 화면에서 단계별 시간 확인)
    span_id: str = ""         # v1.2: 이 질문의 맨 위 span (손님 평가를 같은 trace 에 이어 붙일 때)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def tool_errors(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool" and s.ok is False)


@dataclass
class _Run:
    """질문 하나를 처리하는 동안의 상태 (반복문·LangGraph 노드가 함께 씀)."""
    question: str
    messages: list
    steps: list
    tools: list
    extra: str
    use_llm: bool
    started: float
    seen_calls: dict = field(default_factory=dict)
    answer: str = ""
    policy: str = "rule"
    stopped: str = "max_steps"
    decisions: int = 0     # v0.4: 판단(AI 모드라면 AI 호출) 횟수 - 구조별 비용 비교용


# ---------------------------------------------------------------------- 정책: 다음 행동을 고르는 쪽
class LLMPolicy:
    def __init__(self, llm, prompt: str = SYSTEM_PROMPT):
        self.llm = llm
        self.prompt = prompt  # 멀티 Agent 에서는 역할별 지시문

    @property
    def available(self) -> bool:
        return bool(self.llm and self.llm.available and self.llm.ready)

    def next(self, messages: list[dict], tools: list[dict], extra: str = "") -> ToolTurn:
        return self.llm.tool_step(self.prompt + extra, messages, tools)


class RulePolicy:
    """LLM 없이 같은 도구를 쓰는 규칙 기반 정책: 검색 → (비교/용도 확인) → 답변."""

    name = "rule"

    def __init__(self, ctx: Optional[AgentContext] = None, agent=None):
        self.ctx = ctx
        self.ctx_agent = agent  # 매물 찾기용 (분석 Agent 가 직전 매물을 볼 때)

    def next(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        done = [(m["name"], _loads(m["content"])) for m in messages[start:] if m["role"] == "tool"]
        names = [n for n, _ in done]
        last_search = next((o for n, o in reversed(done) if n == "search_listings" and o.get("ok")), None)
        found = (last_search or {}).get("result", {}).get("매물", [])
        have = {t["name"] for t in tools} if tools else None  # 멀티 Agent: 이 Agent 가 가진 도구만 씀
        if have is not None and "search_listings" not in have:
            return self._analyze(question, done, names, have)
        region_turn = self._regions(question, names, done, have)  # '어느 지역 매물이 제일 많아요?' / '그 지역 매물 보여줘'
        if region_turn is not None:
            return region_turn
        named = _named_listing(self.ctx_agent, question)  # v1.3: '(매물번호)에 공장 지을 수 있어?' → 그 매물을 바로 확인
        if (named and _purpose_in(question) and _asks_buildable(question) and "check_land_use" not in names
                and (have is None or "check_land_use" in have)):
            return _call("check_land_use", {"listing_id": named, "purpose": _purpose_in(question)})
        if named and "check_land_use" in names and not any(n == "search_listings" for n in names):
            return ToolTurn(text=rule_answer(question, done), provider="rule")

        if (_mentions_past(question) and "recall_memory" not in names and "search_listings" not in names
                and (have is None or "recall_memory" in have)):
            return _call("recall_memory", {})  # '지난번에 찾던 거' -> 먼저 장기 기억 확인
        if "search_listings" not in names and self.ctx and self.ctx.last_search and _wants_recheck(question):
            # '정말 다른 지역에는 없어?' -> 방금 검색(뺀 지역 포함)을 그대로 다시 확인
            return _call("search_listings", {"recheck": True, "limit": _wanted_count(question) or 5})
        if "search_listings" not in names and self.ctx and self.ctx.last_search and _region_follow_up(question):
            # '진천군에도 있어?' / '음성은?' -> 앞 조건(용도·예산 등) 그대로, 지역만 바꿔서
            args = {"query": question, "use_previous_conditions": True, "limit": _wanted_count(question) or 5}
            if _sort_in(question):
                args["sort"] = _sort_in(question)
            return _call("search_listings", args)
        if "search_listings" not in names and self.ctx and self.ctx.last_search and _wants_other_region(question):
            # '음성군 말고 다른 지역에도 있어?' / '다른 지역에도 있어?' -> 앞 조건(용도 등) 그대로, 지역만 바꿔 전체에서
            sort = _sort_in(question)
            args = {"query": question, "use_previous_conditions": True, "limit": _wanted_count(question) or 5}
            if not _names_excluded_region(question):
                args["other_regions"] = True  # 지역을 말하지 않았으면 방금 보여준 지역을 뺌
            if sort:
                args["sort"] = sort
            return _call("search_listings", args)
        if "search_listings" not in names:
            sort = _sort_in(question)
            if _mentions_past(question) and self.ctx and self.ctx.last_search:
                return _call("search_listings", {"use_previous_conditions": True, "sort": sort or "score",
                                                 "limit": _wanted_count(question) or 5})
            if self.ctx and self.ctx.last_search and _is_follow_up(question, sort):
                # '그 중에 넓은 건?' -> 직전 조건 그대로, 새 기준으로 1건 (개수를 말하면 그만큼)
                args: dict[str, Any] = {"use_previous_conditions": True, "sort": sort or "score",
                                        "limit": _wanted_count(question) or 1}
                return _call("search_listings", args)
            limit = _wanted_count(question) or (3 if _wants_compare(question) else 5)
            args = {"query": question, "limit": limit}
            if sort:
                args["sort"] = sort
            return _call("search_listings", args)
        if (last_search or {}).get("result", {}).get("앞_조건_이어서"):
            return ToolTurn(text=rule_answer(question, done), provider="rule")
        if have is not None and not have & {"compare_listings", "check_land_use"}:
            return ToolTurn(text=rule_answer(question, done), provider="rule")  # 검색 Agent: 찾은 결과만 보고
        return self._analyze(question, done, names, have, found)

    def _regions(self, question: str, names: list, done: list, have: Optional[set]) -> Optional[ToolTurn]:
        """지역별 매물 수 질문은 수만 답하고, '제일 많은 지역/그 지역 매물' 을 보자고 하면 그 지역으로 검색."""
        from ..regions import find_regions

        if have is not None and "list_regions" not in have:
            return None
        if "search_listings" in names:
            return None
        counted = next((o for n, o in reversed(done) if n == "list_regions" and o.get("ok")), None)
        ranked = (counted or {}).get("result", {}).get("지역별_매물_수") or {}
        ranked = list(ranked) or (list(self.ctx.last_regions) if self.ctx else [])
        idx = _region_rank_ref(question, pronoun_ok=bool(self.ctx and self.ctx.last_regions) or bool(counted))
        wants = idx is not None and _wants_to_see(question) and not _asks_region_counts(question) \
            and not find_regions(question)
        if wants and ranked:  # '제일 많은 지역 매물을 보고 싶어요' → 그 지역 매물
            region = ranked[idx] if -len(ranked) <= idx < len(ranked) else ranked[0]
            focus = self.ctx.region_focus if self.ctx else ""
            if focus and not any(p.search(question) for p, _ in _RANK_REF):  # '그 지역' = 방금 답에서 말한 지역
                region = focus
            args: dict[str, Any] = {"query": question, "regions": [region], "limit": _wanted_count(question) or 5}
            if _sort_in(question):
                args["sort"] = _sort_in(question)
            return _call("search_listings", args)
        if counted:  # 수만 물었으면 수만 답함 (매물 카드 없음)
            if self.ctx and ranked:  # 이 답이 가리키는 지역을 기억 ('그 지역 매물 보여줘')
                asked = [r for r in ranked if r in question or r[:-1] in question]
                self.ctx.region_focus = asked[0] if asked else (ranked[-1] if idx == -1 else ranked[0])
            return ToolTurn(text=rule_answer(question, done), provider="rule")
        if _asks_region_counts(question) or wants:  # 아직 순위를 모르면 먼저 셈
            return _call("list_regions", {})
        return None

    def _analyze(self, question: str, done: list, names: list, have: Optional[set], found=None) -> ToolTurn:
        """비교 / 건축 가능 여부 확인 -> 답. found 가 없으면(분석 Agent) 직전에 보여준 매물을 대상으로."""
        if found is None:
            found = self._shown()
        can = lambda tool: have is None or tool in have  # noqa: E731
        if _wants_compare(question) and len(found) >= 2 and "compare_listings" not in names and can("compare_listings"):
            k = min(len(found), _wanted_count(question) or 2, 5)
            return _call("compare_listings", {"listing_ids": [x["id"] for x in found[:k]]})
        purpose = _purpose_in(question)
        named = _named_listing(self.ctx_agent, question)  # 손님이 매물번호를 말했으면 그 매물이 대상
        if (purpose and (named or found) and "check_land_use" not in names and _asks_buildable(question)
                and can("check_land_use")):
            return _call("check_land_use", {"listing_id": named or found[0]["id"], "purpose": purpose})
        return ToolTurn(text=rule_answer(question, done), provider="rule")

    def _shown(self) -> list[dict]:
        """대화 상태(직전에 보여준 매물) -> [{'id': 도구용 id}]"""
        if not self.ctx or not self.ctx.last_shown:
            return []
        from .tools import tool_id

        out = []
        for lid in self.ctx.last_shown:
            l = self.ctx_agent.find_listing(lid) if self.ctx_agent else None
            out.append({"id": tool_id(l) if l else lid})
        return out


def _call(name: str, args: dict) -> ToolTurn:
    return ToolTurn(tool_calls=[ToolCall(id=f"rule_{uuid.uuid4().hex[:8]}", name=name, arguments=args)],
                    provider="rule")


def _loads(text: str) -> dict:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}


_FOLLOW_UP = re.compile(r"그\s*중|이\s*중|저\s*중|거기서|그\s*가운데|둘\s*중|셋\s*중|방금|위\s*매물|그\s*매물들|그럼|그러면")


_OTHER_REGION = re.compile(r"다른\s*(지역|곳|데|동네|시|군|쪽)|딴\s*(데|곳|지역)|(말고|빼고|외에|제외하고)\s*(다른|딴)?|아닌\s*(곳|데|지역)")
_EXCLUDED_NAME = re.compile(r"([가-힣]{2,8}(?:시|군|구|읍|면)|[가-힣]{2,4})\s*(?:은|는|을|를)?\s*(?:말고|빼고|제외(?:하고)?|외에|아닌)")


_RECHECK = re.compile(r"정말|진짜|확실|혹시|다시\s*(?:한\s*번\s*)?(?:찾|확인|검색|봐|알아)|한\s*번\s*더|더\s*찾아")


def _wants_recheck(q: str) -> bool:
    """'정말 다른 지역에는 없어?', '진짜 없어?', '확실해?', '다시 찾아봐' -> 방금 검색을 다시 확인해 달라는 말.
    단, 새 조건(지역·용도·예산 등)을 함께 말하면 새 검색으로 봄."""
    if not _RECHECK.search(q):
        return False
    from ..regions import find_regions

    stripped = re.sub(r"다른\s*(지역|곳|데|동네)", "", q)
    if find_regions(stripped) or _purpose_in(q) or re.search(r"\d|억|천\s*만|평", q):
        return False
    return bool(re.search(r"없|있|맞|확실|찾|확인|검색|봐", q))


def _region_follow_up(q: str) -> bool:
    """지역만 바꿔 묻는 짧은 후속 질문인지: '진천군에도 있어?', '음성은?', '대소읍 쪽은 어때?'"""
    from ..regions import find_regions

    if len(q.strip()) > 20 or _purpose_in(q) or re.search(r"\d|억|천\s*만|평|말고|빼고|외에|다른", q):
        return False
    return bool(find_regions(q) or re.search(r"[가-힣]{1,4}(읍|면)(에도|은|는|쪽|도|$)", q))


_REGION_COUNT = re.compile(
    r"(어느|어떤|어디|무슨)\s*(지역|동네|곳|시군|쪽|데)?[^?]{0,14}(많|적[은어었다게지나])"
    r"|지역\s*별|지역마다|몇\s*(건|개)\s*씩|어디(에|가)\s*(매물이?\s*)?(제일|가장)?\s*많"
    r"|(매물|땅)[이가은는]?\s*(몇\s*(건|개)|얼마나)\s*(있|되|돼)")
_RANK_REF = [(re.compile(r"(두|2)\s*번\s*째\s*(로)?\s*많"), 1), (re.compile(r"(세|3)\s*번\s*째\s*(로)?\s*많"), 2),
             (re.compile(r"(제일|가장|젤)\s*적[은어었다게지나]"), -1), (re.compile(r"(제일|가장|젤)\s*많"), 0)]
_REGION_PRONOUN = re.compile(r"그\s*지역|거기|그곳|그\s*동네|그\s*쪽|그\s*시군")


def _asks_region_counts(q: str) -> bool:
    """'어느 지역 매물이 제일 많아요?', '지역별로 몇 건이에요?', '음성군에 매물 몇 개 있어요?' - 수를 묻는 질문."""
    return bool(_REGION_COUNT.search(q))


def _region_rank_ref(q: str, pronoun_ok: bool = False) -> Optional[int]:
    """'제일 많은 지역' → 0, '두 번째로 많은' → 1, '제일 적은' → -1, ('그 지역' 은 직전에 순위를 알려줬을 때만) → 0"""
    for pattern, idx in _RANK_REF:
        if pattern.search(q) and re.search(r"지역|곳|데|동네|시|군", q):
            return idx
    if pronoun_ok and _REGION_PRONOUN.search(q):
        return 0
    return None


def _wants_to_see(q: str) -> bool:
    return bool(re.search(r"보여|보고\s*싶|볼\s*수|알려|추천|찾아|소개|매물|땅|뭐\s*있|어떤\s*(게|거)", q))


def _wants_other_region(q: str) -> bool:
    return bool(_OTHER_REGION.search(q))


def _names_excluded_region(q: str) -> bool:
    """'음성군 말고' 처럼 뺄 지역을 직접 말했는지."""
    from ..regions import canonical_region, find_regions

    for m in _EXCLUDED_NAME.finditer(q):
        w = m.group(1)
        if find_regions(w) or find_regions(canonical_region(w)) or w[-1:] in "읍면":
            return True
    return False


def _mentions_past(q: str) -> bool:
    return bool(re.search(r"지난\s*번|저번|지난\s*상담|예전에|전에\s*(찾|보|말)|찾던|보던", q))


def _sort_in(q: str) -> str:
    if re.search(r"평당", q):
        return "price_per_pyeong_asc"
    if re.search(r"비싼|고가", q):
        return "price_desc"
    if re.search(r"싼|저렴|최저가|가격\s*낮", q):
        return "price_asc"
    if re.search(r"넓은|넓|큰\s*(땅|거|건|것)|면적\s*큰|제일\s*큰|가장\s*큰", q):
        return "area_desc"
    return ""


def _is_follow_up(q: str, sort: str) -> bool:
    """직전 결과를 가리키는 후속 질문인지 ('그 중에 넓은 건?', '거기서 제일 싼 거', '더 넓은 건?')."""
    if _FOLLOW_UP.search(q):
        return True
    # 지역·용도 없이 기준만 짧게 물으면 후속 질문으로 봄 ('넓은 건?', '제일 싼 거는?')
    return bool(sort) and len(q.strip()) <= 15 and not _purpose_in(q) and not re.search(r"[가-힣]+(시|군|구|읍|면)\b", q)


def _wants_compare(q: str) -> bool:
    return bool(re.search(r"비교|차이|어느\s*(게|것|쪽)|뭐가\s*나|중에\s*(뭐|어느)", q))


def _asks_buildable(q: str) -> bool:
    return bool(re.search(r"지을\s*수|건축|가능|되나|될까|돼요|되요", q))


def _named_listing(agent, q: str) -> Optional[str]:
    """질문에 나온 매물번호 (공개 id) - 없으면 None."""
    if agent is None:
        return None
    from ..agent import public_office_no
    from .tools import tool_id

    for l in agent.listings():
        for key in {public_office_no(l.office_no), tool_id(l)} - {""}:
            if len(key) >= 3 and re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(key)}(?![A-Za-z0-9_-])", q):
                return tool_id(l)
    return None


def _wanted_count(q: str) -> Optional[int]:
    m = re.search(r"(\d+|한|두|세|네|다섯)\s*(개|곳|건|군데)", q)
    if not m:  # v1.3: '넓은 걸로 하나', '두 개만' 외에 '둘만', '셋 보여줘' 도 개수로 (단 '하나도 없어?' 는 아님)
        m = re.search(r"(?<![가-힣])(하나|둘|셋|넷)(?!도)(?=\s*(?:만|요|$|[?.!]|보여|알려|추천|찾아|줘|골라))", q)
        if not m:
            return None
    words = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "하나": 1, "둘": 2, "셋": 3, "넷": 4}
    n = words.get(m.group(1)) or int(m.group(1))
    return max(1, min(n, 10))


def _purpose_in(q: str) -> str:
    from ..land_use import RULES

    for purpose, rule in RULES.items():
        if purpose in q or re.search(rule["words"], q):
            return purpose
    return ""


_BEST = {"넓은 순": ("가장 넓은", "area"), "가격 낮은 순": ("가장 싼", "price"), "가격 높은 순": ("가장 비싼", "price"),
         "평당가 낮은 순": ("평당가가 가장 낮은", "ppp"), "추천순": ("조건에 가장 잘 맞는", "price")}


def _fact(x: dict, kind: str) -> str:
    if kind == "area":
        return f"{x['면적_평']:,}평, {x['가격']}"
    if kind == "ppp":
        return f"평당 {x['평당가_만원']:,}만원"
    return f"{x['가격']}, {x['면적_평']:,}평"


def rule_answer(question: str, done: list[tuple[str, dict]]) -> str:
    """도구 결과만으로 짧은 요약 답변 (LLM 없이). 매물 상세는 화면의 카드가 보여줌."""
    search = next((o for n, o in reversed(done) if n == "search_listings" and o.get("ok")), None)
    failed = next((o for n, o in reversed(done) if n == "search_listings" and not o.get("ok")), None)
    counted = next((o for n, o in reversed(done) if n == "list_regions" and o.get("ok")), None)
    if counted and search is None:  # '어느 지역 매물이 제일 많아요?' → 수만 답함
        return _region_count_answer(question, counted["result"])
    if search is None and failed:
        return "죄송합니다. 매물을 찾는 중 문제가 생겼습니다: " + failed.get("error", "")
    res = (search or {}).get("result", {})
    found = res.get("매물", [])
    if res.get("다시_확인"):  # '정말 없어?' -> 없으면 왜 없는지까지
        return _recheck_answer(res, {x["id"]: x.get("매물번호") or x.get("표시이름") or x["id"] for x in found})
    if search and not found:
        return NOT_FOUND_MESSAGE
    names = {x["id"]: x.get("매물번호") or x.get("표시이름") or x["id"] for x in found}
    total, cond = res.get("조건에_맞는_매물_수", len(found)), res.get("조건_요약", "말씀하신 조건")
    order = res.get("정렬", "추천순")
    lines: list[str] = []

    if found and res.get("앞_조건_출처") == "지난 상담":
        lines.append(f"지난 상담에서 찾으신 조건({cond})으로 다시 찾았습니다.")
    elif found and res.get("제외_지역") and res.get("앞_조건_이어서") and not res.get("앞서_보여준_매물_수"):
        lines.append(f"{', '.join(res['제외_지역'])} 말고 다른 지역에서 같은 조건으로 다시 찾았습니다.")
    elif found and res.get("앞서_보여준_매물_수"):  # '그 중에 넓은 건?' -> 보여준 매물 안에서 골라 한 줄로
        best, kind = _BEST.get(order, _BEST["추천순"])
        pool = f"앞서 보여드린 {res['앞서_보여준_매물_수']}건"
        if len(found) == 1:
            x = found[0]
            return f"{pool} 중 {best} 매물은 {names[x['id']]}({_fact(x, kind)})입니다."
        return f"{pool} 중 {best} {len(found)}건입니다."
    elif found and res.get("앞_조건_이어서"):  # '진천군에도 있어?' -> 앞 조건 그대로 다른 곳에서
        lines.append("앞의 조건 그대로 다시 찾았습니다.")

    if found and res.get("학습_반영"):  # v1.2 온라인 학습
        lines.append(res["학습_반영"] + ".")
    if found:
        how = "추천 순" if order == "추천순" else order
        lines.append(f"{cond} 매물 {total}건 중 {how}으로 {len(found)}건입니다." if total > len(found) else
                     f"{cond} 매물 {total}건입니다 ({how}).")
    cmp_ = next((o for n, o in reversed(done) if n == "compare_listings" and o.get("ok")), None)
    if cmp_:
        r = cmp_["result"]
        rows = {x["id"]: x for x in r["매물"]}
        names.update({i: x.get("매물번호") or x.get("표시이름") or i for i, x in rows.items()})
        cheap, wide = r["가장_싼_매물"], r["가장_넓은_매물"]
        if cheap == wide:
            lines.append(f"가장 싸고 넓은 매물: {names[cheap]}({rows[cheap]['가격']}, {rows[wide]['면적_평']:,}평)")
        else:
            lines.append(f"최저가 {names[cheap]}({rows[cheap]['가격']}) · 가장 넓은 매물 {names[wide]}"
                         f"({rows[wide]['면적_평']:,}평)")
    elif found:
        top = found[0]
        lines.append(f"1순위 {names[top['id']]}({_fact(top, 'price')})")
    use = next((o for n, o in reversed(done) if n == "check_land_use" and o.get("ok")), None)
    if use and "가능" in use["result"]:
        r = use["result"]
        verdict = "건축 가능해 보입니다" if r["가능"] else "건축이 어려워 보입니다"
        lines.append(f"{names.get(r['id'], r['id'])}: {r['용도']} {verdict} (토지이용계획 확인 필요)")
    return "\n".join(lines) or "질문을 이해하지 못했습니다. 지역·용도·예산을 함께 말씀해 주세요."


def _iga(word: str) -> str:
    """받침이 있으면 '이', 없으면 '가' ('양평군이', '여주시가')."""
    ch = word[-1:] if word else ""
    return "이" if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 else "가"


def _region_count_answer(question: str, res: dict) -> str:
    counts: dict = res.get("지역별_매물_수") or {}
    total = res.get("전체_매물_수", sum(counts.values()))
    if not counts:
        return "지금 보유한 매물이 없습니다."
    asked = [r for r in counts if r in question or r[:-1] in question]
    if asked:  # '음성군에 매물 몇 개 있어요?'
        return " / ".join(f"{r} 매물은 {counts[r]}건입니다" for r in asked) + f" (전체 {total}건 중)."
    ranked = list(counts.items())
    if re.search(r"(제일|가장|젤)\s*적[은어었다게지나]", question):
        r, n = ranked[-1]
        return f"보유 매물 {total}건 중 {r}{_iga(r)} {n}건으로 가장 적습니다."
    (top, n), rest = ranked[0], ranked[1:4]
    line = f"보유 매물 {total}건 중 {top}{_iga(top)} {n}건으로 가장 많습니다."
    if rest:
        line += " 그다음은 " + ", ".join(f"{r} {c}건" for r, c in rest) + "입니다."
    return line


def _recheck_answer(res: dict, names: dict) -> str:
    """'정말 없어?' 에 대한 답: 같은 조건으로 다시 찾은 결과를 개수·지역별로 분명하게."""
    cond, total = res.get("조건_요약", "말씀하신 조건"), res.get("조건에_맞는_매물_수", 0)
    if not total:
        lines = [f"다시 확인해 봐도 {cond} 조건에 맞는 매물은 보유 매물 {res.get('전체_보유_매물_수', 0)}건 중 없습니다."]
        hints = res.get("조건을_줄이면") or {}
        if hints.get("도로_조건을_빼면"):
            lines.append(f"도로 접함 조건을 빼면 {hints['도로_조건을_빼면']}건이 있습니다.")
        if hints.get("용도_조건을_빼면"):
            lines.append(f"용도 조건을 빼면 {hints['용도_조건을_빼면']}건이 있습니다.")
        return "\n".join(lines)
    regions = ", ".join(f"{r} {n}건" for r, n in (res.get("지역별_매물_수") or {}).items())
    found = res.get("매물", [])
    lines = [f"다시 확인했습니다. {cond} 조건에 맞는 매물이 {total}건 있습니다 ({regions})."]
    if found:
        top = found[0]
        lines.append(f"추천 순으로 {len(found)}건을 보여드립니다. 1순위 {names[top['id']]}({_fact(top, 'price')})")
    return "\n".join(lines)


def listing_badges(done: list[tuple[str, dict]]) -> dict[str, list[str]]:
    """도구 결과에서 매물별 표시: '최저가', '가장 넓음', '평당가 최저', '창고 가능' …"""
    badges: dict[str, list[str]] = {}

    def add(lid: str, text: str) -> None:
        if text not in badges.setdefault(lid, []):
            badges[lid].append(text)

    for name, obs in done:
        if not obs.get("ok"):
            continue
        r = obs.get("result", {})
        if name == "compare_listings":
            add(r["가장_싼_매물"], "최저가")
            add(r["가장_넓은_매물"], "가장 넓음")
            add(r["평당가가_가장_낮은_매물"], "평당가 최저")
        elif name == "check_land_use" and "가능" in r:
            add(r["id"], f"{r['용도']} {'가능' if r['가능'] else '어려움'}")
        elif name == "record_feedback":
            add(r["id"], "관심" if r.get("기록") == "like" else "제외")
    return badges


# ---------------------------------------------------------------------- Agent 루프
class ToolAgent:
    uses_policy = True  # v0.4: 고정 파이프라인(Workflow)은 판단하지 않음
    """memory: LongTermMemory (방문자별 장기 기억), owner: 방문자 id,
    checkpoint(messages, ctx): 단계마다 상태 저장, on_event(dict): 진행 상황 실시간 전달 (스트리밍)."""

    def __init__(self, agent, llm=None, ctx: Optional[AgentContext] = None, lock=None,
                 max_steps: int = MAX_STEPS, log_dir: Optional[Path] = None, memory=None, owner: str = "",
                 checkpoint: Optional[Callable[[list, AgentContext], None]] = None,
                 on_event: Optional[Callable[[dict], None]] = None, tool_source=None,
                 system_prompt: str = SYSTEM_PROMPT, only: Optional[set[str]] = None, name: str = "",
                 trace: Optional[list] = None, rule_policy=None, deadline: Optional[float] = None,
                 guards: bool = True):
        """v0.7 멀티 Agent 용: system_prompt(역할 지시), only(쓸 도구), name(trace 에 표시할 이름),
        trace(여러 Agent 가 함께 쓰는 단계 기록), rule_policy(LLM 이 없을 때 쓸 정책).
        v1.0: deadline(질문 하나의 시간 예산, 초), guards(입력·출력 가드레일 사용)."""
        self.agent = agent
        self.registry = ToolRegistry(agent, ctx, lock, memory=memory, owner=owner, tool_source=tool_source,
                                     only=only)
        self.llm_policy = LLMPolicy(llm if llm is not None else agent.llm, system_prompt)
        self.rule_policy = rule_policy or RulePolicy(self.registry.ctx, agent)
        self.name, self.trace = name, trace
        self.answer_label = "최종 답변"  # 멀티 Agent 의 전문 Agent 는 '총괄 Agent 에게 보고'
        self.max_steps = max_steps
        self.log_dir = log_dir if log_dir is not None else getattr(agent.settings, "log_dir", None)
        self.memory, self.owner = memory, owner
        self.checkpoint, self.on_event = checkpoint, on_event
        self.deadline = deadline if deadline is not None else DEADLINE
        self.guards = guards
        self.final_answerer = True  # 손님에게 직접 답하는 Agent (멀티 Agent 의 전문 Agent 는 False)
        self.mode_label = "single"  # v1.1 지표용: single / multi / distributed
        # 답변 검증에 쓸 추가 근거 (멀티 Agent 총괄: 전문 Agent 들이 받은 도구 결과), 검증 실패 시 다시 쓰는 방법
        self.evidence: Callable[[], list] = lambda: []
        self.fallback_answer: Callable[[str, list], str] = rule_answer
        self._input = None

    @property
    def ctx(self) -> AgentContext:
        return self.registry.ctx

    def run(self, question: str, history: Optional[list[dict]] = None) -> tuple[AgentResult, list[dict]]:
        """질문 하나 처리. (결과, 갱신된 대화 기록) 반환. v1.1: 질문 하나 = span 하나 (안의 AI·도구 호출은 하위 span)."""
        return self._traced(lambda: self._run(question, history), question)

    def _traced(self, fn: Callable[[], tuple], question: str) -> tuple[AgentResult, list[dict]]:
        from ..observability.metrics import AGENT_LATENCY, AGENT_REQUESTS
        from ..observability.tracing import TRACER, redact

        started = time.perf_counter()
        with TRACER.span("agent.run", agent=self.name or "상담 Agent", mode=self.mode_label,
                         question=redact(question)) as sp:
            try:
                result, history = fn()
            except Exception:
                if self.final_answerer:
                    AGENT_REQUESTS.inc(mode=self.mode_label, policy="error", stopped="exception")
                raise
            result.trace_id, result.span_id = sp.trace_id, sp.span_id
            sp.set(policy=result.policy, stopped=result.stopped, steps=len(result.steps),
                   tool_calls=result.tool_calls, tool_errors=result.tool_errors,
                   listings=len(result.listing_ids), answer_chars=len(result.answer),
                   guards=sum(1 for s in result.steps if s.kind == "guard") or None)
        if self.final_answerer:
            AGENT_REQUESTS.inc(mode=self.mode_label, policy=result.policy, stopped=result.stopped)
            AGENT_LATENCY.observe(time.perf_counter() - started, mode=self.mode_label)
        return result, history

    def _run(self, question: str, history: Optional[list[dict]] = None) -> tuple[AgentResult, list[dict]]:
        from ..reliability.guardrails import REFUSAL, check_input
        from ..reliability.stats import STATS

        self._input = check_input(question) if self.guards else None
        pre: list[Step] = []
        if self._input and self._input.changed:
            STATS.add("injections_blocked")
            pre.append(Step(0, "guard", "rule", summary="지시를 바꾸려는 문장은 무시함: "
                                                        + " / ".join(self._input.removed)[:120]))
        if self._input and self._input.refuse:  # 상담과 관계없는 지시만 있음 -> AI 를 부르지 않고 정해진 안내
            started = time.perf_counter()
            messages = list(history or []) + [{"role": "user", "content": question},
                                              {"role": "assistant", "content": REFUSAL}]
            for s in pre:
                self._add([], s)
            result = AgentResult(question, REFUSAL, pre, [], "rule", "guard",
                                 round((time.perf_counter() - started) * 1000, 1), uuid.uuid4().hex[:12])
            self._log(result)
            self._emit("done", answer=REFUSAL, listing_ids=[], policy="rule", total_ms=result.total_ms)
            return result, trim_history(messages)
        q = self._input.question if self._input else question
        messages = list(history or []) + [{"role": "user", "content": q}]
        result, history_out = self._loop(q, messages, pre)
        result.question = question  # 화면에는 손님이 쓴 그대로
        return result, history_out

    def resume(self, history: list[dict]) -> tuple[AgentResult, list[dict]]:
        """답을 만들다 끊긴 대화를 마지막 체크포인트부터 이어서 진행."""
        question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return self._traced(lambda: self._resume(history), question)

    def _resume(self, history: list[dict]) -> tuple[AgentResult, list[dict]]:
        from .memory import repair_for_resume

        messages = repair_for_resume(history)
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        done = sum(1 for m in messages[max(i for i, m in enumerate(messages) if m["role"] == "user"):]
                   if m["role"] == "tool")
        return self._loop(question, messages, [Step(0, "resume", "", summary=f"끊긴 상담을 체크포인트에서 이어서 진행 "
                                                                              f"(이미 받은 도구 결과 {done}개 재사용)")])

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            try:
                self.on_event({"type": kind, **data})
            except Exception as e:  # 화면 연결이 끊겨도 Agent 는 계속
                log.debug("event 전달 실패: %s", e)

    def _save(self, messages: list[dict]) -> None:
        if self.checkpoint:
            try:
                self.checkpoint(messages, self.ctx)
            except Exception as e:
                log.warning("체크포인트 저장 실패: %s", e)

    def _add(self, steps: list[Step], step: Step) -> None:
        if self.name and not step.agent:
            step.agent = self.name
        steps.append(step)
        if self.trace is not None:
            self.trace.append(step)
        self._emit("step", step=asdict(step))

    def _loop(self, question: str, messages: list[dict], steps: list[Step]) -> tuple[AgentResult, list[dict]]:
        started = time.perf_counter()
        seen_before = len(self.ctx.seen)
        self.ctx.run_searches = []
        extra = ""
        if self.memory and self.owner:  # 장기 기억: 싫다고 한 매물 제외 + AI 에게 요약 전달
            from .memory import memory_summary
            from .tools import display_name

            mem = self.memory.apply_to(self.owner, self.ctx)
            listing = {l.listing_id: l for l in self.agent.listings()}
            summary = memory_summary(mem, lambda i: display_name(listing[i]) if i in listing else "(지금은 없는 매물)")
            if summary:
                extra = "\n\n이 손님에 대한 기억(지난 상담): " + summary
                self._add(steps, Step(0, "memory", "", summary="장기 기억 불러옴 · " + summary[:160]))
        tools = self.registry.schemas()
        for s in steps:
            if self.name and not s.agent:
                s.agent = self.name
            if self.trace is not None and s not in self.trace:
                self.trace.append(s)
            self._emit("step", step=asdict(s))
        use_llm = self.llm_policy.available
        if not use_llm and self.uses_policy:
            self._add(steps, Step(0, "fallback", "rule", summary="AI(GPT/Claude)를 쓸 수 없어 규칙 기반 정책으로 진행"))
        self._save(messages)

        run = _Run(question, messages, steps, tools, extra, use_llm, started)
        self._run_steps(run)
        answer, policy, stopped = run.answer, run.policy, run.stopped
        self._last_decisions = run.decisions

        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        done = [(m["name"], _loads(m["content"])) for m in messages[start:] if m["role"] == "tool"]
        answer, policy = self._check_output(question, answer, policy, done, steps, messages)
        badges = {}
        for key, values in listing_badges(done).items():  # 도구 id(공개용) -> 실제 매물 id
            badges[self.registry.resolve(key) or key] = values
        result = AgentResult(question, answer or NOT_FOUND_MESSAGE, steps, self._cards(done, answer, seen_before), policy,
                             stopped, round((time.perf_counter() - started) * 1000, 1), uuid.uuid4().hex[:12],
                             badges, decisions=getattr(self, "_last_decisions", 0))
        if self.memory and self.owner:  # 장기 기억 갱신 (새로 찾은 조건, 관심/제외 매물)
            self.memory.update_after(self.owner, self.ctx, self.ctx.run_searches)
        self.last_run_searches = list(self.ctx.run_searches)  # v0.8: 원격 Agent 가 장기 기억 갱신용으로 돌려줌
        self.ctx.run_searches = []
        self._log(result)
        history = trim_history(messages)
        self._emit("done", answer=result.answer, listing_ids=result.listing_ids, policy=result.policy,
                   total_ms=result.total_ms)
        return result, history

    # ------------------------------------------------------------------ 한 단계 (판단 → 도구 실행)
    def _run_steps(self, run: "_Run") -> None:
        """직접 만든 ReAct 반복문. (v0.5: LangGraph 판은 같은 _step 을 그래프 노드로 돌림)"""
        for n in range(1, self.max_steps + 1):
            if self._step(run, n) == "done":
                return
        self._wrap_up(run)

    def _step(self, run: "_Run", n: int) -> str:
        """판단 한 번 + 고른 도구 실행. 반환: 'done'(답을 냄) / 'continue'."""
        from ..reliability.stats import STATS

        messages, steps = run.messages, run.steps
        t0 = time.perf_counter()
        self._emit("thinking", n=n)
        remaining = self.deadline - (t0 - run.started)
        if run.use_llm and remaining <= 0.05:  # v1.0 시간 예산: AI 를 더 기다리지 않고 규칙으로 마무리
            STATS.add("deadline_hits")
            self._add(steps, Step(n, "guard", "rule",
                                  summary=f"시간 예산 {self.deadline:g}초를 다 써서 규칙 기반으로 마무리"))
            run.use_llm = False
        try:
            run.decisions += 1
            turn = (self._llm_call(self.llm_policy.next, remaining, list(messages), run.tools, run.extra)
                    if run.use_llm else self.rule_policy.next(messages, run.tools))
        except LLMUnavailable as e:
            STATS.add("llm_fallbacks")
            self._add(steps, Step(n, "fallback", "rule", summary=f"AI 응답 실패 → 규칙 기반으로 이어서 진행 ({e})",
                                  ms=_ms(t0)))
            run.use_llm = False
            return "continue"
        policy = run.policy = turn.provider or ("rule" if not run.use_llm else "llm")
        if not turn.tool_calls:  # 최종 답변
            run.answer, run.stopped = turn.text.strip(), "answer"
            self._add(steps, Step(n, "answer", policy, summary=self.answer_label, ms=_ms(t0)))
            messages.append({"role": "assistant", "content": run.answer})
            return "done"
        self._execute(run, n, turn)
        return "continue"

    def _execute(self, run: "_Run", n: int, turn: ToolTurn) -> list[dict]:
        """고른 도구들을 실행하고 관찰을 대화에 붙임 (판단과 실행을 나눠 v0.4 다른 구조들도 같은 실행기를 씀)."""
        messages, steps = run.messages, run.steps
        policy = run.policy
        observations = []
        messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls,
                         "raw": turn.raw})
        for i, call in enumerate(turn.tool_calls):
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            run.seen_calls[key] = run.seen_calls.get(key, 0) + 1
            if i >= MAX_CALLS_PER_STEP:
                obs = {"ok": False, "error": f"한 번에 도구는 {MAX_CALLS_PER_STEP}개까지만 실행합니다", "ms": 0}
            elif call.parse_error:
                obs = {"ok": False, "error": call.parse_error, "ms": 0}
            elif run.seen_calls[key] > 2:
                obs = {"ok": False, "error": "같은 도구를 같은 인자로 이미 여러 번 불렀습니다. 결과를 바탕으로 답하세요.",
                       "ms": 0}
            else:
                self._emit("tool_start", n=n, tool=call.name, args=call.arguments, agent=self.name)
                obs = self.registry.run(call.name, call.arguments)
                if self.registry.external and call.name == "search_listings" and obs["ok"]:
                    # MCP 서버가 검색 상태를 들고 있으므로, 여기서는 '직전 검색이 있었다' 는 것과 보여준 매물만 기억
                    self.ctx.last_search = dict(call.arguments) or {"query": run.question}
                    self.ctx.last_shown = [self.registry.resolve(x["id"]) or x["id"]
                                           for x in obs["result"].get("매물", [])]
                    self.ctx.last_regions, self.ctx.region_focus = [], ""
                if self.registry.external and call.name == "list_regions" and obs["ok"]:  # 지역 순위도 이쪽에서 기억
                    self.ctx.last_regions = list(obs["result"].get("지역별_매물_수") or {})
            self._add(steps, Step(n, "tool", policy, call.name, call.arguments, obs["ok"],
                                  summarize(call.name, obs), obs.get("ms", 0),
                                  thought=turn.text.strip()[:300] if i == 0 else ""))
            messages.append({"role": "tool", "call_id": call.id, "name": call.name,
                             "content": observation_text(obs), "is_error": not obs["ok"]})
            observations.append(obs)
        self._save(messages)  # 체크포인트: 여기서 끊겨도 이 도구 결과부터 이어서 진행
        return observations

    def _wrap_up(self, run: "_Run") -> None:
        """단계 수를 다 쓰면 도구 없이 답만 받고, 그것도 안 되면 규칙으로 정리."""
        messages = run.messages
        t0 = time.perf_counter()
        try:
            if not run.use_llm:
                raise LLMUnavailable("규칙 기반")
            turn = self._llm_call(self.llm_policy.llm.tool_step, self.deadline - (time.perf_counter() - run.started),
                                  self.llm_policy.prompt + run.extra + "\n이제 도구 없이 지금까지 결과로 답하세요.",
                                  list(messages), [])
            if turn.tool_calls or not turn.text.strip():
                raise LLMUnavailable("도구 없이 답하지 못함")
            run.answer, run.policy = turn.text.strip(), turn.provider or "llm"
        except LLMUnavailable:
            start = max(i for i, m in enumerate(messages) if m["role"] == "user")
            done = [(m["name"], _loads(m["content"])) for m in messages[start:] if m["role"] == "tool"]
            run.answer, run.policy = rule_answer(run.question, done), "rule"
        run.stopped = "max_steps"
        self._add(run.steps, Step(self.max_steps + 1, "guard", run.policy,
                                  summary=f"최대 {self.max_steps}단계 도달 → 정리해서 답변", ms=_ms(t0)))
        messages.append({"role": "assistant", "content": run.answer})

    def _llm_call(self, fn: Callable, budget: float, *args: Any) -> ToolTurn:
        """AI 호출을 남은 시간 예산만큼만 기다림 (멈춘 AI 때문에 손님이 계속 기다리지 않게)."""
        from concurrent.futures import TimeoutError as FutureTimeout

        if budget <= 0.05:
            raise LLMUnavailable("시간 예산을 다 씀")
        from ..observability.tracing import copy_context

        future = _llm_pool().submit(copy_context().run, fn, *args)  # 같은 trace 를 이어 감
        try:
            return future.result(timeout=budget)
        except FutureTimeout:
            from ..reliability.stats import STATS

            STATS.add("deadline_hits")
            raise LLMUnavailable(f"AI 응답이 시간 예산({self.deadline:g}초) 안에 오지 않음") from None

    def _check_output(self, question: str, answer: str, policy: str, done: list, steps: list[Step],
                      messages: list[dict]) -> tuple[str, str]:
        """v1.0 출력 가드: AI 가 쓴 답의 매물·가격·면적이 도구 결과에 있는지, 개인정보가 없는지 확인."""
        if not self.guards:
            return answer, policy
        from ..reliability.guardrails import Evidence, check_answer, contact_notice
        from ..reliability.stats import STATS

        n = steps[-1].n if steps else 0
        if answer and policy not in {"rule", ""}:
            history_obs = [(m["name"], _loads(m["content"])) for m in messages if m["role"] == "tool"]
            problems = check_answer(answer, Evidence.from_observations(history_obs + list(self.evidence())),
                                    self.agent)
            if problems:
                STATS.add("guard_fixes")
                fixed = self.fallback_answer(question, done)
                self._add(steps, Step(n, "guard", "rule",
                                      summary="답변 검증 실패 → 도구 결과로 다시 작성 (" + "; ".join(problems)[:200] + ")"))
                answer, policy = fixed, "rule"
        if answer and self.final_answerer and self._input and self._input.asks_contact:
            notice = contact_notice()
            if notice not in answer:
                answer = answer + "\n" + notice
        if messages and messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"):
            messages[-1]["content"] = answer
        return answer, policy

    def _cards(self, done: list[tuple[str, dict]], answer: str, seen_before: int) -> list[str]:
        """화면 카드로 보여줄 매물: 이번 질문의 마지막 비교(없으면 검색) 결과 순서대로 + 답변에서 따로 말한 매물."""
        base: list[str] = []
        for name, obs in reversed(done):
            if obs.get("ok") and name in {"compare_listings", "search_listings"}:
                base = [x["id"] for x in obs["result"].get("매물", [])]
                break
        if not base:
            base = [obs["result"]["id"] for name, obs in done
                    if obs.get("ok") and name in {"get_listing", "check_land_use"} and "id" in obs.get("result", {})]
        if answer.strip() == NOT_FOUND_MESSAGE:
            return []
        base = [self.registry.resolve(i) or i for i in base]
        extra = [i for i in self._mentioned(answer, seen_before) if i not in base]
        return list(dict.fromkeys(base + extra))[:6]

    def _mentioned(self, answer: str, seen_before: int = 0) -> list[str]:
        """답변에 나온 매물 (id 또는 매물번호) - 없으면 이번 질문에서 도구로 본 매물."""
        ids = []
        from ..agent import public_office_no

        for l in self.agent.listings():
            for key in {l.listing_id, l.office_no, public_office_no(l.office_no)} - {""}:
                if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(key)}(?![A-Za-z0-9_-])", answer):
                    ids.append((answer.find(key), l.listing_id))
                    break
        ordered = [lid for _, lid in sorted(ids)]
        return ordered[:6] or self.ctx.seen[seen_before:][:3]

    def _log(self, r: AgentResult) -> None:
        if not self.log_dir:
            return
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            row = {"run_id": r.run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": r.question[:300],
                   "policy": r.policy, "stopped": r.stopped, "steps": len(r.steps), "tool_calls": r.tool_calls,
                   "tool_errors": r.tool_errors, "total_ms": r.total_ms,
                   "tools": [s.tool for s in r.steps if s.kind == "tool"]}
            with open(Path(self.log_dir) / "agent_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("agent 실행 기록 실패: %s", e)


def summarize(name: str, obs: dict[str, Any]) -> str:
    if not obs.get("ok"):
        return "오류: " + str(obs.get("error", ""))[:160]
    r = obs.get("result", {})
    if name == "search_listings":
        return f"{r.get('조건에_맞는_매물_수', 0)}건 찾음 ({r.get('이해한_조건', '')}) → {len(r.get('매물', []))}건 반환"
    if name == "compare_listings":
        return f"{len(r.get('매물', []))}건 비교 · 가장 싼 매물 {r.get('가장_싼_매물')}"
    if name == "check_land_use":
        return f"{r.get('용도')}: " + ("판단 규칙 없음" if "가능" not in r else ("가능" if r["가능"] else "어려움")
                                       + f" ({r.get('근거', '')})")
    if name == "get_listing":
        return f"{r.get('매물번호') or r.get('id')} 상세 · {r.get('가격', '')} · {r.get('면적_평', '')}평"
    if name == "record_feedback":
        return f"{r.get('id')} {r.get('기록')} 기록"
    if name == "list_regions":
        return f"지역 {len(r.get('지역별_매물_수', {}))}곳"
    if name.startswith("delegate_"):  # v0.7 멀티 Agent: 전문 Agent 의 보고
        first = (r.get("보고") or "").split("\n")[0]
        return f"{r.get('agent', '')} 보고 · 매물 {len(r.get('매물', []))}건 · {first[:90]}"
    return "완료"


def trim_history(messages: list[dict]) -> list[dict]:
    """오래된 대화는 버리되, 도구 호출/결과 짝이 끊기지 않게 '질문' 메시지부터 시작."""
    if len(messages) <= MAX_HISTORY:
        return messages
    tail = messages[-MAX_HISTORY:]
    first_user = next((i for i, m in enumerate(tail) if m["role"] == "user"), len(tail))
    return tail[first_user:]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def step_dict(s: Step) -> dict:
    return asdict(s)
