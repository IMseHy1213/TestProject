"""Tool-use Agent 가 쓰는 도구 (v0.2).

각 도구 = 이름 + 설명 + 인자 JSON schema + 실행 함수.
- 인자는 실행 전에 schema 로 검사하고, 틀리면 실행하지 않고 오류를 '관찰(observation)'로 돌려줍니다.
  (모델이 오류를 보고 인자를 고쳐 다시 부를 수 있게)
- 도구 결과에는 소유자 연락처와 정확한 지번을 넣지 않습니다. (LLM 에게 보내지 않음)
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..land_use import RULES as USE_RULES
from ..land_use import check_use
from ..models import PYEONG_M2, CustomerRequirement, LandListing, ScoredListing, format_price
from ..recommender import NOT_FOUND_MESSAGE

MAX_OBSERVATION_CHARS = 6000
PURPOSES = sorted(USE_RULES)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class AgentContext:
    """한 상담(대화) 동안 도구들이 함께 쓰는 상태."""

    rejected: dict[str, str] = field(default_factory=dict)   # 손님이 싫다고 한 매물 -> 이유
    liked: list[str] = field(default_factory=list)
    seen: list[str] = field(default_factory=list)            # 도구 결과로 보여준 매물 (순서대로)
    customer: str = ""
    last_search: dict[str, Any] = field(default_factory=dict)  # 직전 검색 조건 ('그 중에 넓은 건?' 같은 후속 질문용)
    last_shown: list[str] = field(default_factory=list)  # 직전에 손님에게 보여준 매물 ('그 중' = 이 안에서 고름)
    last_search_from_memory: bool = False  # last_search 가 장기 기억(지난 상담)에서 온 것인지
    run_searches: list[dict] = field(default_factory=list)  # 이번 질문에서 새로 찾은 조건 (장기 기억에 저장)
    pref_key: str = ""     # v1.2 온라인 학습: 이 방문자의 학습 모델 이름 (있으면 검색 순서에 반영)
    variant: str = ""      # v1.2 '자동' 상담 방식이 이 대화에 고른 방식 (single / multi)
    last_regions: list[str] = field(default_factory=list)  # 직전에 알려준 '매물 많은 지역' 순위 ('두 번째로 많은 지역')
    region_focus: str = ""  # 직전 답에서 가리킨 지역 ('제일 적은 곳은 여주시' → '그 지역 매물 보여줘' = 여주시)


# ---------------------------------------------------------------------- 인자 검사
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """간단한 JSON schema 검사: required / type / enum / minimum·maximum / minItems·maxItems / 모르는 인자.
    반환: (정리된 인자, 오류 목록)."""
    props = schema.get("properties", {})
    errors: list[str] = []
    clean: dict[str, Any] = {}
    for key in schema.get("required", []):
        if args.get(key) in (None, "", []):
            errors.append(f"'{key}' 는 꼭 필요합니다")
    for key, value in args.items():
        if key not in props:
            errors.append(f"모르는 인자 '{key}' (쓸 수 있는 인자: {', '.join(props)})")
            continue
        if value is None:
            continue
        spec = props[key]
        t = spec.get("type")
        if t == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if t == "number" and isinstance(value, str):
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                pass
        if t == "array" and isinstance(value, str):  # 'a, b' 처럼 문자열로 준 목록
            value = [v.strip() for v in value.split(",") if v.strip()]
        expected = _TYPES.get(t)
        if expected and (not isinstance(value, expected) or (t in {"integer", "number"} and isinstance(value, bool))):
            errors.append(f"'{key}' 는 {t} 이어야 합니다 (받은 값: {value!r})")
            continue
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"'{key}' 는 {spec['enum']} 중 하나여야 합니다 (받은 값: {value!r})")
            continue
        if t in {"integer", "number"}:
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"'{key}' 는 {spec['minimum']} 이상이어야 합니다")
                continue
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"'{key}' 는 {spec['maximum']} 이하여야 합니다")
                continue
        if t == "array":
            item_t = _TYPES.get(spec.get("items", {}).get("type", ""))
            if item_t and not all(isinstance(v, item_t) for v in value):
                errors.append(f"'{key}' 의 항목은 {spec['items']['type']} 이어야 합니다")
                continue
            if len(value) < spec.get("minItems", 0) or len(value) > spec.get("maxItems", 10 ** 6):
                errors.append(f"'{key}' 는 {spec.get('minItems', 0)}~{spec.get('maxItems', '')}개여야 합니다")
                continue
        clean[key] = value
    return clean, errors


# ---------------------------------------------------------------------- 매물 -> 도구 결과 (공개 정보만)
def tool_id(l: LandListing) -> str:
    """AI 에게 보여줄 매물 id: 매물번호 뒤 괄호 속 이름·메모를 뺀 것 ('S.24250409(이재순님)' -> 'S.24250409')."""
    from ..agent import public_office_no

    return public_office_no(l.listing_id) or l.listing_id


def display_name(l: LandListing) -> str:
    """손님에게 매물을 가리킬 이름: 매물번호, 없으면 '음성군 물류부지 15,005평 매물'
    (매물 종류 = 매물 폴더 이름 괄호 안의 용도, 없으면 '토지'. 프로그램 내부 번호 F-… 는 쓰지 않음)."""
    from ..agent import public_office_no, public_region

    no = public_office_no(l.office_no)
    if no:
        return no
    region = (public_region(l.address).split() or [""])[-1]
    kind = l.use_tag or "토지"
    return " ".join(x for x in (region, kind, f"{l.area_pyeong:,.0f}평 매물") if x)


def brief(l: LandListing, s: Optional[ScoredListing] = None) -> dict[str, Any]:
    from ..agent import mask_address, public_office_no

    d = {
        "id": tool_id(l),
        "매물번호": public_office_no(l.office_no) or None,
        "표시이름": display_name(l),
        "주소": mask_address(l.address),
        "지목": l.jimok or None,
        "용도지역": l.zoning or None,
        "면적_평": round(l.area_pyeong),
        "가격": format_price(l.price_manwon),
        "가격_만원": l.price_manwon,
        "평당가_만원": round(l.price_per_pyeong),
        "매물특징": (l.features or "")[:200] or None,
    }
    if s is not None:
        d["점수"] = round(s.score, 1)
        d["장점"] = s.reasons[:3]
        if s.penalties:
            d["확인필요"] = s.penalties[:2]
    return {k: v for k, v in d.items() if v not in (None, [], "")}


def condition_summary(req: CustomerRequirement) -> str:
    """손님에게 보여줄 조건 요약: '음성군 · 창고 용도 · 3억 이하 · 도로 접한 땅'"""
    parts = []
    if req.regions:
        parts.append(", ".join(req.regions))
    if req.excluded_regions:
        parts.append(", ".join(req.excluded_regions) + " 제외")
    if req.purpose:
        parts.append(f"{req.purpose} 용도")
    if req.jimok:
        parts.append("지목 " + "·".join(req.jimok))
    if req.budget_min and req.budget_max:
        parts.append(f"{format_price(req.budget_min)}~{format_price(req.budget_max)}")
    elif req.budget_max:
        parts.append(f"{format_price(req.budget_max)} 이하")
    elif req.budget_min:
        parts.append(f"{format_price(req.budget_min)} 이상")
    if req.area_min_m2 and req.area_max_m2:
        parts.append(f"{req.area_min_m2 / PYEONG_M2:,.0f}~{req.area_max_m2 / PYEONG_M2:,.0f}평")
    elif req.area_min_m2:
        parts.append(f"{req.area_min_m2 / PYEONG_M2:,.0f}평 이상")
    elif req.area_max_m2:
        parts.append(f"{req.area_max_m2 / PYEONG_M2:,.0f}평 이하")
    if req.keywords:
        parts.append(", ".join(req.keywords[:2]))
    return " · ".join(parts) or "전체"


def shown_region(l: LandListing) -> str:
    """매물이 있는 시/군/구 ('충청북도 음성군 ***' -> '음성군').
    주소가 '충북 음성 대소면 …', '충청북도음성군대소면 …' 처럼 적혀 있어도 시/군을 찾고,
    주소에 시/군이 아예 없으면 매물 폴더 이름(예: 음성군/…)에서 찾음."""
    from ..agent import public_region
    from ..regions import find_regions, sido_of

    parts = public_region(l.address).split()
    if parts and not sido_of(parts[-1]):
        return parts[-1]
    extra = getattr(l, "extra", {}) or {}
    for text in (l.address, str(extra.get("_매물폴더", "")), str(getattr(l, "source_file", "") or "")):
        names = [r for r in find_regions(text) if not sido_of(r)]
        if names:
            return names[0]
    return parts[-1] if parts else ""


def _without_regions(query: str, agent) -> str:
    """직전 검색 문장에서 지역 이름을 지움 ('음성군 공장 땅' -> '공장 땅'): 지역만 바꿔 다시 찾을 때."""
    import re as _re

    from ..regions import find_regions

    names = set(agent.known_regions()) | set(find_regions(query))
    for name in sorted(names, key=len, reverse=True):
        stem = _re.sub(r"(시|군|구)$", "", name)
        query = query.replace(name, " ")
        if len(stem) >= 2 and name[-1:] in "시군구":
            query = _re.sub(rf"{_re.escape(stem)}(?=\s|에|의|쪽|$)", " ", query)
    return " ".join(query.split())


def requirement_from_args(agent, given: dict[str, Any]) -> CustomerRequirement:
    """search_listings 인자 (query + 따로 준 조건) -> 손님 조건. 실시간 매칭 워커도 같은 함수를 씀."""
    from ..regions import canonical_region
    from ..requirement_parser import RequirementParser

    query, regions, jimok = given.get("query", ""), given.get("regions"), given.get("jimok")
    purpose = given.get("purpose", "")
    budget_min_manwon, budget_max_manwon = given.get("budget_min_manwon"), given.get("budget_max_manwon")
    area_min_pyeong, area_max_pyeong = given.get("area_min_pyeong"), given.get("area_max_pyeong")
    keywords = given.get("keywords")

    # 문장 해석은 규칙 기반으로 (도구 안에서 LLM 을 또 부르지 않음 - 판단은 바깥의 Agent 가)
    parser = RequirementParser(None, agent.known_regions(), [l.address for l in agent.listings()])
    req = parser.parse(query) if query.strip() else CustomerRequirement()
    if regions:
        req.regions = [canonical_region(r) for r in regions]
    if jimok:
        req.jimok = list(jimok)
    if purpose:
        req.purpose = purpose
    if budget_min_manwon is not None:
        req.budget_min = budget_min_manwon
    if budget_max_manwon is not None:
        req.budget_max = budget_max_manwon
    if area_min_pyeong is not None:
        req.area_min_m2 = area_min_pyeong * PYEONG_M2
    if area_max_pyeong is not None:
        req.area_max_m2 = area_max_pyeong * PYEONG_M2
    if keywords:
        req.keywords = list(dict.fromkeys(req.keywords + list(keywords)))
    for r in given.get("exclude_regions") or []:
        c = canonical_region(r)
        if c not in req.excluded_regions:
            req.excluded_regions.append(c)
    if req.excluded_regions:
        from ..regions import same_place

        req.regions = [r for r in req.regions
                       if r not in req.excluded_regions and not any(same_place(r, x) for x in req.excluded_regions)]
        req.excluded_keywords = [k for k in req.excluded_keywords if canonical_region(k) not in req.excluded_regions]
    return req


class ToolRegistry:
    """도구 목록 + 실행 (인자 검사, 오류를 관찰로 변환, 실행 시간 측정)."""

    def __init__(self, agent, ctx: Optional[AgentContext] = None, lock: Optional[threading.RLock] = None,
                 memory=None, owner: str = "", tool_source=None, only: Optional[set[str]] = None):
        """tool_source: MCP 서버 등 바깥에서 받아 온 도구 (있으면 내장 도구 대신 사용).
        only: 이 이름의 도구만 쓰기 (멀티 Agent 에서 역할별로 도구를 나눌 때)."""
        self.agent = agent
        self.ctx = ctx or AgentContext()
        self.lock = lock
        self.memory = memory  # LongTermMemory (없으면 recall_memory 는 '기록 없음')
        self.owner = owner
        self.external = tool_source is not None  # 도구가 MCP 서버 쪽에서 실행됨 (검색 상태도 서버에)
        self.tools: dict[str, Tool] = {}
        for tool in (tool_source.tools() if tool_source is not None else build_tools(self)):
            if only is None or tool.name in only:
                self.tools[tool.name] = tool

    def resolve(self, some_id: str) -> Optional[str]:
        """도구가 쓴 id(공개용) -> 실제 매물 id."""
        for l in self.agent.listings() + list(getattr(self.agent, "hidden_listings", [])):
            if some_id in {l.listing_id, tool_id(l)}:
                return l.listing_id
        return None

    def _call(self, tool: "Tool", args: dict[str, Any]) -> Any:
        """도구 실행. v1.0: DB 잠김 같은 일시 오류는 잠깐 쉬었다가 2번까지 다시 (그래도 안 되면 오류 관찰)."""
        for attempt in range(3):
            try:
                with self.lock or nullcontext():  # 매물 목록·DB 는 여러 접속자가 함께 씀
                    return tool.fn(**args)
            except Exception as e:
                if attempt == 2 or not _transient(e):
                    raise
                from ..reliability.stats import STATS

                STATS.add("tool_retries")
                time.sleep(0.05 * 2 ** attempt)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """도구 실행 -> {'ok': bool, 'result' 또는 'error', 'ms'}. v1.1: 도구 호출 하나 = span 하나 + 지표."""
        from ..observability.metrics import TOOL_CALLS, TOOL_LATENCY
        from ..observability.tracing import TRACER

        with TRACER.span(f"tool.{name}", tool=name, remote=self.external or None,
                         args=json.dumps(args or {}, ensure_ascii=False, default=str)[:200]) as sp:
            out = self._run(name, args)
            sp.set(ok=out["ok"])
            if out["ok"]:
                r = out.get("result")
                if isinstance(r, dict) and isinstance(r.get("매물"), list):
                    sp.set(results=len(r["매물"]))
            else:
                sp.error(out.get("error", ""))
        TOOL_CALLS.inc(tool=name if name in self.tools else "unknown", ok=str(out["ok"]).lower())
        TOOL_LATENCY.observe(out["ms"] / 1000, tool=name if name in self.tools else "unknown")
        return out

    def _run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            out = {"ok": False, "error": f"'{name}' 라는 도구는 없습니다. 쓸 수 있는 도구: {', '.join(self.tools)}"}
        else:
            clean, errors = validate_arguments(tool.parameters, args or {})
            if errors:
                out = {"ok": False, "error": "인자 오류: " + "; ".join(errors)}
            else:
                try:
                    result = self._call(tool, clean)
                    out = {"ok": True, "result": result}
                except ToolError as e:
                    out = {"ok": False, "error": str(e)}
                except Exception as e:  # 도구 하나가 실패해도 대화는 계속
                    out = {"ok": False, "error": f"도구 실행 중 오류: {type(e).__name__}: {e}"}
        out["ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out


def _transient(e: Exception) -> bool:
    """잠깐 기다리면 풀리는 오류 (SQLite 'database is locked' 등)."""
    import sqlite3

    return isinstance(e, sqlite3.OperationalError) and ("locked" in str(e) or "busy" in str(e))


class ToolError(Exception):
    """모델에게 그대로 보여줄 도구 오류 (예: 없는 매물 번호)."""


def observation_text(obs: dict[str, Any]) -> str:
    """모델에게 보낼 관찰 문자열 (너무 길면 자름)."""
    body = {k: v for k, v in obs.items() if k != "ms"}
    text = json.dumps(body, ensure_ascii=False, default=str)
    if len(text) > MAX_OBSERVATION_CHARS:
        text = text[:MAX_OBSERVATION_CHARS] + '…"(결과가 길어 잘림)"}'
    return text


# ---------------------------------------------------------------------- 도구 정의
def build_tools(reg: ToolRegistry) -> list[Tool]:
    agent, ctx = reg.agent, reg.ctx

    def find(listing_id: str) -> LandListing:
        from ..agent import public_office_no

        key = listing_id.strip()
        for l in agent.listings():
            if key in {l.listing_id, tool_id(l)} or (l.office_no and key in {l.office_no, public_office_no(l.office_no)}):
                return l
        raise ToolError(f"매물 '{listing_id}' 을(를) 찾을 수 없습니다. search_listings 결과의 id 를 쓰세요.")

    def search_listings(query: str = "", regions: Optional[list[str]] = None, jimok: Optional[list[str]] = None,
                        purpose: str = "", budget_min_manwon: Optional[int] = None,
                        budget_max_manwon: Optional[int] = None, area_min_pyeong: Optional[float] = None,
                        area_max_pyeong: Optional[float] = None, keywords: Optional[list[str]] = None,
                        sort: str = "score", limit: int = 5, use_previous_conditions: bool = False,
                        exclude_regions: Optional[list[str]] = None, other_regions: bool = False,
                        recheck: bool = False) -> dict[str, Any]:
        # '정말 없어?' / '확실해?' / '다시 찾아봐' -> 방금 검색을 같은 조건(뺀 지역 포함) 그대로, 전체 매물에서 다시
        if recheck and ctx.last_search:
            return _recheck(limit=max(limit, 5), sort=sort)

        given = {k: v for k, v in dict(query=query, regions=regions, jimok=jimok, purpose=purpose,
                                         budget_min_manwon=budget_min_manwon, budget_max_manwon=budget_max_manwon,
                                         area_min_pyeong=area_min_pyeong, area_max_pyeong=area_max_pyeong,
                                         keywords=keywords, exclude_regions=exclude_regions).items()
                 if v not in (None, "", [])}
        follow_up = (use_previous_conditions or other_regions) and bool(ctx.last_search)
        from_memory = follow_up and ctx.last_search_from_memory
        # '다른 지역에도 있어?' -> 직전에 보여준 매물이 있던 지역(과 직전 검색 지역)을 빼고, 나머지 조건은 그대로
        moved: list[str] = []
        if other_regions:
            from ..regions import sido_of

            # 앞에서 손님이 말한 지역: 따로 준 지역 + 문장 속 지역 ('음성군에 창고 지을 땅 있어?' → 음성군)
            prev = requirement_from_args(agent, ctx.last_search) if ctx.last_search else None
            moved = list(ctx.last_search.get("regions") or []) + list(prev.regions if prev else [])
            # 보여준 매물이 있던 시/군 (도 단위로는 빼지 않음 - 주소에서 시/군을 못 읽었을 때 도 전체가 빠지지 않게)
            moved += [r for r in (shown_region(l) for l in agent.listings() if l.listing_id in ctx.last_shown)
                      if r and not sido_of(r)]
            moved = [r for r in dict.fromkeys(moved) if r]
        # 이번 질문에서 지역을 새로 말했는지 (따로 준 인자 또는 문장: '음성군 말고', '진천군에도 있어?')
        said = requirement_from_args(agent, {"query": given["query"]}) if given.get("query") else None
        named_region = bool(given.get("regions") or given.get("exclude_regions")
                            or (said and (said.regions or said.excluded_regions)))
        region_change = other_regions or named_region
        if follow_up:  # '그 중에 넓은 건?' -> 직전 검색 조건 그대로 + 새 정렬/개수 (새로 준 조건은 덧붙임)
            merged = dict(ctx.last_search)
            if region_change:  # 지역을 바꾸는 질문이면 직전 지역 조건(찾을 지역·뺄 지역)은 버리고 새로 말한 것만
                merged.pop("regions", None)
                merged.pop("exclude_regions", None)
                merged["query"] = _without_regions(merged.get("query", ""), agent)
            for k, v in given.items():
                if k == "query":
                    merged[k] = f"{merged.get('query', '')} {v}".strip()
                elif k == "exclude_regions":
                    merged[k] = list(dict.fromkeys(list(merged.get(k) or []) + list(v)))
                else:
                    merged[k] = v
            given = merged
        if moved:
            given["exclude_regions"] = list(dict.fromkeys(list(given.get("exclude_regions") or []) + moved))
        ctx.last_search = dict(given)
        ctx.last_search_from_memory = False
        if not (ctx.region_focus and list(given.get("regions") or []) == [ctx.region_focus]):
            # 새 검색을 하면 '그 지역' 은 이제 검색한 지역을 가리킴 (방금 답한 지역 안에서 찾은 거면 '그 중에' 를 계속 그 지역으로)
            ctx.last_regions, ctx.region_focus = [], ""
        req = requirement_from_args(agent, given)
        req.customer_name = ctx.customer
        session = agent.recommend(req, pref_key=ctx.pref_key or None)  # v1.2: 이 방문자에게서 배운 중요도로 점수
        ranked = [s for s in session.ranked if s.listing.listing_id not in ctx.rejected]
        pool = list(ctx.last_shown) if follow_up and not region_change else []  # 지역을 바꾸면 전체에서 다시
        if pool:  # '그 중에' -> 직전에 보여준 매물 안에서만
            ranked = [s for s in ranked if s.listing.listing_id in pool]
        if sort == "price_asc":
            ranked.sort(key=lambda s: s.listing.price_manwon)
        elif sort == "price_desc":
            ranked.sort(key=lambda s: -s.listing.price_manwon)
        elif sort == "area_desc":
            ranked.sort(key=lambda s: -s.listing.area_m2)
        elif sort == "price_per_pyeong_asc":
            ranked.sort(key=lambda s: s.listing.price_per_pyeong)
        shown = ranked[:limit]
        for s in shown:
            if s.listing.listing_id not in ctx.seen:
                ctx.seen.append(s.listing.listing_id)
        if not pool:  # 새 검색이면 '보여준 매물' 갱신 (후속 질문은 같은 후보 안에서 계속 고르도록 유지)
            ctx.last_shown = [s.listing.listing_id for s in shown]
        if not follow_up or from_memory:
            ctx.run_searches.append({"조건": condition_summary(req), "args": dict(given)})
        order = {"score": "추천순", "price_asc": "가격 낮은 순", "price_desc": "가격 높은 순", "area_desc": "넓은 순",
                 "price_per_pyeong_asc": "평당가 낮은 순"}[sort]
        out: dict[str, Any] = {"이해한_조건": req.describe(), "조건_요약": condition_summary(req), "정렬": order,
                               "앞_조건_이어서": follow_up, "앞서_보여준_매물_수": len(pool) or None,
                               "제외_지역": req.excluded_regions or None,
                               "앞_조건_출처": "지난 상담" if from_memory else None,
                               "조건에_맞는_매물_수": len(ranked),
                               "매물": [brief(s.listing, s) for s in shown]}
        if not ranked:
            out["안내"] = NOT_FOUND_MESSAGE
        if req.notes:
            out["참고"] = req.notes
        if ctx.rejected:
            out["제외한_매물"] = [tool_id(x) for x in agent.listings() if x.listing_id in ctx.rejected]
        if sort == "score" and ranked and not pool:  # v1.2: 추천순이면 이 손님에게서 배운 중요도가 반영됨
            from ..online.learning import learned_note

            note = learned_note(agent, ctx.pref_key)
            if note:
                out["학습_반영"] = note
        return out

    def _recheck(limit: int, sort: str) -> dict[str, Any]:
        from collections import Counter as _Counter

        given = dict(ctx.last_search)
        req = requirement_from_args(agent, given)
        req.customer_name = ctx.customer
        ranked = [s for s in agent.recommend(req).ranked if s.listing.listing_id not in ctx.rejected]
        if sort == "price_asc":
            ranked.sort(key=lambda s: s.listing.price_manwon)
        elif sort == "area_desc":
            ranked.sort(key=lambda s: -s.listing.area_m2)
        shown = ranked[:limit]
        ctx.last_shown = [s.listing.listing_id for s in shown]
        for s in shown:
            if s.listing.listing_id not in ctx.seen:
                ctx.seen.append(s.listing.listing_id)
        by_region = _Counter(shown_region(s.listing) for s in ranked)
        out: dict[str, Any] = {
            "다시_확인": True, "이해한_조건": req.describe(), "조건_요약": condition_summary(req),
            "제외_지역": req.excluded_regions or None, "정렬": "추천순" if sort == "score" else sort,
            "조건에_맞는_매물_수": len(ranked), "지역별_매물_수": dict(by_region.most_common()),
            "전체_보유_매물_수": len(agent.listings()), "매물": [brief(s.listing, s) for s in shown]}
        if not ranked:  # 정말 없으면 어떤 조건 때문에 없는지 알려 줌
            out["안내"] = NOT_FOUND_MESSAGE
            hints = {}
            if req.need_road:
                relaxed = requirement_from_args(agent, given)
                relaxed.need_road = False
                hints["도로_조건을_빼면"] = len(agent.recommend(relaxed).ranked)
            if req.purpose:
                relaxed = requirement_from_args(agent, given)
                relaxed.purpose, relaxed.need_road = "", False
                hints["용도_조건을_빼면"] = len(agent.recommend(relaxed).ranked)
            out["조건을_줄이면"] = hints
        return out

    def get_listing(listing_id: str) -> dict[str, Any]:
        l = find(listing_id)
        d = brief(l)
        d.update({"도로": l.road_access or None, "상태": l.status,
                  "등록일": l.registered_date.isoformat() if l.registered_date else None,
                  "특이사항": (l.description or "")[:400] or None, "면적_㎡": round(l.area_m2)})
        if l.listing_id not in ctx.seen:
            ctx.seen.append(l.listing_id)
        return {k: v for k, v in d.items() if v not in (None, "")}

    def check_land_use(listing_id: str, purpose: str) -> dict[str, Any]:
        l = find(listing_id)
        res = check_use(l, purpose)
        if res is None:
            return {"id": tool_id(l), "용도": purpose, "판단": "규칙 없음"}
        out = {"id": tool_id(l), "용도": purpose, "가능": res.ok, "근거": res.reason}
        if res.caution:
            out["확인필요"] = res.caution
        out["주의"] = "프로그램의 1차 판단입니다. 실제 건축 가능 여부는 토지이용계획확인원·관할 관청 확인 필요."
        return out

    def compare_listings(listing_ids: list[str]) -> dict[str, Any]:
        items = [find(i) for i in dict.fromkeys(listing_ids)]
        if len(items) < 2:
            raise ToolError("비교하려면 서로 다른 매물이 2개 이상 필요합니다.")
        rows = [brief(l) for l in items]
        for l in items:
            if l.listing_id not in ctx.seen:
                ctx.seen.append(l.listing_id)
        ctx.last_shown = [l.listing_id for l in items]  # 비교한 매물이 손님 화면에 나옴
        def pick(key):
            return tool_id(min(items, key=key))
        return {
            "매물": rows,
            "가장_싼_매물": pick(lambda l: l.price_manwon),
            "가장_넓은_매물": tool_id(max(items, key=lambda l: l.area_m2)),
            "평당가가_가장_낮은_매물": pick(lambda l: l.price_per_pyeong),
            "가격차_만원": max(l.price_manwon for l in items) - min(l.price_manwon for l in items),
        }

    def record_feedback(listing_id: str, action: str, reason: str = "") -> dict[str, Any]:
        l = find(listing_id)
        agent.storage.add_feedback(ctx.customer, l.listing_id, action, reason)
        if action == "reject":
            ctx.rejected[l.listing_id] = reason
            note = "이 상담에서는 이 매물을 다시 추천하지 않습니다."
        else:
            if l.listing_id not in ctx.liked:
                ctx.liked.append(l.listing_id)
            note = "관심 매물로 기록했습니다."
        from ..online.learning import learn_from_feedback

        learned = learn_from_feedback(agent, ctx, l, action, reason)  # v1.2 온라인 학습: 다음 검색부터 반영
        return {"id": tool_id(l), "기록": action, "이유": reason or None, "안내": note, "배운_점": learned or None}

    def recall_memory() -> dict[str, Any]:
        mem = reg.memory.load(reg.owner) if reg.memory and reg.owner else None
        if not mem or not (mem["searches"] or mem["liked"] or mem["rejected"]):
            return {"기억": "이 손님과의 지난 상담 기록이 없습니다."}
        by_id = {l.listing_id: l for l in agent.listings() + list(getattr(agent, "hidden_listings", []))}
        return {
            "방문_횟수": mem.get("visits", 0), "마지막_방문": mem.get("last_seen"),
            "최근_조건": [s["조건"] for s in reversed(mem["searches"][-5:])],
            "관심_매물": [brief(by_id[i]) for i in list(mem["liked"])[-5:] if i in by_id],
            "싫다고_한_매물": [{"이름": display_name(by_id[i]), "이유": (v or {}).get("reason") or None}
                          for i, v in list(mem["rejected"].items())[-5:] if i in by_id],
            "안내": "싫다고 한 매물은 검색에서 자동으로 빠집니다. 지난 조건으로 다시 찾으려면 "
                  "search_listings 에 use_previous_conditions=true.",
        }

    def list_regions(limit: int = 40) -> dict[str, Any]:
        """지금 살 수 있는 매물(계약완료·보류 제외)을 시/군 단위로 셈. 순위는 대화 상태에 기억 ('그 지역 매물 보여줘')."""
        available = [l for l in agent.listings() if l.status not in {"계약완료", "보류"}]
        by_city: Counter = Counter(shown_region(l) for l in available if shown_region(l))
        by_town: Counter = Counter()
        for l in available:
            for tok in set(l.address.split()[2:3]):
                if len(tok) >= 2 and tok[-1:] in "읍면동":
                    by_town[tok] += 1
        ranked = [r for r, _ in by_city.most_common()]
        ctx.last_regions = ranked
        return {"지역별_매물_수": dict(by_city.most_common(limit)), "전체_매물_수": len(available),
                "가장_많은_지역": ranked[0] if ranked else None, "가장_적은_지역": ranked[-1] if ranked else None,
                "읍면별_매물_수": dict(by_town.most_common(10)),
                "안내": "손님이 지역별 매물 수만 물었으면 숫자로만 답하고 매물 목록은 보여주지 마세요. "
                        "그 지역 매물을 보고 싶다고 하면 search_listings(regions=[그 지역])."}

    string_list = {"type": "array", "items": {"type": "string"}}
    return [
        Tool("search_listings",
             "사무실이 보유한 토지 매물을 조건으로 찾아 추천 순위대로 돌려줍니다. 손님 문장을 query 에 그대로 넣거나 "
             "조건을 따로 지정할 수 있습니다(둘 다 쓰면 따로 지정한 조건이 우선). 가격 단위는 만원(1억=10000).",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "손님이 말한 조건 문장 (예: '음성군 창고 지을 땅 3억 이하')"},
                 "regions": {**string_list, "description": "지역 (예: ['음성군'], ['대소읍'])"},
                 "jimok": {**string_list, "description": "지목 (전, 답, 임야, 대, 공장용지, 창고용지 …)"},
                 "purpose": {"type": "string", "enum": PURPOSES, "description": "지을 건물/용도"},
                 "budget_min_manwon": {"type": "integer", "minimum": 0},
                 "budget_max_manwon": {"type": "integer", "minimum": 0},
                 "area_min_pyeong": {"type": "number", "minimum": 0},
                 "area_max_pyeong": {"type": "number", "minimum": 0},
                 "keywords": {**string_list, "description": "원하는 특징 (남향, 계곡, IC 인근 …)"},
                 "sort": {"type": "string", "enum": ["score", "price_asc", "price_desc", "area_desc",
                                                     "price_per_pyeong_asc"], "description": "정렬 (기본 추천순)"},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "최대 개수 (기본 5)"},
                 "exclude_regions": {**string_list, "description":
                     "빼고 찾을 지역 모두 ('음성군 말고' -> ['음성군'], '진천군이랑 음성군 말고' -> ['진천군', '음성군'])"},
                 "other_regions": {"type": "boolean", "description":
                     "손님이 '다른 지역에도 있어?'처럼 지역만 바꿔 달라고 하면 true: 직전 조건(용도·예산 등)은 그대로, "
                     "직전에 보여준 매물이 있던 지역은 빼고 전체 매물에서 다시 찾음"},
                 "recheck": {"type": "boolean", "description":
                     "손님이 '정말 없어?', '진짜 다른 지역에는 없어?', '확실해?', '다시 찾아봐'처럼 방금 검색 결과를 "
                     "다시 확인해 달라고 하면 true: 직전 조건(뺀 지역 포함)을 그대로 전체 매물에서 다시 찾고 지역별 개수를 돌려줌"},
                 "use_previous_conditions": {"type": "boolean", "description":
                     "손님이 '그 중에', '거기서'처럼 직전 결과를 가리키면 true: 직전에 보여준 매물들 안에서만 "
                     "sort 기준으로 다시 골라 limit(하나만 물으면 1)만큼 돌려줌"},
             }}, search_listings),
        Tool("get_listing", "매물 하나의 자세한 정보 (도로, 상태, 등록일, 특이사항 포함).",
             {"type": "object", "properties": {"listing_id": {"type": "string", "description": "매물 id 또는 매물번호"}},
              "required": ["listing_id"]}, get_listing),
        Tool("check_land_use", "매물에 특정 용도(창고, 공장, 전원주택 등)의 건물을 지을 수 있는지 지목·용도지역으로 1차 판단합니다.",
             {"type": "object", "properties": {"listing_id": {"type": "string"},
                                               "purpose": {"type": "string", "enum": PURPOSES}},
              "required": ["listing_id", "purpose"]}, check_land_use),
        Tool("compare_listings", "매물 2~5개를 가격·면적·평당가로 비교합니다.",
             {"type": "object", "properties": {"listing_ids": {**string_list, "minItems": 2, "maxItems": 5}},
              "required": ["listing_ids"]}, compare_listings),
        Tool("record_feedback", "손님이 매물을 마음에 들어하거나(like) 싫다고(reject) 하면 기록합니다. "
             "reject 한 매물은 이 상담에서 다시 검색되지 않습니다.",
             {"type": "object", "properties": {"listing_id": {"type": "string"},
                                               "action": {"type": "string", "enum": ["like", "reject"]},
                                               "reason": {"type": "string", "description": "손님이 말한 이유"}},
              "required": ["listing_id", "action"]}, record_feedback),
        Tool("recall_memory", "이 손님과의 지난 상담 기억(최근에 찾은 조건, 관심 매물, 싫다고 한 매물)을 불러옵니다. "
             "손님이 '지난번', '전에 찾던' 처럼 예전 상담을 말하면 먼저 부르세요.",
             {"type": "object", "properties": {}}, recall_memory),
        Tool("list_regions", "지금 살 수 있는 매물의 지역별(시·군) 매물 수와 가장 많은/적은 지역. '어느 지역 매물이 제일 많아요?', "
             "'지역별로 몇 건이에요?', '음성군에 매물 몇 개 있어요?' 처럼 수를 물을 때 씁니다 (이때는 매물 목록을 보여주지 않음).",
             {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
             list_regions),
    ]
