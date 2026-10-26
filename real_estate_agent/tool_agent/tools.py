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
def brief(l: LandListing, s: Optional[ScoredListing] = None) -> dict[str, Any]:
    from ..agent import mask_address

    d = {
        "id": l.listing_id,
        "매물번호": l.office_no or None,
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


class ToolRegistry:
    """도구 목록 + 실행 (인자 검사, 오류를 관찰로 변환, 실행 시간 측정)."""

    def __init__(self, agent, ctx: Optional[AgentContext] = None, lock: Optional[threading.RLock] = None):
        self.agent = agent
        self.ctx = ctx or AgentContext()
        self.lock = lock
        self.tools: dict[str, Tool] = {}
        for tool in build_tools(self):
            self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """도구 실행 -> {'ok': bool, 'result' 또는 'error', 'ms'}"""
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
                    with self.lock or nullcontext():  # 매물 목록·DB 는 여러 접속자가 함께 씀
                        result = tool.fn(**clean)
                    out = {"ok": True, "result": result}
                except ToolError as e:
                    out = {"ok": False, "error": str(e)}
                except Exception as e:  # 도구 하나가 실패해도 대화는 계속
                    out = {"ok": False, "error": f"도구 실행 중 오류: {type(e).__name__}: {e}"}
        out["ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out


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
        key = listing_id.strip()
        for l in agent.listings():
            if l.listing_id == key or (l.office_no and l.office_no == key):
                return l
        raise ToolError(f"매물 '{listing_id}' 을(를) 찾을 수 없습니다. search_listings 결과의 id 를 쓰세요.")

    def search_listings(query: str = "", regions: Optional[list[str]] = None, jimok: Optional[list[str]] = None,
                        purpose: str = "", budget_min_manwon: Optional[int] = None,
                        budget_max_manwon: Optional[int] = None, area_min_pyeong: Optional[float] = None,
                        area_max_pyeong: Optional[float] = None, keywords: Optional[list[str]] = None,
                        sort: str = "score", limit: int = 5) -> dict[str, Any]:
        from ..regions import canonical_region
        from ..requirement_parser import RequirementParser

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
        req.customer_name = ctx.customer
        session = agent.recommend(req)
        ranked = [s for s in session.ranked if s.listing.listing_id not in ctx.rejected]
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
        order = {"score": "추천순", "price_asc": "가격 낮은 순", "price_desc": "가격 높은 순", "area_desc": "넓은 순",
                 "price_per_pyeong_asc": "평당가 낮은 순"}[sort]
        out: dict[str, Any] = {"이해한_조건": req.describe(), "정렬": order, "조건에_맞는_매물_수": len(ranked),
                               "매물": [brief(s.listing, s) for s in shown]}
        if not ranked:
            out["안내"] = NOT_FOUND_MESSAGE
        if req.notes:
            out["참고"] = req.notes
        if ctx.rejected:
            out["제외한_매물"] = list(ctx.rejected)
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
            return {"id": l.listing_id, "용도": purpose, "판단": "규칙 없음"}
        out = {"id": l.listing_id, "용도": purpose, "가능": res.ok, "근거": res.reason}
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
        def pick(key):
            return min(items, key=key).listing_id
        return {
            "매물": rows,
            "가장_싼_매물": pick(lambda l: l.price_manwon),
            "가장_넓은_매물": max(items, key=lambda l: l.area_m2).listing_id,
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
        return {"id": l.listing_id, "기록": action, "이유": reason or None, "안내": note}

    def list_regions(limit: int = 40) -> dict[str, Any]:
        counts: Counter = Counter()
        for l in agent.listings():
            for tok in set(l.address.split()):
                if len(tok) >= 2 and tok[-1:] in "시군구읍면":
                    counts[tok] += 1
        return {"지역별_매물_수": dict(counts.most_common(limit)), "전체_매물_수": len(agent.listings())}

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
        Tool("list_regions", "보유 매물이 있는 지역(시·군·읍·면)과 지역별 매물 수.",
             {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
             list_regions),
    ]
