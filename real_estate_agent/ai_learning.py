"""생성형 AI(GPT/Claude)로 손님 취향 학습.

손님이 매물을 거절하거나 관심을 보일 때마다, 지금까지의 반응 기록 전체를 AI 에게 보여주고
"이 손님이 실제로 중요하게 여기는 것"을 추론하게 합니다. AI 의 답(JSON)은 검증한 뒤 추천에 반영합니다.

  - weights          : 항목 중요도 올림/내림  (예: 반복해서 '좁다'고 함 -> 면적 ↑)
  - prefer/avoid     : 좋아하는/싫어하는 특징·지목 (예: 계곡 매물에만 관심 -> 계곡 선호)
  - add_keywords     : 검색 조건에 추가할 특징
  - add_excluded     : 제외할 특징 - 손님이 직접 말한 단어일 때만 반영 (AI 의 추측으로 매물을 빼지 않도록)
  - summary          : 화면에 보여줄 한두 문장 ("가격보다 넓은 면적과 좋은 도로 조건을 중시하십니다")

AI 를 쓸 수 없으면(키 없음, 연결 실패) 아무것도 하지 않고, 기존 규칙 기반 학습(preference.py)만 동작합니다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .llm import LLMRouter, LLMUnavailable
from .models import JIMOK_TYPES, format_price
from .preference import BASE_WEIGHTS, CRITERIA_KO, FEATURE_SYNONYMS, MULT_MAX, MULT_MIN

log = logging.getLogger(__name__)

AI_STEP = 1.3  # AI 가 '중요도 올림' 이라고 하면 30% ↑
MAX_EVENTS = 12

SYSTEM_PROMPT = f"""당신은 토지 중개 상담을 돕는 AI 입니다.
손님의 검색 조건과, 지금까지 보여준 매물에 대한 손님의 반응(거절/관심, 이유)을 보고
손님의 숨은 취향을 추론하세요. 반응이 일관되게 가리키는 것만 말하고, 근거가 없으면 비워두세요.

반드시 아래 형식의 JSON 하나만 출력하세요.
{{
  "summary": "손님 취향 한두 문장 (존댓말, 손님에게 보여줌)",
  "weights": {{"항목": "up" 또는 "down"}},
  "prefer_features": [], "avoid_features": [],
  "prefer_jimok": [], "avoid_jimok": [],
  "add_keywords": [], "add_excluded_keywords": []
}}
- weights 의 항목: {", ".join(f"{k}({v})" for k, v in CRITERIA_KO.items())}
- prefer/avoid_features 는 다음 중에서만: {", ".join(FEATURE_SYNONYMS)}
- prefer/avoid_jimok 는 다음 중에서만: {", ".join(JIMOK_TYPES)}
- add_excluded_keywords 는 손님이 직접 싫다고 말한 것만 넣으세요."""


def describe_event(event: dict[str, Any]) -> str:
    l = event["listing"]
    act = {"reject": "거절", "like": "관심"}.get(event["action"], event["action"])
    reason = f" / 이유: {event['reason']}" if event.get("reason") else ""
    return (f"[{act}] {l['region']} · 지목 {l['jimok'] or '-'} · {l['zoning'] or '-'} · {l['area_pyeong']}평 · "
            f"{l['price']} · 도로 {l['road'] or '-'} · 특징 {l['description'] or '-'}{reason}")


def listing_brief(listing) -> dict[str, Any]:
    """AI 에 보낼 매물 정보 (지번·연락처 제외)."""
    region = " ".join(t for t in listing.address.split() if t[-1:] in "도시군구읍면리")[:40]
    return {
        "region": region, "jimok": listing.jimok, "zoning": listing.zoning,
        "area_pyeong": round(listing.area_pyeong), "price": format_price(listing.price_manwon),
        "road": listing.road_access, "description": listing.description[:120],
    }


def ask_ai(llm: LLMRouter, requirement_text: str, events: list[dict]) -> Optional[dict]:
    if not llm or not llm.available or not events:
        return None
    lines = "\n".join(describe_event(e) for e in events[-MAX_EVENTS:])
    try:
        raw = llm.complete_json(SYSTEM_PROMPT, f"손님 검색 조건: {requirement_text}\n\n손님 반응 기록:\n{lines}",
                                max_tokens=1200, optional=True)
    except LLMUnavailable as e:
        log.info("AI 취향 학습 건너뜀: %s", e)
        return None
    return sanitize_plan(raw)


def _str_list(v: Any, allowed: Optional[set] = None, limit: int = 5) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        x = str(x).strip()
        if x and (allowed is None or x in allowed) and x not in out:
            out.append(x)
    return out[:limit]


def sanitize_plan(raw: dict) -> dict:
    """AI 답을 검증: 모르는 항목·값은 버림."""
    weights = {}
    if isinstance(raw.get("weights"), dict):
        for k, v in raw["weights"].items():
            if k in BASE_WEIGHTS and str(v).lower() in {"up", "down"}:
                weights[k] = str(v).lower()
    return {
        "summary": str(raw.get("summary") or "").strip()[:200],
        "weights": weights,
        "prefer_features": _str_list(raw.get("prefer_features"), set(FEATURE_SYNONYMS)),
        "avoid_features": _str_list(raw.get("avoid_features"), set(FEATURE_SYNONYMS)),
        "prefer_jimok": _str_list(raw.get("prefer_jimok"), set(JIMOK_TYPES)),
        "avoid_jimok": _str_list(raw.get("avoid_jimok"), set(JIMOK_TYPES)),
        "add_keywords": _str_list(raw.get("add_keywords"), limit=3),
        "add_excluded_keywords": _str_list(raw.get("add_excluded_keywords"), limit=3),
    }


def apply_plan(session, plan: dict) -> list[str]:
    """검증된 AI 학습 결과를 세션(손님 학습 모델 + 조건)에 반영. 바뀐 내용을 문장으로 반환."""
    pref = session.opt.pref
    req = session.req
    changes: list[str] = []
    if pref is not None:
        for k, d in plan["weights"].items():
            m = pref.multipliers.get(k, 1.0) * (AI_STEP if d == "up" else 1 / AI_STEP)
            pref.multipliers[k] = max(MULT_MIN, min(MULT_MAX, m))
            changes.append(f"{CRITERIA_KO[k]} 중요도 {'↑' if d == 'up' else '↓'}")
        protected = set(req.keywords) | set(req.jimok)
        for kind, key, idx in (("feature", "prefer_features", 0), ("feature", "avoid_features", 1),
                               ("jimok", "prefer_jimok", 0), ("jimok", "avoid_jimok", 1)):
            for v in plan[key]:
                if idx == 1 and v in protected:
                    continue  # 손님이 직접 원한 것은 '싫어함'으로 배우지 않음
                pref.attrs.setdefault(kind, {}).setdefault(v, [0.0, 0.0])[idx] += 1.0
                changes.append(f"'{v}' {'선호' if idx == 0 else '비선호'}")
        pref.ai_summary = plan["summary"] or getattr(pref, "ai_summary", "")
    said = " ".join([req.raw_text] + [e.get("reason", "") for e in session.events])
    for k in plan["add_keywords"]:
        if k not in req.keywords:
            req.keywords.append(k)
            changes.append(f"찾는 특징에 '{k}' 추가")
    for k in plan["add_excluded_keywords"]:
        if k in said and k not in req.excluded_keywords and k not in req.keywords:  # 손님이 직접 말한 것만
            req.excluded_keywords.append(k)
            changes.append(f"'{k}' 제외")
    return changes


def plan_to_json(plan: Optional[dict]) -> str:
    return json.dumps(plan or {}, ensure_ascii=False)
