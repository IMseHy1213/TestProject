"""v1.2 온라인 학습: 손님 반응이 오는 그 순간 모델을 고쳐, 바로 다음 검색부터 순서가 달라짐.

    ♡ 관심 / ✕ 별로예요(+이유) 또는 대화 중 "이건 비싸서 싫어" (record_feedback 도구)
      → 방금 보여준 다른 매물과 비교해 '이 손님이 중요하게 보는 것'을 조정 (preference.py, 찾아보기와 같은 학습기)
         예) 넓은 땅을 ♡, 좁은 땅을 ✕ → 면적 중요도 ↑    "너무 비싸요" → 가격 중요도 ↑
      → 이 방문자의 모델(agent:<바꾼 id>) + 사무실 공통 모델(처음 온 손님의 기본값) 저장
      → 다음 search_listings 가 이 가중치로 점수 계산

다시 학습(재훈련)을 기다리지 않고 반응 하나마다 조금씩 고치는 방식이라 '온라인' 학습이라고 부릅니다.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from ..recommender import _text_blob, requested_criteria, score_listing
from ..storage import OFFICE_MODEL


def pref_key(owner: str) -> str:
    """방문자 학습 모델의 저장 이름 (방문자 id 는 바꾼 값만 저장)."""
    return "agent:" + hashlib.sha256(owner.encode()).hexdigest()[:16] if owner else ""


def learn_from_feedback(agent, ctx, listing, action: str, reason: str = "") -> list[str]:
    """반응 하나로 이 방문자 모델과 사무실 공통 모델을 고침. 바뀐 내용을 문장으로 돌려줌."""
    from ..observability.metrics import METRICS
    from ..tool_agent.tools import requirement_from_args

    if not ctx.pref_key:
        return []
    req = requirement_from_args(agent, ctx.last_search or {"query": ""})
    opt = agent._options(ctx.pref_key)
    cur = score_listing(listing, req, opt)
    by_id = {l.listing_id: l for l in agent.listings()}
    shown = [i for i in dict.fromkeys(list(ctx.last_shown) + list(ctx.seen[-10:])) if i != listing.listing_id]
    others = [score_listing(by_id[i], req, opt).factors for i in shown if i in by_id][:8]
    kw = dict(jimok=listing.jimok, zoning=listing.zoning, text=_text_blob(listing), reason=reason,
              requested=requested_criteria(req), protected=set(req.keywords) | set(req.jimok) | set(req.zoning))
    changes = opt.pref.learn(action, cur.factors, others, **kw)
    opt.prior.learn(action, cur.factors, others, **kw)
    # 온라인 로지스틱 회귀: 조건으로 말하지 않은 '넓은 땅 / 싼 땅' 취향 (방문자 모델에만 - 사람마다 다르므로)
    from ..preference import numeric_features

    changes += opt.pref.learn_numeric(numeric_features(listing, agent.feature_stats()), action == "like")
    agent.storage.save_preference(ctx.pref_key, opt.pref)
    agent.storage.save_preference(OFFICE_MODEL, opt.prior)
    METRICS.counter("rea_online_learning_total", "온라인 학습 횟수 (손님 반응 종류별)").inc(action=action)
    return changes


def learned_note(agent, key: str) -> Optional[str]:
    """검색 결과에 붙일 안내: 이 방문자에게서 배운 것 (없으면 None)."""
    if not key:
        return None
    pref = agent.storage.load_preference(key)
    if not pref.n_feedback:
        return None
    w = pref.weights(agent.office_model)
    from ..preference import BASE_WEIGHTS, CRITERIA_KO

    base_total = sum(BASE_WEIGHTS.values())
    ups = [k for k in w if w[k] > BASE_WEIGHTS[k] * 100 / base_total * 1.25]
    likes = [v for kind in pref.attrs.values() for v, (l, d) in kind.items() if l - d >= 1][:3]
    parts = []
    if ups:
        parts.append(", ".join(CRITERIA_KO[k] for k in sorted(ups, key=lambda k: -w[k])[:2]) + " 중요")
    n = pref.numeric or {}
    taste = [label for k, sign, label in (("area", 1, "넓은 땅"), ("area", -1, "아담한 땅"), ("price", -1, "싼 땅"),
                                          ("ppp", -1, "평당가 낮은 땅")) if n.get(k, 0) * sign >= 0.4]
    if taste:
        parts.append(", ".join(taste[:2]) + " 선호")
    if likes:
        parts.append("선호: " + ", ".join(likes))
    return f"반응 {pref.n_feedback}건을 반영해 순서를 맞췄습니다 ({' · '.join(parts)})" if parts else None


def model_view(agent, key: str) -> dict[str, Any]:
    pref = agent.storage.load_preference(key)
    return {"n": pref.n_feedback, "describe": pref.describe(agent.office_model)}
