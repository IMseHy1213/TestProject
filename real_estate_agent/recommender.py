"""매물 점수화·순위화 + '다음 매물 보기' 세션.

점수(100점 만점) = 지역 15 + 가격 25 + 면적 20 + 지목 15 + 용도지역 10 + 도로 5 + 키워드 5 + 최신성 5
   (+ 공공 실거래가 대비 저렴하면 가점 최대 5, 상한 100)
   손님 피드백을 학습하면 위 비율이 손님마다 달라지고, 선호 속성에 ±6점 가감됩니다 (preference.py).
손님이 지정하지 않은 항목은 중립 점수(70%)를 줍니다.

필수 조건(예산 초과·면적 크게 벗어남·맹지·제외 키워드·지역 불일치·계약완료)은 후보에서 제외하되,
후보가 너무 적으면 에이전트가 조건을 단계적으로 완화합니다(완화 내역은 결과에 표시).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .models import CustomerRequirement, LandListing, ScoredListing, format_price
from .preference import BASE_WEIGHTS as WEIGHTS
from .preference import feature_hit
from .land_use import check_use
from .regions import address_in_region
NEUTRAL = 0.7
# 조건 완화와 무관하게 고정된 감점 기울기: 면적 30% 벗어나면 0점, 예산 15% 초과면 0점
AREA_DECAY = 0.30
PRICE_DECAY = 0.15

# 목적별 선호 지목 / 용도지역 (1.0 = 최적, 0.6 = 가능하나 전용·허가 필요)
PURPOSE_PROFILE: dict[str, dict[str, dict[str, float]]] = {
    "전원주택": {
        "jimok": {"대": 1.0, "전": 0.75, "답": 0.6, "과수원": 0.6, "잡종지": 0.6, "임야": 0.5},
        "zoning": {"계획관리지역": 1.0, "자연녹지지역": 0.9, "생산관리지역": 0.7, "보전관리지역": 0.5,
                   "제1종일반주거지역": 0.9, "제1종전용주거지역": 0.9},
    },
    "창고": {
        "jimok": {"창고용지": 1.0, "잡종지": 0.95, "공장용지": 0.9, "대": 0.8, "전": 0.55, "답": 0.5},
        "zoning": {"계획관리지역": 1.0, "준공업지역": 0.9, "일반공업지역": 0.9, "생산관리지역": 0.6,
                   "자연녹지지역": 0.6},
    },
    "공장": {
        "jimok": {"공장용지": 1.0, "잡종지": 0.9, "창고용지": 0.8, "대": 0.75, "전": 0.5, "답": 0.45},
        "zoning": {"계획관리지역": 1.0, "일반공업지역": 1.0, "준공업지역": 0.95, "전용공업지역": 0.9,
                   "생산관리지역": 0.5},
    },
    "농사": {
        "jimok": {"전": 1.0, "답": 1.0, "과수원": 1.0, "목장용지": 0.7, "임야": 0.4},
        "zoning": {"농림지역": 1.0, "생산관리지역": 1.0, "계획관리지역": 0.8, "보전관리지역": 0.8,
                   "생산녹지지역": 0.9},
    },
    "근린생활": {
        "jimok": {"대": 1.0, "잡종지": 0.85, "전": 0.6, "답": 0.55},
        "zoning": {"계획관리지역": 1.0, "자연녹지지역": 0.9, "일반상업지역": 1.0, "근린상업지역": 1.0,
                   "제2종일반주거지역": 0.8},
    },
    "캠핑장": {
        "jimok": {"잡종지": 1.0, "임야": 0.8, "전": 0.8, "답": 0.6, "대": 0.8},
        "zoning": {"계획관리지역": 1.0, "생산관리지역": 0.7, "자연녹지지역": 0.8, "보전관리지역": 0.6},
    },
    "태양광": {
        "jimok": {"잡종지": 1.0, "임야": 0.8, "전": 0.8, "답": 0.8},
        "zoning": {"계획관리지역": 1.0, "생산관리지역": 0.9, "보전관리지역": 0.6, "농림지역": 0.6},
    },
    "투자": {
        "jimok": {"대": 1.0, "잡종지": 0.9, "전": 0.8, "답": 0.75, "임야": 0.7},
        "zoning": {"계획관리지역": 1.0, "자연녹지지역": 1.0, "생산관리지역": 0.7, "보전관리지역": 0.5},
    },
}


@dataclass
class RecommendOptions:
    budget_tolerance: float = 0.10  # 예산 초과 허용 폭
    area_tolerance: float = 0.20  # 면적 범위 벗어남 허용 폭
    strict_region: bool = True
    min_candidates: int = 3
    market_price_per_m2: dict[str, dict[str, float]] = field(default_factory=dict)  # {시군구: {지목: 만원/㎡}}
    pref: Any = None  # 이 손님의 학습 모델 (PreferenceModel)
    prior: Any = None  # 사무실 공통 학습 모델 (PreferenceModel)
    # 손님 반응으로 생긴 '방향' 조건 (예: "너무 비싸요" -> 방금 매물보다 싼 것만). 조건 완화 대상이 아님
    price_below: Optional[int] = None  # 만원, 이 가격 미만만
    price_above: Optional[int] = None  # 만원, 이 가격 초과만
    area_above: Optional[float] = None  # ㎡, 이 면적 초과만
    area_below: Optional[float] = None  # ㎡, 이 면적 미만만

    def weights(self) -> dict[str, float]:
        if self.pref is not None:
            return self.pref.weights(self.prior)
        if self.prior is not None:
            return self.prior.weights()
        return dict(WEIGHTS)


def _region_stem(r: str) -> str:
    return re.sub(r"(특별시|광역시|특별자치시|특별자치도|시|군|구|읍|면|동|리)$", "", r) or r


def region_match(listing: LandListing, regions: list[str]) -> Optional[str]:
    addr = listing.address.replace(" ", "")
    for r in regions:
        rr = r.replace(" ", "")
        if rr in addr or address_in_region(listing.address, r):  # '경북' <-> '경상북도/문경시 …'
            return r
        stem = _region_stem(rr)
        if len(stem) >= 2 and re.search(re.escape(stem) + r"(시|군|구|읍|면|동|리)", addr):
            return r
    return None


def _text_blob(l: LandListing) -> str:
    extra = " ".join(str(v) for v in l.extra.values() if isinstance(v, (str, int, float)))
    return f"{l.address} {l.road_access} {l.description} {l.zoning} {l.jimok} {extra}"


def hard_filter(l: LandListing, req: CustomerRequirement, opt: RecommendOptions) -> Optional[str]:
    """제외 사유를 반환 (통과하면 None)."""
    if l.status in {"계약완료", "보류"}:
        return f"상태 '{l.status}'"
    if opt.price_below is not None and l.price_manwon >= opt.price_below:
        return "앞서 본 매물보다 비쌈"
    if opt.price_above is not None and l.price_manwon <= opt.price_above:
        return "앞서 본 매물보다 저렴함"
    if opt.area_above is not None and l.area_m2 <= opt.area_above:
        return "앞서 본 매물보다 좁음"
    if opt.area_below is not None and l.area_m2 >= opt.area_below:
        return "앞서 본 매물보다 넓음"
    if req.budget_max and l.price_manwon > req.budget_max * (1 + opt.budget_tolerance):
        return "예산 초과"
    if req.budget_min and l.price_manwon < req.budget_min * (1 - opt.budget_tolerance):
        return "예산보다 저렴"  # '30억 이상' 을 찾는데 1억짜리를 보여주지 않음
    if req.area_min_m2 and l.area_m2 < req.area_min_m2 * (1 - opt.area_tolerance):
        return "면적 부족"
    if req.area_max_m2 and l.area_m2 > req.area_max_m2 * (1 + opt.area_tolerance * 2):
        return "면적 과다"
    if req.purpose:  # '창고를 지을 수 있는 토지' -> 창고를 지을 수 없는 땅은 제외
        use = check_use(l, req.purpose)
        if use is not None and not use.ok:
            return f"{req.purpose} 건축 불가"
    if req.need_road and l.is_blind_land:
        return "맹지"
    blob = _text_blob(l)
    for k in req.excluded_keywords:
        if k and k in blob and not re.search(re.escape(k) + r".{0,4}(없|무|아님|X)", blob):
            return f"제외 조건 '{k}'"
    if req.regions and opt.strict_region and not region_match(l, req.regions):
        return "지역 불일치"
    return None


def score_listing(l: LandListing, req: CustomerRequirement, opt: RecommendOptions,
                  today: Optional[date] = None) -> ScoredListing:
    today = today or date.today()
    reasons: list[str] = []
    penalties: list[str] = []
    f: dict[str, float] = {}
    profile = PURPOSE_PROFILE.get(req.purpose, {})

    # 지역
    if req.regions:
        hit = region_match(l, req.regions)
        f["region"] = 1.0 if hit else 0.2
        (reasons if hit else penalties).append(f"희망 지역 {'일치(' + hit + ')' if hit else '아님'}")
    else:
        f["region"] = NEUTRAL

    # 가격
    p = l.price_manwon
    if req.budget_max:
        if p <= req.budget_max:
            f["price"] = 0.8 + 0.2 * (1 - p / req.budget_max)
            if req.budget_min and p < req.budget_min:
                f["price"] = 0.65
                penalties.append("예산 하한보다 저렴 (조건 재확인 필요)")
            else:
                reasons.append(f"예산 내 ({format_price(p)} / 예산 {format_price(req.budget_max)})")
        else:
            over = (p - req.budget_max) / req.budget_max
            f["price"] = max(0.0, 0.45 * (1 - over / PRICE_DECAY))
            penalties.append(f"예산 {over * 100:.0f}% 초과 (협상 필요)")
    elif req.budget_min:
        f["price"] = 1.0 if p >= req.budget_min else 0.6
    else:
        f["price"] = NEUTRAL

    # 면적
    a = l.area_m2
    lo, hi = req.area_min_m2, req.area_max_m2
    if lo or hi:
        if (not lo or a >= lo * 0.98) and (not hi or a <= hi * 1.02):  # 2% 이내는 반올림 오차로 보고 충족
            f["area"] = 1.0
            reasons.append(f"희망 면적 충족 ({l.area_pyeong:,.0f}평)")
        else:
            ref = lo if (lo and a < lo) else hi
            dev = abs(a - ref) / ref
            f["area"] = max(0.0, 1 - dev / AREA_DECAY)
            penalties.append(f"면적 {'부족' if lo and a < lo else '초과'} ({l.area_pyeong:,.0f}평, {dev * 100:.0f}%)")
    else:
        f["area"] = NEUTRAL

    # 목적(창고/공장/전원주택 …)으로 쓸 수 있는 근거
    use = check_use(l, req.purpose) if req.purpose else None
    if use is not None and use.ok:
        reasons.append(f"{req.purpose} 가능: {use.reason}")
        if use.caution:
            penalties.append(use.caution)

    # 지목
    if req.jimok:
        if l.jimok in req.jimok:
            f["jimok"] = 1.0
            reasons.append(f"희망 지목 '{l.jimok}'")
        else:
            f["jimok"] = profile.get("jimok", {}).get(l.jimok, 0.2) * 0.7
            penalties.append(f"지목 '{l.jimok or '미기재'}' (희망: {','.join(req.jimok)})")
    elif profile.get("jimok"):
        f["jimok"] = profile["jimok"].get(l.jimok, 0.3)
        if f["jimok"] >= 0.9:
            reasons.append(f"{req.purpose} 용도에 적합한 지목 '{l.jimok}'")
        elif f["jimok"] <= 0.6 and l.jimok:
            penalties.append(f"지목 '{l.jimok}' - {req.purpose} 용도는 전용/개발행위허가 확인 필요")
    else:
        f["jimok"] = NEUTRAL

    # 용도지역
    if req.zoning:
        z_hit = any(z in l.zoning or l.zoning in z for z in req.zoning) if l.zoning else False
        f["zoning"] = 1.0 if z_hit else profile.get("zoning", {}).get(l.zoning, 0.25) * 0.7
        (reasons if z_hit else penalties).append(f"용도지역 '{l.zoning or '미기재'}'")
    elif profile.get("zoning"):
        f["zoning"] = profile["zoning"].get(l.zoning, 0.4 if l.zoning else 0.5)
        if f["zoning"] >= 0.9:
            reasons.append(f"{req.purpose}에 유리한 '{l.zoning}'")
    else:
        f["zoning"] = NEUTRAL

    # 도로
    if l.is_blind_land:
        f["road"] = 0.0
        penalties.append("맹지 (진입로 확보 필요)")
    elif l.road_access:
        f["road"] = 1.0
        if req.need_road:
            reasons.append(f"도로 조건: {l.road_access}")
    else:
        f["road"] = 0.5

    # 키워드
    if req.keywords:
        blob = _text_blob(l)
        hits = [k for k in req.keywords if feature_hit(k, blob)]  # '채광' 은 남향·일조 등도 인정
        f["keyword"] = len(hits) / len(req.keywords)
        if hits:
            reasons.append("특징 일치: " + ", ".join(hits))
    else:
        f["keyword"] = NEUTRAL

    # 최신성
    if l.registered_date:
        days = (today - l.registered_date).days
        f["fresh"] = 1.0 if days <= 90 else 0.75 if days <= 365 else 0.4
        if days <= 30:
            reasons.append("최근 등록 매물")
    else:
        f["fresh"] = 0.6

    f = {k: max(0.0, min(1.0, v)) for k, v in f.items()}
    weights = opt.weights()
    score = sum(weights[k] * v for k, v in f.items())

    # 피드백 학습: 이 손님이 좋아했던/싫어했던 지목·용도지역·특징
    if opt.pref is not None:
        bonus, notes = opt.pref.attribute_bonus(l.jimok, l.zoning, _text_blob(l))
        score += bonus
        (reasons if bonus > 0 else penalties).extend(notes)

    # 공공 실거래가 대비
    market = _market_ppm(l, opt.market_price_per_m2)
    if market:
        ratio = (l.price_manwon / l.area_m2) / market
        l.extra["market_ratio"] = round(ratio, 2)
        if ratio <= 0.85:
            score += 5
            reasons.append(f"주변 실거래 시세 대비 {100 - ratio * 100:.0f}% 저렴")
        elif ratio >= 1.3:
            score -= 3
            penalties.append(f"주변 실거래 시세보다 {ratio * 100 - 100:.0f}% 높음")

    return ScoredListing(l, round(max(0.0, min(100.0, score)), 1), reasons, penalties, f)


def _market_ppm(l: LandListing, market: dict[str, dict[str, float]]) -> Optional[float]:
    for region, by_jimok in market.items():
        if region in l.address:
            return by_jimok.get(l.jimok) or by_jimok.get("*")
    return None


# 자동 조건 완화 (기본은 꺼짐: 조건에 맞는 매물이 없으면 NOT_FOUND_MESSAGE 로 안내).
# 지역은 어떤 경우에도 넓히지 않음 - 다른 지역 매물을 엉뚱하게 보여주지 않기 위해.
RELAX_STEPS = [
    ("예산 허용 폭 20%로 확대", {"budget_tolerance": 0.20}),
    ("면적 허용 폭 35%로 확대", {"area_tolerance": 0.35}),
]
NOT_FOUND_MESSAGE = "해당 조건에 맞는 매물이 없습니다."  # 처음부터 맞는 매물이 없을 때
EXHAUSTED_MESSAGE = "해당 조건에 맞는 매물은 더 이상 없습니다."  # 보여주다가 다 떨어졌을 때


def rank(listings: list[LandListing], req: CustomerRequirement, opt: Optional[RecommendOptions] = None,
         exclude_ids: set[str] | None = None, today: Optional[date] = None,
         auto_relax: bool = False) -> tuple[list[ScoredListing], list[str], dict[str, int]]:
    """(순위 목록, 완화된 조건 목록, 제외 사유 통계)"""
    opt = opt or RecommendOptions()
    exclude_ids = exclude_ids or set()
    relaxed: list[str] = []
    pool = [l for l in listings if l.listing_id not in exclude_ids]

    def run() -> tuple[list[ScoredListing], dict[str, int]]:
        stats: dict[str, int] = {}
        out = []
        for l in pool:
            reason = hard_filter(l, req, opt)
            if reason:
                key = reason.split(" '")[0]
                stats[key] = stats.get(key, 0) + 1
                continue
            out.append(score_listing(l, req, opt, today))
        out.sort(key=lambda s: (-s.score, s.listing.price_manwon))
        return out, stats

    result, stats = run()
    if auto_relax:
        for label, change in RELAX_STEPS:
            if len(result) >= opt.min_candidates:
                break
            for k, v in change.items():
                setattr(opt, k, v)
            relaxed.append(label)
            result, stats = run()
    return result, relaxed, stats


def requested_criteria(req: CustomerRequirement) -> set[str]:
    """손님이 직접 말한 조건 항목."""
    out = set()
    if req.regions:
        out.add("region")
    if req.budget_min or req.budget_max:
        out.add("price")
    if req.area_min_m2 or req.area_max_m2:
        out.add("area")
    if req.jimok or req.purpose:
        out.add("jimok")
    if req.zoning or req.purpose:
        out.add("zoning")
    if req.need_road:
        out.add("road")
    if req.keywords:
        out.add("keyword")
    return out


# 거절 사유 -> 다음 매물이 가야 할 방향
_TOO_CHEAP = r"너무\s*싸|싸서|더\s*비싼|비싸도\s*(돼|괜찮)"
_TOO_EXPENSIVE = (r"비싸|비쌈|저렴한|더\s*싼|싼\s*(거|걸|것|매물|땅)|싸게|가격.*(높|부담|내려|낮)|"
                  r"예산.*(초과|넘)|돈이\s*부족")
_TOO_BIG = r"너무\s*(넓|커|크)|넓어서|커서|더\s*(작|좁)|작은\s*(거|걸|것|땅|매물)|좁은\s*(거|걸|것|땅|매물)"
_TOO_SMALL = (r"좁아|좁다|좁네|좁은데|좁고|작아|작다|작네|작은데|작고|너무\s*(좁|작)|더\s*(넓|큰)|"
              r"넓은\s*(거|걸|것|땅|매물)|큰\s*(거|걸|것|땅|매물)|면적.*(부족|작|좁)|평수.*(부족|작)")


def direction_from_reason(reason: str) -> dict[str, str]:
    """'너무 비싸요' -> {'price': 'lower'}, '너무 좁아요' -> {'area': 'larger'} ..."""
    d: dict[str, str] = {}
    if not reason:
        return d
    if re.search(_TOO_CHEAP, reason):
        d["price"] = "higher"
    elif re.search(_TOO_EXPENSIVE, reason):
        d["price"] = "lower"
    if re.search(_TOO_BIG, reason):
        d["area"] = "smaller"
    elif re.search(_TOO_SMALL, reason):
        d["area"] = "larger"
    return d


_DIRECTION_FIELDS = ("price_below", "price_above", "area_above", "area_below")


class RecommendationSession:
    """한 손님에 대한 추천 세션.

    current() 가 지금 보여줄 1순위 매물입니다. 손님이 마음에 안 들어하면 next() 를 부르면
      1) 그 매물을 거절 목록에 넣고
      2) 거절에서 손님 취향을 학습한 뒤 (opt.pref 가 있을 때)
      3) 남은 매물을 다시 순위 매겨 그 다음 순위(2순위, 3순위 ...) 매물을 보여줍니다.
    """

    def __init__(self, listings: list[LandListing], req: CustomerRequirement,
                 opt: Optional[RecommendOptions] = None, exclude_ids: set[str] | None = None):
        self.listings = listings
        self.req = req
        self.opt = opt or RecommendOptions()
        self.rejected: dict[str, str] = {}  # listing_id -> 사유
        self.history: list[str] = []  # 이번 상담에서 넘긴 매물 (순서대로)
        self._direction_history: list[dict] = []  # 넘길 때마다 직전 방향 조건 (이전 버튼으로 되돌리기용)
        self.events: list[dict] = []  # 손님 반응 기록 (생성형 AI 학습에 사용)
        self.ai_changes: list[str] = []  # 직전 AI 학습으로 바뀐 내용
        self.base_exclude = set(exclude_ids or set())
        self.last_learning: list[str] = []  # 직전 피드백으로 배운 내용
        self.reordered = False  # 학습 때문에 원래 2순위가 아닌 매물이 올라왔는지
        self._rerank()

    def _rerank(self) -> None:
        opt = RecommendOptions(**{**self.opt.__dict__})  # 조건 완화가 원본 옵션을 바꾸지 않도록 복사
        self.ranked, self.relaxed, self.filter_stats = rank(
            self.listings, self.req, opt, exclude_ids=self.base_exclude | set(self.rejected)
        )

    def _learn(self, action: str, cur: ScoredListing, reason: str = "") -> None:
        from .ai_learning import listing_brief

        self.events.append({"action": action, "reason": reason, "listing": listing_brief(cur.listing)})
        self.last_learning = []
        if self.opt.pref is None:
            return
        others = [s.factors for s in self.ranked if s is not cur][:8]
        l = cur.listing
        kw = dict(jimok=l.jimok, zoning=l.zoning, text=_text_blob(l), reason=reason,
                  requested=requested_criteria(self.req),
                  protected=set(self.req.keywords) | set(self.req.jimok) | set(self.req.zoning))
        self.last_learning = self.opt.pref.learn(action, cur.factors, others, **kw)
        if self.opt.prior is not None:  # 사무실 공통 모델도 함께 학습
            self.opt.prior.learn(action, cur.factors, others, **kw)

    @property
    def cursor(self) -> int:
        """지금까지 넘긴 매물 수 (현재 매물의 순위 = cursor + 1)."""
        return len(self.history)

    @property
    def total(self) -> int:
        return len(self.ranked) + len(self.history)

    def current(self) -> Optional[ScoredListing]:
        return self.ranked[0] if self.ranked else None

    def next(self, reason: str = "") -> Optional[ScoredListing]:
        """현재 매물이 마음에 안 들 때: 거절 + 학습 후 다음 순위 매물."""
        cur = self.current()
        if cur is None:
            return None
        expected = self.ranked[1].listing.listing_id if len(self.ranked) > 1 else None
        self._learn("reject", cur, reason)
        self._push_direction(cur, reason)
        self.rejected[cur.listing.listing_id] = reason or "마음에 들지 않음"
        self.history.append(cur.listing.listing_id)
        self._rerank()
        nxt = self.current()
        self.reordered = bool(nxt and expected and nxt.listing.listing_id != expected)
        return nxt

    def like(self) -> None:
        """관심 표시: 학습만 하고 현재 매물은 그대로 보여줌."""
        cur = self.current()
        if cur is None:
            return
        self._learn("like", cur)
        self._rerank()
        self._bring_to_front(cur.listing.listing_id)

    def previous(self) -> Optional[ScoredListing]:
        """직전에 넘긴 매물로 돌아가기 (학습 내용은 유지)."""
        if self.history:
            listing_id = self.history.pop()
            if self._direction_history:  # 그 매물을 넘기면서 생긴 '더 싼 것만' 같은 조건도 되돌림
                for k, v in self._direction_history.pop().items():
                    setattr(self.opt, k, v)
            self.rejected.pop(listing_id, None)
            self._rerank()
            self._bring_to_front(listing_id)
        return self.current()

    def reject_current(self, reason: str) -> None:
        """피드백으로 조건을 바꾸기 전에 현재 매물을 거절·학습 (재순위는 update_requirement 에서)."""
        cur = self.current()
        if cur:
            self._learn("reject", cur, reason)
            self._push_direction(cur, reason)
            self.rejected[cur.listing.listing_id] = reason
            self.history.append(cur.listing.listing_id)

    def _push_direction(self, cur: ScoredListing, reason: str) -> None:
        """'너무 비싸요' 면 방금 매물보다 싼 매물만, '너무 좁아요' 면 더 넓은 매물만 보여주도록 조건 추가."""
        self._direction_history.append({k: getattr(self.opt, k) for k in _DIRECTION_FIELDS})
        d = direction_from_reason(reason)
        l = cur.listing
        o = self.opt
        if d.get("price") == "lower":
            o.price_below = min(o.price_below or l.price_manwon, l.price_manwon)
            self.last_learning.append(f"{format_price(l.price_manwon)}보다 저렴한 매물만 보여드려요")
        elif d.get("price") == "higher":
            o.price_above = max(o.price_above or l.price_manwon, l.price_manwon)
            self.last_learning.append(f"{format_price(l.price_manwon)}보다 비싼 매물도 괜찮은 것으로 보고 찾습니다")
        if d.get("area") == "larger":
            o.area_above = max(o.area_above or l.area_m2, l.area_m2)
            self.last_learning.append(f"{l.area_pyeong:,.0f}평보다 넓은 매물만 보여드려요")
        elif d.get("area") == "smaller":
            o.area_below = min(o.area_below or l.area_m2, l.area_m2)
            self.last_learning.append(f"{l.area_pyeong:,.0f}평보다 좁은 매물만 보여드려요")

    def rerank_keep_order_of_seen(self) -> None:
        """AI 학습 결과(중요도·선호) 반영 후 남은 매물 순위 다시 매김."""
        self._rerank()

    def update_requirement(self, req: CustomerRequirement) -> None:
        """피드백으로 조건이 바뀌면 이미 거절한 매물은 빼고 다시 순위를 매김."""
        self.req = req
        self._rerank()

    def top(self, n: int = 5) -> list[ScoredListing]:
        return self.ranked[:n]

    def _bring_to_front(self, listing_id: str) -> None:
        for i, s in enumerate(self.ranked):
            if s.listing.listing_id == listing_id:
                self.ranked.insert(0, self.ranked.pop(i))
                return
