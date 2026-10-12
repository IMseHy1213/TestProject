"""매물 점수화·순위화 + '다음 매물 보기' 세션.

점수(100점 만점) = 지역 15 + 가격 25 + 면적 20 + 지목 15 + 용도지역 10 + 도로 5 + 키워드 5 + 최신성 5
   (+ 공공 실거래가 대비 저렴하면 가점 최대 5, 상한 100)
손님이 지정하지 않은 항목은 중립 점수(70%)를 줍니다.

필수 조건(예산 초과·면적 크게 벗어남·맹지·제외 키워드·지역 불일치·계약완료)은 후보에서 제외하되,
후보가 너무 적으면 에이전트가 조건을 단계적으로 완화합니다(완화 내역은 결과에 표시).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .models import CustomerRequirement, LandListing, ScoredListing, format_price

WEIGHTS = {
    "region": 15, "price": 25, "area": 20, "jimok": 15, "zoning": 10, "road": 5, "keyword": 5, "fresh": 5,
}
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


def _region_stem(r: str) -> str:
    return re.sub(r"(특별시|광역시|특별자치시|특별자치도|시|군|구|읍|면|동|리)$", "", r) or r


def region_match(listing: LandListing, regions: list[str]) -> Optional[str]:
    addr = listing.address.replace(" ", "")
    for r in regions:
        rr = r.replace(" ", "")
        if rr in addr:
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
    if req.budget_max and l.price_manwon > req.budget_max * (1 + opt.budget_tolerance):
        return "예산 초과"
    if req.area_min_m2 and l.area_m2 < req.area_min_m2 * (1 - opt.area_tolerance):
        return "면적 부족"
    if req.area_max_m2 and l.area_m2 > req.area_max_m2 * (1 + opt.area_tolerance * 2):
        return "면적 과다"
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
        hits = [k for k in req.keywords if k.lower() in blob.lower()]
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

    score = sum(WEIGHTS[k] * max(0.0, min(1.0, v)) for k, v in f.items())

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

    return ScoredListing(l, round(max(0.0, min(100.0, score)), 1), reasons, penalties)


def _market_ppm(l: LandListing, market: dict[str, dict[str, float]]) -> Optional[float]:
    for region, by_jimok in market.items():
        if region in l.address:
            return by_jimok.get(l.jimok) or by_jimok.get("*")
    return None


RELAX_STEPS = [
    ("예산 허용 폭 20%로 확대", {"budget_tolerance": 0.20}),
    ("면적 허용 폭 35%로 확대", {"area_tolerance": 0.35}),
    ("인근 지역까지 확대 (지역은 점수로만 반영)", {"strict_region": False}),
    ("면적 허용 폭 60%로 확대", {"area_tolerance": 0.60}),
]


def rank(listings: list[LandListing], req: CustomerRequirement, opt: Optional[RecommendOptions] = None,
         exclude_ids: set[str] | None = None, today: Optional[date] = None,
         auto_relax: bool = True) -> tuple[list[ScoredListing], list[str], dict[str, int]]:
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


class RecommendationSession:
    """한 손님에 대한 추천 세션. 마음에 안 들면 next() 로 다음 순위 매물을 보여줍니다."""

    def __init__(self, listings: list[LandListing], req: CustomerRequirement,
                 opt: Optional[RecommendOptions] = None, exclude_ids: set[str] | None = None):
        self.listings = listings
        self.req = req
        self.opt = opt or RecommendOptions()
        self.rejected: dict[str, str] = {}  # listing_id -> 사유
        self.base_exclude = set(exclude_ids or set())
        self.cursor = 0
        self._rerank()

    def _rerank(self) -> None:
        opt = RecommendOptions(**{**self.opt.__dict__})
        self.ranked, self.relaxed, self.filter_stats = rank(
            self.listings, self.req, opt, exclude_ids=self.base_exclude | set(self.rejected)
        )
        self.cursor = 0

    @property
    def total(self) -> int:
        return len(self.ranked)

    def current(self) -> Optional[ScoredListing]:
        return self.ranked[self.cursor] if self.cursor < len(self.ranked) else None

    def next(self, reason: str = "") -> Optional[ScoredListing]:
        """현재 매물을 거절하고 다음 순위로."""
        cur = self.current()
        if cur:
            self.rejected[cur.listing.listing_id] = reason or "마음에 들지 않음"
        self.cursor += 1
        return self.current()

    def previous(self) -> Optional[ScoredListing]:
        if self.cursor > 0:
            self.cursor -= 1
            self.rejected.pop(self.ranked[self.cursor].listing.listing_id, None)
        return self.current()

    def update_requirement(self, req: CustomerRequirement) -> None:
        """피드백으로 조건이 바뀌면 이미 거절한 매물은 빼고 다시 순위를 매김."""
        self.req = req
        self._rerank()

    def top(self, n: int = 5) -> list[ScoredListing]:
        return self.ranked[self.cursor:self.cursor + n]
