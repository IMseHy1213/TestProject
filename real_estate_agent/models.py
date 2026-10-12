"""도메인 모델: 토지 매물, 손님 요구사항, 추천 결과."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Optional

PYEONG_M2 = 3.305785  # 1평 = 3.305785㎡

# 지목 28종 (공간정보관리법) 중 토지 매물에서 쓰이는 정식 명칭
JIMOK_TYPES = [
    "전", "답", "과수원", "목장용지", "임야", "광천지", "염전", "대", "공장용지", "학교용지",
    "주차장", "주유소용지", "창고용지", "도로", "철도용지", "제방", "하천", "구거", "유지",
    "양어장", "수도용지", "공원", "체육용지", "유원지", "종교용지", "사적지", "묘지", "잡종지",
]


@dataclass
class LandListing:
    listing_id: str
    address: str  # 소재지 전체 (예: 경기도 양평군 양서면 목왕리 123-4)
    area_m2: float
    price_manwon: int  # 매매가 (만원)
    jimok: str = ""  # 지목 (전/답/임야/대 ...)
    zoning: str = ""  # 용도지역 (계획관리지역, 자연녹지지역 ...)
    road_access: str = ""  # 도로 조건 (예: 4m 포장도로 접함 / 맹지)
    deal_type: str = "매매"
    status: str = "거래가능"  # 거래가능 / 보류 / 계약완료
    registered_date: Optional[date] = None
    description: str = ""
    owner_contact: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: str = "local"  # local / serve / naver / public ...
    source_file: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def area_pyeong(self) -> float:
        return self.area_m2 / PYEONG_M2

    @property
    def price_per_pyeong(self) -> float:
        return self.price_manwon / self.area_pyeong if self.area_m2 > 0 else 0.0

    @property
    def is_blind_land(self) -> bool:
        """맹지 여부."""
        return "맹지" in self.road_access or "맹지" in self.description

    def summary(self) -> str:
        return (
            f"[{self.listing_id}] {self.address} | {self.jimok or '-'} | {self.zoning or '-'} | "
            f"{self.area_m2:,.0f}㎡({self.area_pyeong:,.0f}평) | {format_price(self.price_manwon)} "
            f"(평당 {self.price_per_pyeong:,.0f}만원) | 도로: {self.road_access or '-'}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["registered_date"] = self.registered_date.isoformat() if self.registered_date else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LandListing":
        d = dict(d)
        if d.get("registered_date"):
            d["registered_date"] = date.fromisoformat(d["registered_date"])
        return cls(**d)


@dataclass
class CustomerRequirement:
    """손님의 문의/요구사항을 구조화한 결과."""

    customer_name: str = ""
    regions: list[str] = field(default_factory=list)  # 선호 지역 (시/군/읍/면/리)
    jimok: list[str] = field(default_factory=list)  # 원하는 지목
    zoning: list[str] = field(default_factory=list)  # 원하는 용도지역
    budget_min: Optional[int] = None  # 만원
    budget_max: Optional[int] = None  # 만원
    area_min_m2: Optional[float] = None
    area_max_m2: Optional[float] = None
    purpose: str = ""  # 전원주택 / 창고 / 농사 / 투자 / 공장 / 주말농장 ...
    need_road: bool = False  # 맹지 제외
    keywords: list[str] = field(default_factory=list)  # 남향, 계곡, IC 인근 ...
    excluded_keywords: list[str] = field(default_factory=list)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)  # 해석 중 참고사항 (예: 취급하지 않는 매물 종류)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustomerRequirement":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def describe(self) -> str:
        parts = []
        if self.regions:
            parts.append("지역=" + ",".join(self.regions))
        if self.jimok:
            parts.append("지목=" + ",".join(self.jimok))
        if self.zoning:
            parts.append("용도지역=" + ",".join(self.zoning))
        if self.budget_min or self.budget_max:
            lo = format_price(self.budget_min) if self.budget_min else "-"
            hi = format_price(self.budget_max) if self.budget_max else "-"
            parts.append(f"예산={lo}~{hi}")
        if self.area_min_m2 or self.area_max_m2:
            lo = f"{self.area_min_m2 / PYEONG_M2:,.0f}평" if self.area_min_m2 else "-"
            hi = f"{self.area_max_m2 / PYEONG_M2:,.0f}평" if self.area_max_m2 else "-"
            parts.append(f"면적={lo}~{hi}")
        if self.purpose:
            parts.append("목적=" + self.purpose)
        if self.need_road:
            parts.append("도로접함 필수")
        if self.keywords:
            parts.append("키워드=" + ",".join(self.keywords))
        if self.excluded_keywords:
            parts.append("제외=" + ",".join(self.excluded_keywords))
        return " / ".join(parts) or "(조건 없음)"


@dataclass
class ScoredListing:
    listing: LandListing
    score: float
    reasons: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    factors: dict[str, float] = field(default_factory=dict)  # 항목별 충족도 0~1 (학습에 사용)


def format_price(manwon: Optional[int]) -> str:
    """만원 단위 금액을 '3억 2,000만원' 형태로."""
    if manwon is None:
        return "-"
    eok, man = divmod(int(manwon), 10000)
    if eok and man:
        return f"{eok}억 {man:,}만원"
    if eok:
        return f"{eok}억원"
    return f"{man:,}만원"
