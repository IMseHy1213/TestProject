"""도메인 모델: 매물(토지·공장), 손님 요구사항, 추천 결과."""

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


# 매물 종류 (매물 폴더: 매물/토지/…, 매물/공장/매매|매매임대|임대/음성군/…)
PROPERTY_TYPES = ["토지", "공장"]
DEAL_TYPES = ["매매", "매매/임대", "임대"]


def normalize_deal(value: Any) -> str:
    """'매매' / '매매임대', '매매·임대', '매매 또는 임대' -> '매매/임대' / '임대', '월세' -> '임대'."""
    s = str(value or "")
    sale = "매매" in s or "매도" in s
    lease = any(w in s for w in ("임대", "월세", "전세", "렌트"))
    if sale and lease:
        return "매매/임대"
    if lease:
        return "임대"
    return "매매"


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
    property_type: str = "토지"  # 토지 / 공장
    building_area_m2: Optional[float] = None  # 공장: 건물(연)면적
    deposit_manwon: Optional[int] = None  # 임대 보증금 (만원)
    monthly_rent_manwon: Optional[int] = None  # 임대 월세 (만원)

    @property
    def for_sale(self) -> bool:
        """살 수 있는 매물 (매매, 매매/임대)."""
        return self.deal_type != "임대"

    @property
    def for_lease(self) -> bool:
        """빌릴 수 있는 매물 (임대, 매매/임대)."""
        return "임대" in self.deal_type

    @property
    def building_area_pyeong(self) -> Optional[float]:
        return self.building_area_m2 / PYEONG_M2 if self.building_area_m2 else None

    @property
    def lease_text(self) -> str:
        """'보증금 5,000만원 / 월 300만원'"""
        parts = []
        if self.deposit_manwon:
            parts.append(f"보증금 {format_price(self.deposit_manwon)}")
        if self.monthly_rent_manwon:
            parts.append(f"월 {format_price(self.monthly_rent_manwon)}")
        return " / ".join(parts) or "임대 조건 상담"

    @property
    def price_text(self) -> str:
        """화면·답변에 쓰는 가격: 매매 '3억원', 임대 '보증금 5,000만원 / 월 300만원', 둘 다면 '매매 … · 임대 …'."""
        if self.deal_type == "임대":
            return f"임대 {self.lease_text}"
        if self.deal_type == "매매/임대":
            lease = self.lease_text if (self.deposit_manwon or self.monthly_rent_manwon) else "조건 상담"
            return f"매매 {format_price(self.price_manwon)} · 임대 {lease}"
        return format_price(self.price_manwon)

    @property
    def kind_label(self) -> str:
        """'토지' / '공장(매매)' / '공장(임대)' / '공장(매매/임대)'"""
        return self.property_type if self.property_type == "토지" else f"{self.property_type}({self.deal_type})"

    @property
    def area_pyeong(self) -> float:
        return self.area_m2 / PYEONG_M2

    @property
    def price_per_pyeong(self) -> float:
        return self.price_manwon / self.area_pyeong if self.area_m2 > 0 and self.for_sale else 0.0

    @property
    def office_no(self) -> str:
        """매물 자료(엑셀)에 적힌 매물번호. 없으면 빈 문자열 (프로그램이 붙인 번호는 표시하지 않음)."""
        return str(self.extra.get("매물번호", "") or "")

    @property
    def features(self) -> str:
        """화면의 '매물 특징': '● 대소IC-15분 거리(중부 고속도로) / ● 25톤 트럭 진입'"""
        if "_매물특징" in self.extra:
            return str(self.extra["_매물특징"] or "")
        from .collectors.listing_folder import feature_lines, format_features

        return format_features(feature_lines(self.description))

    @property
    def use_tag(self) -> str:
        """매물 폴더 이름의 괄호 안 용도 (예: '물류부지', '상가 부지')."""
        return str(self.extra.get("_용도태그", "") or "")

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
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if d.get("registered_date"):
            d["registered_date"] = date.fromisoformat(d["registered_date"])
        return cls(**d)


@dataclass
class CustomerRequirement:
    """손님의 문의/요구사항을 구조화한 결과."""

    customer_name: str = ""
    regions: list[str] = field(default_factory=list)  # 선호 지역 (시/군/읍/면/리)
    excluded_regions: list[str] = field(default_factory=list)  # 빼고 찾을 지역 ('음성군 말고 다른 지역')
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
    property_types: list[str] = field(default_factory=list)  # 찾는 매물 종류 (['공장'], ['토지']; 비면 전체)
    deal: str = ""  # '매매' (사려는 것) / '임대' (빌리려는 것) / '' (상관없음)
    deposit_max: Optional[int] = None  # 임대 보증금 상한 (만원)
    rent_max: Optional[int] = None  # 월세 상한 (만원)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustomerRequirement":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def describe(self) -> str:
        parts = []
        if self.property_types:
            parts.append("종류=" + ",".join(self.property_types))
        if self.deal:
            parts.append("거래=" + self.deal)
        if self.regions:
            parts.append("지역=" + ",".join(self.regions))
        if self.excluded_regions:
            parts.append("제외지역=" + ",".join(self.excluded_regions))
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
        if self.deposit_max:
            parts.append(f"보증금={format_price(self.deposit_max)} 이하")
        if self.rent_max:
            parts.append(f"월세={format_price(self.rent_max)} 이하")
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
