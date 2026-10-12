"""매물 정제: 오류 매물 필터링 + 중복 매물 제거.

오류(ERROR)  -> 추천 대상에서 제외 (주소/면적/가격 누락, 비현실적인 값)
경고(WARN)   -> 추천은 하되 보고서에 표시 (알 수 없는 지목, 오래된 매물 등)
중복         -> 같은 땅이 여러 파일/플랫폼에 올라온 경우 하나만 남김
               (가장 최근 등록 + 정보가 가장 많은 레코드를 대표로, 출처는 모두 기록)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher

from .collectors.base import RawRecord
from .models import JIMOK_TYPES, LandListing
from .normalizer import map_column, normalize_address, record_to_listing

# 토지 평당가 허용 범위 (만원/평). 이 범위를 벗어나면 단위 착오(원/만원, ㎡/평)일 가능성이 큼.
MIN_PRICE_PER_PYEONG = 1  # 1만원/평 미만
MAX_PRICE_PER_PYEONG = 30000  # 3억원/평 초과
MIN_AREA_M2 = 10
MAX_AREA_M2 = 5_000_000
STALE_DAYS = 365 * 2


@dataclass
class QualityIssue:
    level: str  # ERROR / WARN / DUPLICATE
    listing_id: str
    message: str
    source_file: str = ""


@dataclass
class CleanResult:
    listings: list[LandListing]
    issues: list[QualityIssue] = field(default_factory=list)
    total_raw: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    @property
    def duplicate_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "DUPLICATE")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARN")

    def report(self) -> str:
        lines = [
            f"원본 레코드 {self.total_raw}건 -> 정상 매물 {len(self.listings)}건 "
            f"(오류 제외 {self.error_count}건, 중복 제거 {self.duplicate_count}건, 경고 {self.warning_count}건)"
        ]
        for i in self.issues:
            src = f" ({i.source_file.split('/')[-1].split(chr(92))[-1]})" if i.source_file else ""
            lines.append(f"  [{i.level}] {i.listing_id}: {i.message}{src}")
        return "\n".join(lines)


def validate(listing: LandListing, today: date | None = None) -> tuple[list[str], list[str]]:
    """(errors, warnings)"""
    today = today or date.today()
    errors, warnings = [], []
    if listing.area_m2 < MIN_AREA_M2 or listing.area_m2 > MAX_AREA_M2:
        errors.append(f"면적 비정상 ({listing.area_m2:,.1f}㎡)")
    if listing.price_manwon <= 0:
        errors.append(f"가격 비정상 ({listing.price_manwon})")
    elif listing.area_m2 > 0:
        ppp = listing.price_per_pyeong
        if ppp < MIN_PRICE_PER_PYEONG or ppp > MAX_PRICE_PER_PYEONG:
            errors.append(f"평당가 비현실적 ({ppp:,.1f}만원/평) - 단위 입력 오류 의심")
    if not re.search(r"(시|군|구)\b|(시|군|구)\s", listing.address + " "):
        warnings.append("주소에 시/군/구 정보 없음")
    if not re.search(r"\d", listing.address):
        warnings.append("주소에 지번 없음")
    if listing.jimok and listing.jimok not in JIMOK_TYPES:
        warnings.append(f"알 수 없는 지목 '{listing.jimok}'")
    if not listing.jimok:
        warnings.append("지목 미기재")
    if listing.registered_date:
        if listing.registered_date > today + timedelta(days=1):
            warnings.append(f"등록일이 미래 ({listing.registered_date})")
        elif (today - listing.registered_date).days > STALE_DAYS:
            warnings.append(f"오래된 매물 ({listing.registered_date}) - 거래 가능 여부 확인 필요")
    if listing.lat is not None and not (33.0 <= listing.lat <= 38.7):
        warnings.append("위도가 대한민국 범위를 벗어남")
        listing.lat = None
    if listing.lon is not None and not (124.5 <= listing.lon <= 132.0):
        warnings.append("경도가 대한민국 범위를 벗어남")
        listing.lon = None
    return errors, warnings


def _completeness(l: LandListing) -> int:
    return sum(bool(v) for v in (l.jimok, l.zoning, l.road_access, l.description, l.owner_contact,
                                 l.registered_date, l.lat))


def _jibun(addr: str) -> str:
    addr = re.sub(r"외\s*\d+\s*필지.*$", "", addr)  # '417-2 외 36필지' 의 36 은 지번이 아님
    m = re.findall(r"(산?\d+(?:-\d+)?)", addr)
    return m[-1] if m else ""


def is_duplicate(a: LandListing, b: LandListing) -> bool:
    na, nb = normalize_address(a.address), normalize_address(b.address)
    area_close = abs(a.area_m2 - b.area_m2) <= max(1.0, 0.01 * max(a.area_m2, b.area_m2))
    if na == nb and area_close:
        return True
    # 표기만 조금 다른 같은 필지 (예: '양평군 양서면 목왕리 123-4' vs '경기 양평 양서 목왕리 123-4')
    if _jibun(a.address) and _jibun(a.address) == _jibun(b.address) and area_close:
        if SequenceMatcher(None, na, nb).ratio() >= 0.75:
            return True
    return False


def _pick_primary(group: list[LandListing]) -> LandListing:
    return max(group, key=lambda l: (l.registered_date or date.min, _completeness(l)))


def deduplicate(listings: list[LandListing]) -> tuple[list[LandListing], list[QualityIssue]]:
    groups: list[list[LandListing]] = []
    # 같은 지번끼리만 비교해 O(n^2) 를 줄임
    buckets: dict[str, list[int]] = {}
    for l in listings:
        key = _jibun(l.address) or normalize_address(l.address)
        placed = False
        for gi in buckets.get(key, []):
            if any(is_duplicate(l, other) for other in groups[gi]):
                groups[gi].append(l)
                placed = True
                break
        if not placed:
            groups.append([l])
            buckets.setdefault(key, []).append(len(groups) - 1)

    result, issues = [], []
    for g in groups:
        primary = _pick_primary(g)
        if len(g) > 1:
            sources = sorted({x.source for x in g})
            primary.extra["duplicate_sources"] = sources
            primary.extra["duplicate_ids"] = [x.listing_id for x in g if x is not primary]
            prices = {x.price_manwon for x in g}
            for dup in g:
                if dup is primary:
                    continue
                msg = f"'{primary.listing_id}' 와 중복 -> 제거"
                if dup.price_manwon != primary.price_manwon:
                    msg += f" (가격 상이: {dup.price_manwon:,} vs {primary.price_manwon:,}만원, 최신 정보 사용)"
                issues.append(QualityIssue("DUPLICATE", dup.listing_id, msg, dup.source_file))
            if len(prices) > 1:
                primary.extra["price_history"] = sorted(prices)
        result.append(primary)
    return result, issues


def clean(records: list[RawRecord], today: date | None = None) -> CleanResult:
    issues: list[QualityIssue] = []
    valid: list[LandListing] = []
    seen_ids: dict[str, int] = {}
    for idx, rec in enumerate(records):
        listing, errs = record_to_listing(rec.data, rec.source_file, rec.source)
        if listing is None:
            raw_id = next((str(v) for k, v in rec.data.items() if map_column(k) == "listing_id" and v), "")
            issues.append(QualityIssue("ERROR", raw_id or f"row#{idx + 1}", "; ".join(errs), rec.source_file))
            continue
        errors, warnings = validate(listing, today)
        if errors:
            issues.append(QualityIssue("ERROR", listing.listing_id, "; ".join(errors), rec.source_file))
            continue
        for w in warnings:
            issues.append(QualityIssue("WARN", listing.listing_id, w, rec.source_file))
        # 서로 다른 매물이 같은 번호를 쓰는 경우 번호를 구분
        if listing.listing_id in seen_ids:
            seen_ids[listing.listing_id] += 1
            listing.listing_id = f"{listing.listing_id}-{seen_ids[listing.listing_id]}"
        else:
            seen_ids[listing.listing_id] = 1
        valid.append(listing)

    deduped, dup_issues = deduplicate(valid)
    issues.extend(dup_issues)
    return CleanResult(listings=deduped, issues=issues, total_raw=len(records))
