"""원본 매물 레코드(엑셀/CSV/텍스트의 한 행)를 LandListing 으로 정규화.

부동산 사무실마다 엑셀 컬럼명이 제각각이므로 별칭(alias) 사전으로 흡수합니다.
부모님 엑셀 양식에 없는 컬럼명이 있으면 COLUMN_ALIASES 에 추가하면 됩니다.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Optional

from .models import JIMOK_TYPES, PYEONG_M2, LandListing

COLUMN_ALIASES: dict[str, list[str]] = {
    "listing_id": ["매물번호", "번호", "관리번호", "매물id", "id", "no", "매물코드"],
    "address": ["소재지", "주소", "지번주소", "위치", "소재지주소", "address", "물건지"],
    "sido": ["시도", "시/도"],
    "sigungu": ["시군구", "시/군/구", "시군"],
    "eupmyeondong": ["읍면동", "읍/면/동", "동리", "읍면"],
    "ri": ["리"],
    "jibun": ["지번", "번지"],
    "area_m2": ["면적", "면적(㎡)", "면적㎡", "대지면적", "토지면적", "면적(m2)", "area", "공부면적"],
    "area_pyeong": ["면적(평)", "평수", "평", "토지평수"],
    "price_manwon": ["매매가", "가격", "금액", "매매가(만원)", "매매금액", "희망가", "price", "총액"],
    "price_per_pyeong": ["평당가", "평단가", "평당가격", "평당(만원)"],
    "jimok": ["지목", "토지종류"],
    "zoning": ["용도지역", "용도", "지역지구"],
    "road_access": ["도로", "도로조건", "도로접면", "접도", "진입로", "도로여부"],
    "deal_type": ["거래유형", "거래종류", "거래구분"],
    "status": ["상태", "진행상태", "거래상태", "매물상태"],
    "registered_date": ["등록일", "접수일", "등록일자", "날짜", "date", "작성일"],
    "description": ["특이사항", "비고", "메모", "설명", "상세설명", "내용", "특징"],
    "owner_contact": ["연락처", "소유자연락처", "매도인연락처", "전화번호", "소유주"],
    "lat": ["위도", "lat", "latitude", "y좌표"],
    "lon": ["경도", "lon", "lng", "longitude", "x좌표"],
    "source": ["출처", "매물출처", "source"],
}

_ALIAS_LOOKUP = {
    re.sub(r"\s+", "", alias).lower(): field_name
    for field_name, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}

JIMOK_ALIASES = {
    "밭": "전", "논": "답", "산": "임야", "대지": "대", "공장": "공장용지", "창고": "창고용지",
    "과원": "과수원", "목장": "목장용지", "주차장용지": "주차장",
}

STATUS_MAP = {
    "계약완료": "계약완료", "거래완료": "계약완료", "완료": "계약완료", "매매완료": "계약완료", "sold": "계약완료",
    "보류": "보류", "hold": "보류",
}


def map_column(name: Any) -> Optional[str]:
    key = re.sub(r"\s+", "", str(name)).lower()
    return _ALIAS_LOOKUP.get(key)


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    return str(v).strip() in {"", "-", "nan", "NaN", "None", "null"}


def parse_price_manwon(value: Any) -> Optional[int]:
    """'3억 2천만원', '3.2억', '32,000', '32000만원', '3억2000' -> 만원 단위 정수.

    단위 없는 숫자는 만원으로 간주하되, 1억 이상으로 보이는 큰 수(>= 10억 '원')는 원 단위로 봅니다.
    """
    if _is_blank(value):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return int(round(v / 10000)) if v >= 1e8 else int(round(v))
    s = str(value).replace(",", "").replace(" ", "").replace("원", "")
    if not s:
        return None
    total = 0.0
    matched = False
    m = re.search(r"([\d.]+)억", s)
    if m:
        total += float(m.group(1)) * 10000
        matched = True
        s = s[m.end():]
    m = re.search(r"([\d.]+)천", s)
    if m:
        total += float(m.group(1)) * 1000
        matched = True
        s = s[m.end():]
    m = re.search(r"([\d.]+)백", s)
    if m:
        total += float(m.group(1)) * 100
        matched = True
        s = s[m.end():]
    m = re.search(r"([\d.]+)", s.replace("만", ""))
    if m:
        num = float(m.group(1))
        if not matched and num >= 1e8:
            num /= 10000
        total += num
        matched = True
    return int(round(total)) if matched else None


def parse_area_m2(value: Any, assume_pyeong: bool = False) -> Optional[float]:
    """'330㎡', '330m2', '100평', '1,000' -> ㎡."""
    if _is_blank(value):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v * PYEONG_M2 if assume_pyeong else v
    s = str(value).replace(",", "").strip()
    m = re.search(r"([\d.]+)", s)
    if not m:
        return None
    num = float(m.group(1))
    if "평" in s or (assume_pyeong and not re.search(r"㎡|m2|m²|제곱", s, re.I)):
        return num * PYEONG_M2
    return num


def parse_date(value: Any) -> Optional[date]:
    if _is_blank(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        return value.to_pydatetime().date()
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d", "%y.%m.%d", "%y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.rstrip("."), fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_float(value: Any) -> Optional[float]:
    if _is_blank(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def normalize_jimok(value: Any) -> str:
    if _is_blank(value):
        return ""
    s = str(value).strip()
    if s in JIMOK_TYPES:
        return s
    if s in JIMOK_ALIASES:
        return JIMOK_ALIASES[s]
    for j in sorted(JIMOK_TYPES, key=len, reverse=True):  # '임야(보전)' 같은 표기
        if s.startswith(j):
            return j
    return s  # 알 수 없는 값은 그대로 두고 cleaner 가 경고로 처리


def normalize_zoning(value: Any) -> str:
    """'계획관리' -> '계획관리지역', '2종일반주거' -> '제2종일반주거지역'."""
    if _is_blank(value):
        return ""
    s = re.sub(r"\s+", "", str(value))
    if re.match(r"^\d종", s):
        s = "제" + s
    if not s.endswith("지역") and re.search(r"(관리|녹지|주거|상업|공업|농림|보전)$", s):
        s += "지역"
    return s


def normalize_status(value: Any) -> str:
    if _is_blank(value):
        return "거래가능"
    s = re.sub(r"\s+", "", str(value)).lower()
    for k, v in STATUS_MAP.items():
        if k in s:
            return v
    return "거래가능"


def normalize_address(addr: str) -> str:
    """중복 판단용 주소 정규화: 공백/특수문자 제거, 광역시도 약칭 통일."""
    a = re.sub(r"\s+", " ", addr or "").strip()
    replacements = {
        "경기도": "경기", "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북", "충청남도": "충남",
        "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
        "제주특별자치도": "제주", "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
        "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산",
        "세종특별자치시": "세종",
    }
    for full, short in replacements.items():
        a = a.replace(full, short)
    a = re.sub(r"번지|\(.*?\)", "", a)
    return re.sub(r"[\s,]", "", a)


def make_listing_id(address: str, area_m2: float) -> str:
    h = hashlib.sha1(f"{normalize_address(address)}|{area_m2:.0f}".encode()).hexdigest()[:10]
    return f"L-{h}"


def record_to_listing(record: dict[str, Any], source_file: str = "", default_source: str = "local") -> tuple[
    Optional[LandListing], list[str]
]:
    """원본 레코드 하나를 LandListing 으로 변환. (listing 또는 None, 오류 메시지 목록) 반환."""
    mapped: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in record.items():
        f = map_column(k)
        if f and f not in mapped:
            mapped[f] = v
        elif not _is_blank(v):
            extra[str(k)] = v if isinstance(v, (str, int, float, bool)) else str(v)

    errors: list[str] = []

    address = "" if _is_blank(mapped.get("address")) else str(mapped["address"]).strip()
    if not address:  # 시도/시군구/읍면동/지번이 분리된 양식
        parts = [mapped.get(p) for p in ("sido", "sigungu", "eupmyeondong", "ri", "jibun")]
        address = " ".join(str(p).strip() for p in parts if not _is_blank(p))
    elif not _is_blank(mapped.get("jibun")) and str(mapped["jibun"]).strip() not in address:
        address = f"{address} {str(mapped['jibun']).strip()}"

    area = parse_area_m2(mapped.get("area_m2"))
    if area is None:
        area = parse_area_m2(mapped.get("area_pyeong"), assume_pyeong=True)

    price = parse_price_manwon(mapped.get("price_manwon"))
    if price is None and area and not _is_blank(mapped.get("price_per_pyeong")):
        ppp = parse_price_manwon(mapped.get("price_per_pyeong"))
        if ppp:
            price = int(round(ppp * area / PYEONG_M2))

    if not address:
        errors.append("소재지(주소) 없음")
    if area is None:
        errors.append(f"면적 해석 불가: {mapped.get('area_m2', mapped.get('area_pyeong'))!r}")
    if price is None:
        errors.append(f"가격 해석 불가: {mapped.get('price_manwon')!r}")
    if errors:
        return None, errors

    listing_id = "" if _is_blank(mapped.get("listing_id")) else str(mapped["listing_id"]).strip()
    if listing_id.endswith(".0"):  # 엑셀 숫자형 번호
        listing_id = listing_id[:-2]

    listing = LandListing(
        listing_id=listing_id or make_listing_id(address, area),
        address=address,
        area_m2=round(area, 2),
        price_manwon=price,
        jimok=normalize_jimok(mapped.get("jimok")),
        zoning=normalize_zoning(mapped.get("zoning")),
        road_access="" if _is_blank(mapped.get("road_access")) else str(mapped["road_access"]).strip(),
        deal_type="매매" if _is_blank(mapped.get("deal_type")) else str(mapped["deal_type"]).strip(),
        status=normalize_status(mapped.get("status")),
        registered_date=parse_date(mapped.get("registered_date")),
        description="" if _is_blank(mapped.get("description")) else str(mapped["description"]).strip(),
        owner_contact="" if _is_blank(mapped.get("owner_contact")) else str(mapped["owner_contact"]).strip(),
        lat=parse_float(mapped.get("lat")),
        lon=parse_float(mapped.get("lon")),
        source=default_source if _is_blank(mapped.get("source")) else str(mapped["source"]).strip(),
        source_file=source_file,
        extra=extra,
    )
    return listing, []
