"""원본 매물 레코드(엑셀/CSV/텍스트의 한 행)를 LandListing 으로 정규화.

부동산 사무실마다 엑셀 컬럼명이 제각각이므로 별칭(alias) 사전으로 흡수합니다.
부모님 엑셀 양식에 없는 컬럼명이 있으면 COLUMN_ALIASES 에 추가하면 됩니다.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Optional

from .models import JIMOK_TYPES, PYEONG_M2, LandListing, normalize_deal

COLUMN_ALIASES: dict[str, list[str]] = {
    "listing_id": ["매물번호", "번호", "관리번호", "매물id", "id", "no", "매물코드"],
    "address": ["소재지", "주소", "지번주소", "위치", "소재지주소", "address", "물건지", "소재지번", "토지소재지",
                "물건소재지", "대상지", "소재"],
    "sido": ["시도", "시/도"],
    "sigungu": ["시군구", "시/군/구", "시군"],
    "eupmyeondong": ["읍면동", "읍/면/동", "동리", "읍면"],
    "ri": ["리"],
    "jibun": ["지번", "번지"],
    "area_m2": ["면적", "면적(㎡)", "면적㎡", "대지면적", "토지면적", "면적(m2)", "area", "공부면적", "총면적",
                "총면적(㎡)", "합계면적", "면적합계", "면적합계(㎡)", "토지면적(㎡)", "공부면적(㎡)", "대지면적(㎡)",
                "지적면적", "실면적", "면적(m²)"],
    "area_pyeong": ["면적(평)", "평수", "평", "토지평수", "총면적(평)", "총평수", "평수(평)", "환산평수",
                    "대지면적(평)", "토지면적(평)", "대지(평)", "대지평수"],
    "price_manwon": ["매매가", "가격", "금액", "매매가(만원)", "매매금액", "희망가", "price", "총액", "매도가",
                     "매도금액", "매매대금", "총매매가", "총매매금액", "매가", "금액(만원)", "가격(만원)",
                     "매매가격", "총금액", "매매가(원)", "금액(원)", "총액(만원)"],
    "building_area_m2": ["건물면적", "연면적", "건축면적", "공장면적", "건물면적(㎡)", "연면적(㎡)", "건축면적(㎡)",
                         "공장면적(㎡)", "건평(㎡)", "바닥면적"],
    "building_area_pyeong": ["건물면적(평)", "연면적(평)", "건축면적(평)", "공장면적(평)", "건평", "건물평수",
                             "공장평수", "건물(평)"],
    "deposit_manwon": ["보증금", "임대보증금", "보증금(만원)", "보증금(원)"],
    "monthly_rent_manwon": ["월세", "월임대료", "임대료", "월차임", "차임", "월세(만원)", "임대료(만원)", "월임대료(만원)"],
    "property_type": ["매물종류", "물건종류", "매물유형", "물건유형", "부동산종류"],
    "price_per_pyeong": ["평당가", "평단가", "평당가격", "평당(만원)"],
    "jimok_zoning": ["지목/용도", "지목/용도지역", "지목·용도지역", "지목및용도지역", "지목/지역지구", "지목/지역",
                     "지목(용도지역)", "지목/용도지구", "용도지역/지목", "용도/지목"],
    "jimok": ["지목", "토지종류", "지목(공부)", "공부지목", "지목(현황)", "현황지목", "공부상지목", "지목(대장)"],
    "zoning": ["용도지역", "용도", "지역지구", "용도지역지구", "용도지역(지구)", "지역지구등", "지역지구구역",
               "용도지역및지구", "용도지역지구등", "토지이용계획", "용도지역(계획)", "지역지구(용도지역)"],
    "road_access": ["도로", "도로조건", "도로접면", "접도", "진입로", "도로여부", "도로현황", "접면도로"],
    "deal_type": ["거래유형", "거래종류", "거래구분"],
    "status": ["상태", "진행상태", "거래상태", "매물상태"],
    "registered_date": ["등록일", "접수일", "등록일자", "날짜", "date", "작성일"],
    "description": ["특이사항", "비고", "메모", "설명", "상세설명", "내용", "특징", "물건특징", "매물특징", "참고사항"],
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
    hit = _ALIAS_LOOKUP.get(key)
    if hit is None:  # '지역/지구', '지역·지구', '용도 지역:' 처럼 사이에 기호가 들어간 항목 이름
        hit = _ALIAS_LOOKUP.get(re.sub(r"[/·・,.:：\-_]", "", key))
    return hit


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


def parse_lease_manwon(value: Any, won_from: int) -> Optional[int]:
    """보증금·월세 -> 만원. 단위 없이 큰 숫자는 '원' 으로 적은 것으로 봄.
    보증금 65,000,000 / 월세 6,500,000 (원) -> 6500 / 650 (만원). ('6500만원', '650', '6천5백만' 은 그대로)
    won_from: 이 값 이상이면 원 단위 (보증금 100만 = 만원으로는 100억이라 비현실적, 월세 10만 = 만원으로는 월 10억)."""
    if _is_blank(value):
        return None
    has_unit = isinstance(value, str) and re.search(r"[억천만백]", value)
    v = parse_price_manwon(value)
    if v is None:
        return None
    if not has_unit:
        if isinstance(value, (int, float)) and float(value) >= 1e8:
            return v  # parse_price_manwon 이 이미 원 -> 만원으로 바꿈
        if v >= won_from:
            return int(round(v / 10000))
    return v


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


_ZONE = re.compile(r"(관리|녹지|주거|상업|공업|농림|보전)지역$")


def _zoning_token(s: str) -> str:
    s = re.sub(r"\s+", "", s).strip("()[]<>「」『』-")
    if re.match(r"^\d종", s):
        s = "제" + s
    if not s.endswith("지역") and re.search(r"(관리|녹지|주거|상업|공업|농림|보전)$", s):
        s += "지역"
    return s


def zoning_parts(value: Any) -> list[str]:
    """'계획관리지역, 성장관리계획구역' / '계획관리지역\n성장관리계획구역' -> ['계획관리지역', '성장관리계획구역']"""
    if _is_blank(value):
        return []
    tokens = [_zoning_token(t) for t in re.split(r"[,，/·・;\n\r]+|\s{2,}|\s+및\s+", str(value))]
    return list(dict.fromkeys(t for t in tokens if t and t not in {"-", "없음"}))


def normalize_zoning(value: Any) -> str:
    """대표 용도지역 하나: '계획관리' -> '계획관리지역', '2종일반주거' -> '제2종일반주거지역',
    '계획관리지역, 성장관리계획구역' -> '계획관리지역' (나머지 지역·지구는 zoning_parts 로)."""
    parts = zoning_parts(value)
    if not parts:
        return ""
    return next((p for p in parts if _ZONE.search(p)), parts[0])


def split_jimok_zoning(value: Any) -> tuple[str, str]:
    """'대 / 계획관리' -> ('대', '계획관리'), '계획관리지역, 공장용지' -> ('공장용지', '계획관리지역')."""
    if _is_blank(value):
        return "", ""
    jimok, rest = "", []
    for tok in re.split(r"[/,，·・;\n\r]+|\s{2,}|\s+및\s+", str(value)):
        tok = tok.strip()
        if not tok:
            continue
        j = normalize_jimok(tok)
        if not jimok and j in JIMOK_TYPES and not re.search(r"지역|구역|지구", tok):
            jimok = j
        else:
            rest.append(tok)
    return jimok, ", ".join(rest)


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

    building = parse_area_m2(mapped.get("building_area_m2"))
    if building is None:
        building = parse_area_m2(mapped.get("building_area_pyeong"), assume_pyeong=True)
    if area is None and building:  # 공장인데 대지면적 없이 건물면적만 적힌 경우
        area = 0.0

    price = parse_price_manwon(mapped.get("price_manwon"))
    if price is None and area and not _is_blank(mapped.get("price_per_pyeong")):
        ppp = parse_price_manwon(mapped.get("price_per_pyeong"))
        if ppp:
            price = int(round(ppp * area / PYEONG_M2))
    deposit = parse_lease_manwon(mapped.get("deposit_manwon"), won_from=1_000_000)
    rent = parse_lease_manwon(mapped.get("monthly_rent_manwon"), won_from=100_000)

    # 매물 종류·거래 방식: 표의 칸 -> 매물 폴더 이름 (공장/매매임대/…)
    ptype = str(record.get("_매물종류") or "").strip()
    if not _is_blank(mapped.get("property_type")):
        ptype = "공장" if "공장" in str(mapped["property_type"]) else ptype or "토지"
    ptype = ptype if ptype in {"토지", "공장"} else "토지"
    deal_raw = record.get("_거래방식") or ("" if _is_blank(mapped.get("deal_type")) else mapped.get("deal_type"))
    deal = normalize_deal(deal_raw) if deal_raw else ("임대" if price is None and (deposit or rent) else "매매")
    if deal == "매매/임대" and price is None and (deposit or rent):
        deal = "임대"  # 매매가가 비어 있으면 임대만
    if deal == "임대" and price is None:
        price = 0  # 임대 전용 매물은 매매가가 없음

    if not address:
        errors.append("소재지(주소) 없음")
    if area is None:
        errors.append(f"면적 해석 불가: {mapped.get('area_m2', mapped.get('area_pyeong'))!r}")
    if price is None:
        errors.append(f"가격 해석 불가: {mapped.get('price_manwon')!r}")
    if deal == "임대" and not (deposit or rent) and not errors:
        errors.append("임대 매물인데 보증금·월세가 없음")
    if errors:
        return None, errors

    listing_id = "" if _is_blank(mapped.get("listing_id")) else str(mapped["listing_id"]).strip()
    if listing_id.endswith(".0"):  # 엑셀 숫자형 번호
        listing_id = listing_id[:-2]
    # 엑셀에 적힌 매물번호 (화면 표시용). 매물 폴더 방식은 수집기가 따로 넘겨줌 (자동 번호와 구분)
    extra.pop("_파일매물번호", None)
    extra.pop("_매물종류", None)
    extra.pop("_거래방식", None)
    for key in ("_매물특징", "_용도태그", "_지목구성", "_용도지역구성"):  # 매물 폴더 수집기가 넘긴 표시용 값 (빈 값도 유지)
        if key in record:
            extra[key] = "" if _is_blank(record[key]) else str(record[key])
    file_no = str(record.get("_파일매물번호") or "").strip() if "_파일매물번호" in record else listing_id
    if file_no:
        extra["매물번호"] = file_no

    if not _is_blank(mapped.get("jimok_zoning")):  # '지목/용도' 한 칸에 '대 / 계획관리지역, 성장관리계획구역'
        j, z = split_jimok_zoning(mapped["jimok_zoning"])
        if j and _is_blank(mapped.get("jimok")):
            mapped["jimok"] = j
        if z and _is_blank(mapped.get("zoning")):
            mapped["zoning"] = z
    zones = zoning_parts(mapped.get("zoning"))
    if len(zones) > 1:  # '계획관리지역, 성장관리계획구역' -> 대표 용도지역 + 전체 지역·지구 (화면에 같이 보여줌)
        extra["_지역지구"] = ", ".join(zones)
    listing = LandListing(
        listing_id=listing_id or make_listing_id(address, area),
        address=address,
        area_m2=round(area, 2),
        price_manwon=price,
        jimok=normalize_jimok(mapped.get("jimok")),
        zoning=normalize_zoning(mapped.get("zoning")),
        road_access="" if _is_blank(mapped.get("road_access")) else str(mapped["road_access"]).strip(),
        deal_type=deal,
        status=normalize_status(mapped.get("status")),
        registered_date=parse_date(mapped.get("registered_date")),
        description="" if _is_blank(mapped.get("description")) else str(mapped["description"]).strip(),
        owner_contact="" if _is_blank(mapped.get("owner_contact")) else str(mapped["owner_contact"]).strip(),
        lat=parse_float(mapped.get("lat")),
        lon=parse_float(mapped.get("lon")),
        source=default_source if _is_blank(mapped.get("source")) else str(mapped["source"]).strip(),
        source_file=source_file,
        extra=extra,
        property_type=ptype,
        building_area_m2=round(building, 2) if building else None,
        deposit_manwon=deposit,
        monthly_rent_manwon=rent,
    )
    return listing, []
