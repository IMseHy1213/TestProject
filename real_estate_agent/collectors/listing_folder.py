"""'매물 하나 = 폴더 하나' 구조 읽기.

    매물/
    └─ 음성군/                                              <- 지역 폴더 (주소에 지역이 없으면 앞에 붙임)
        └─ 음성군 감곡면 문촌리 417-2 외 36필지(물류부지)/    <- 매물 폴더 (이름 = 주소, 괄호 = 용도/특징)
            └─ 음성군 감곡면 문촌리 417-2 외 36필지.xlsx      <- 매물 정보 엑셀

폴더(또는 파일) 이름이 '○○리 123-4' 처럼 지번이 들어간 주소 모양이면 그 안의 엑셀 전체를 '매물 1건'으로 봅니다.
엑셀 안의 내용은 다음 형태를 모두 읽습니다.
  1) 필지 목록표   : 지번 | 지목 | 면적 | (용도지역 | 금액) ... 여러 줄 + 합계 줄  -> 면적은 합산, 지목은 면적이 가장 큰 것
  2) 물건 정보지   : '소재지 | 값', '매매가 | 값' 처럼 항목 옆(또는 아래)에 값이 있는 양식
  3) 한 줄짜리 표  : 일반 장부 양식에 매물 1줄
총 매매가는 '매매가/총액/매도가' 항목 -> 합계 줄 -> 필지별 금액 합계 순서로 찾습니다.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from ..normalizer import map_column, normalize_jimok, normalize_zoning, parse_area_m2, parse_price_manwon

# '감곡면 문촌리 417-2', '금왕읍 무극리 산 12', '○○동 123' 처럼 동/리 + 지번
_LISTING_NAME = re.compile(r"[가-힣]+(리|동|가)\s*(산\s*)?\d+(-\d+)?")
_TOTAL_ROW = re.compile(r"^\s*(합\s*계|총\s*계|소\s*계|계|total)\s*$", re.I)
_MULTI_PARCEL = re.compile(r"외\s*(\d+)\s*필지")


_REGION_NAME = re.compile(r"^[가-힣]{1,8}(특별시|광역시|특별자치시|특별자치도|도|시|군|구)$")


def is_region_name(name: str) -> bool:
    """'음성군', '청주시', '충청북도' 는 지역 이름, '기타지역', '2026년', '정리중' 은 아님."""
    return bool(_REGION_NAME.match(name.strip())) and "기타" not in name


def _complete_address(address: str, sheet_address: str) -> str:
    """폴더 이름 주소('감곡면 문촌리 417-2 외 36필지')에 시/군이 없으면 엑셀 소재지('충북 음성군 감곡면 …')에서 보충."""
    if not sheet_address or re.search(r"[가-힣]+(시|군|구)(\s|$)", address):
        return address
    first = address.split()[0] if address.split() else ""
    idx = sheet_address.find(first) if first else -1
    if idx > 0:
        return f"{sheet_address[:idx].strip()} {address}"
    return address


def looks_like_listing_name(name: str) -> bool:
    return bool(_LISTING_NAME.search(name))


def split_name(name: str) -> tuple[str, list[str]]:
    """'음성군 감곡면 문촌리 417-2 외 36필지(물류부지)' -> ('음성군 감곡면 문촌리 417-2 외 36필지', ['물류부지'])"""
    tags = [t.strip() for t in re.findall(r"[(\[（]([^)\]）]+)[)\]）]", name) if t.strip()]
    base = re.sub(r"\s*[(\[（][^)\]）]*[)\]）]\s*", " ", name)
    return re.sub(r"\s+", " ", base).strip(), tags


# ---------------------------------------------------------------- 엑셀/CSV -> 셀 격자
def read_grids(path: Path) -> list[list[list[Any]]]:
    """시트마다 2차원 셀 목록."""
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        return [df.where(df.notna(), None).values.tolist() for df in sheets.values()]
    if ext == ".csv":
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                with open(path, encoding=enc, newline="") as f:
                    return [[[c if c != "" else None for c in r] for r in csv.reader(f)]]
            except UnicodeDecodeError:
                continue
    return []


def _text(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def _label(v: Any) -> Optional[str]:
    t = _text(v)
    if not t or len(t) > 20:
        return None
    return map_column(t.rstrip(":：").strip())


# ---------------------------------------------------------------- 격자 해석
def _is_header_row(row: list[Any]) -> bool:
    """표의 머리글 줄인지: 항목 이름이 2개 이상이고, 그 오른쪽이 대부분 값이 아니라 다른 항목/빈칸."""
    labels = [i for i, c in enumerate(row) if _label(c)]
    if len(labels) < 2:
        return False
    followed_by_value = sum(
        1 for i in labels if i + 1 < len(row) and _text(row[i + 1]) and not _label(row[i + 1])
    )
    return followed_by_value * 2 < len(labels)  # '소재지 | 값 | 면적 | 값' 같은 정보지 줄은 머리글이 아님


_UNIT = re.compile(r"^\s*(㎡|m2|m²|평|원|만원|천원|%|m|kw|kW|톤)\s*$")


def _merge_unit_row(header: list[str], units: list[Any]) -> list[str]:
    """머리글 아래 단위 줄 ('면적' 아래 '㎡ | 평') -> ['면적(㎡)', '면적(평)']: 빈 머리글은 왼쪽 머리글 이름을 이어받음."""
    out = list(header)
    for j, u in enumerate(units[:len(out)] if len(units) >= len(out) else list(units) + [None] * (len(out) - len(units))):
        unit = _text(u)
        if not unit:
            continue
        base = out[j] or next((out[k] for k in range(j - 1, -1, -1) if out[k]), "")
        base = re.sub(r"\(.*?\)$", "", base).strip()
        if base:
            out[j] = f"{base}({unit})"
    return out


def extract_table(grid: list[list[Any]]) -> tuple[list[dict], Optional[dict], set[int]]:
    """(필지/매물 줄 목록, 합계 줄, 표에 쓰인 행 번호)
    - 머리글 바로 아래 단위 줄('㎡ | 평')은 머리글에 합침
    - 합계 줄에서 표가 끝남 (그 아래 '건물' 목록은 필지가 아님)"""
    for h, row in enumerate(grid[:30]):
        if not _is_header_row(row):
            continue
        header = [_text(c) for c in row]
        rows, total, used = [], None, {h}
        start = h + 1
        if start < len(grid):
            nxt = [_text(c) for c in grid[start]]
            if any(nxt) and all(not t or _UNIT.match(t) for t in nxt):
                header = _merge_unit_row(header, grid[start])
                used.add(start)
                start += 1
        for r in range(start, len(grid)):
            cells = grid[r]
            if all(not _text(c) for c in cells):
                if rows:
                    break
                continue
            rec = {header[j]: cells[j] for j in range(min(len(header), len(cells))) if header[j]}
            first = next((_text(c) for c in cells if _text(c)), "")
            filled = sum(1 for k, v in rec.items() if map_column(k) and _text(v))
            is_total = bool(_TOTAL_ROW.match(first) or any(_TOTAL_ROW.match(_text(c)) for c in cells[:3]))
            if filled < 2 and any(_label(c) for c in cells) and not is_total:
                break  # 표가 끝나고 '대지면적 | 300' 같은 다른 항목이 시작됨 (그 줄은 항목·값으로 따로 읽음)
            used.add(r)
            if is_total:
                total = rec
                break  # 합계 아래는 다른 목록 (예: '건물' 동 목록)
            if rows and any(_text(c) == "건물" for c in cells[:3]):
                break  # '구분' 이 '건물' 로 바뀌면 필지 목록 끝
            if filled >= 2:
                rows.append(rec)
        if rows or total:
            return rows, total, used
    return [], None, set()


_NUMERIC_FIELDS = {"area_m2", "area_pyeong", "building_area_m2", "building_area_pyeong", "price_manwon",
                   "price_per_pyeong", "deposit_manwon", "monthly_rent_manwon"}


def _fits(field: str, value: Any) -> bool:
    """지목·용도지역 칸의 값이 정말 그 값처럼 보이는지 ('지목' 오른쪽 칸이 '지역/지구' 같은 다른 항목 이름이면 아님)."""
    t = _text(value)
    if field == "jimok":
        from ..models import JIMOK_TYPES

        return normalize_jimok(t) in JIMOK_TYPES
    if field == "zoning":
        return bool(re.search(r"지역|구역|지구|관리|녹지|주거|상업|공업|농림|보전", t)) and not map_column(t)
    if field in _NUMERIC_FIELDS:
        return bool(re.search(r"\d", t))
    if field == "jimok_zoning":
        from ..normalizer import split_jimok_zoning

        return any(split_jimok_zoning(t)) and not map_column(t)
    return True


def extract_key_values(grid: list[list[Any]], skip_rows: set[int]) -> dict[str, Any]:
    """'항목 | 값' (오른쪽) 또는 '항목 / 값' (아래) 형태. 같은 항목은 처음 것만.
    지목·용도지역은 값이 그럴듯할 때만 (아니면 아래 칸, 그래도 아니면 다음에 나오는 같은 항목에서)."""
    kv: dict[str, Any] = {}
    for r, row in enumerate(grid):
        if r in skip_rows:
            continue
        for c, cell in enumerate(row):
            field = _label(cell)
            if not field or field in kv:
                continue
            value = None
            for j in range(c + 1, min(c + 4, len(row))):  # 오른쪽 3칸 안의 첫 값
                if _text(row[j]):
                    value = None if _label(row[j]) else row[j]
                    unit = _text(row[j + 1]) if j + 1 < len(row) else ""
                    if value is not None and field in {"area_m2", "building_area_m2"} and unit == "평":
                        value = f"{_text(value)}평"  # '토지면적 | 600 | 평' -> 600평 (㎡ 가 아님)
                    break
            if value is not None and not _fits(field, value):
                value = None
            if value is None and r + 1 < len(grid) and r + 1 not in skip_rows and c < len(grid[r + 1]):
                below = grid[r + 1][c]
                if _text(below) and not _label(below) and _fits(field, below):
                    value = below
            if value is not None:
                kv[field] = (_text(cell), value)
    return kv


OFFICE_NO_COLUMNS = {"매물번호", "관리번호", "매물코드", "매물no", "매물id"}


# ---------------------------------------------------------------- 매물 특징
BULLETS = "●•·∙○◎▶►▷■□◆◇◈★☆✔✓-*"


def feature_lines(text: Any) -> list[str]:
    """'● 대소IC 15분\n● 2차선 도로 접함' / '정남향, 채광 우수' -> ['● 대소IC 15분', '● 2차선 도로 접함'] 처럼 한 줄에 하나씩, 앞에 ●."""
    out = []
    for line in re.split(r"[\r\n]+|\s+/\s+", _text(text)):
        line = line.strip()
        if not line:
            continue
        if line.startswith("※"):  # 참고 문구는 그대로
            out.append(line)
            continue
        body = line.lstrip(BULLETS + " \t").strip()
        if body:
            out.append(f"● {body}")
    return out


def extract_features(grid: list[list[Any]], skip_rows: set[int] = frozenset()) -> list[str]:
    """'매물 특징' 칸의 값 + 그 아래로 이어지는 '●' 줄들까지 모두. (필지 표 안의 '비고' 열은 제외)"""
    for r, row in enumerate(grid):
        if r in skip_rows:
            continue
        for c, cell in enumerate(row):
            if _label(cell) != "description":
                continue
            lines: list[str] = []
            col = None
            for j in range(c + 1, min(c + 4, len(row))):  # 오른쪽 값
                if _text(row[j]) and not _label(row[j]):
                    lines += feature_lines(row[j])
                    col = j
                    break
            col = c if col is None else col  # 오른쪽이 비었으면 아래 칸들이 값
            for rr in range(r + 1, min(r + 30, len(grid))):  # 아래로 이어지는 줄
                if rr in skip_rows:
                    break
                below = grid[rr]
                if c < len(below) and _label(below[c]):
                    break  # 다음 항목 시작
                value = _text(below[col]) if col < len(below) else ""
                if not value:
                    if lines:
                        break
                    continue
                if col != c and not value.startswith(tuple(BULLETS + "※")):
                    break
                lines += feature_lines(value)
            if lines:
                return lines
    return []


def format_features(lines: list[str], n_parcels: int = 0) -> str:
    """화면 표시용: '● 대소IC-15분 거리(중부 고속도로) / ● 25톤 트럭 진입'
    ('N필지 일괄 매매' 는 붙이지 않음 - 필지 수는 _필지수 로만 보관)"""
    return " / ".join(lines)


# ---------------------------------------------------------------- 매물 1건으로 합치기
def _value(rec: dict, field: str) -> tuple[Optional[str], Any]:
    for k, v in rec.items():
        if map_column(k) == field and _text(v):
            return k, v
    return None, None


def _area_of(rec: dict) -> Optional[float]:
    _, v = _value(rec, "area_m2")
    area = parse_area_m2(v) if v is not None else None
    if area is None:
        _, v = _value(rec, "area_pyeong")
        area = parse_area_m2(v, assume_pyeong=True) if v is not None else None
    return area


def _price_of(rec: dict) -> Optional[int]:
    key, v = _value(rec, "price_manwon")
    if v is None:
        return None
    if key and "(원)" in key.replace(" ", "") and isinstance(v, (int, float)):
        return int(round(float(v) / 10000))
    return parse_price_manwon(v)


def build_listing_record(unit_name: str, region_parts: list[str], grids: list[list[list[Any]]],
                         modified: Optional[date] = None) -> tuple[dict[str, Any], list[str]]:
    """매물 폴더 하나를 일반 장부의 한 줄과 같은 형태(dict)로 만듦. (레코드, 참고 메모)"""
    notes: list[str] = []
    parcels: list[dict] = []
    total: Optional[dict] = None
    kv: dict[str, tuple[str, Any]] = {}
    features: list[str] = []
    for grid in grids:
        rows, tot, used = extract_table(grid)
        features = features or extract_features(grid, used)
        parcels.extend(rows)
        total = total or tot
        for f, pair in extract_key_values(grid, used).items():
            kv.setdefault(f, pair)

    def kv_val(field: str) -> Any:
        return kv[field][1] if field in kv else None

    base, tags = split_name(unit_name)

    # 주소: 폴더 이름이 주소 모양이면 그것 (여러 필지 묶음을 대표), 아니면 엑셀의 소재지
    sheet_address = _text(kv_val("address")) or _text(_value(parcels[0], "address")[1] if parcels else "")
    address = base if looks_like_listing_name(base) else sheet_address or base
    address = _complete_address(address, sheet_address)
    for region in reversed(region_parts):  # '음성군' 폴더 안이면 주소 앞에 지역 보충 ('기타지역' 같은 이름은 제외)
        if not is_region_name(region):
            continue
        stem = re.sub(r"(시|군|구)$", "", region)
        if region not in address and not (len(stem) >= 2 and stem in address):
            address = f"{region} {address}"

    # 면적: 항목값 -> 합계 줄 -> 필지 합
    area = None
    if "area_m2" in kv:
        area = parse_area_m2(kv["area_m2"][1])
    elif "area_pyeong" in kv:
        area = parse_area_m2(kv["area_pyeong"][1], assume_pyeong=True)
    if area is None and total:
        area = _area_of(total)
    parcel_areas = [a for a in (_area_of(p) for p in parcels) if a]
    if area is None and parcel_areas:
        area = sum(parcel_areas)

    # 가격: 항목값 -> 합계 줄 -> 필지별 금액 합 (모든 필지에 금액이 있을 때만)
    price = None
    if "price_manwon" in kv:
        label = kv["price_manwon"][0]
        v = kv["price_manwon"][1]
        price = (int(round(float(v) / 10000)) if "(원)" in label.replace(" ", "") and isinstance(v, (int, float))
                 else parse_price_manwon(v))
    if price is None and total:
        price = _price_of(total)
    parcel_prices = [_price_of(p) for p in parcels]
    if price is None and parcels and all(parcel_prices):
        price = sum(parcel_prices)  # type: ignore[arg-type]

    # 지목·용도지역: 항목값, 없으면 필지 중 면적이 가장 큰 것
    from ..normalizer import split_jimok_zoning, zoning_parts

    def jz(get) -> tuple[Optional[str], Optional[str]]:
        """(지목, 용도지역) - '지목' / '용도지역' 칸, 없으면 '지목/용도' 한 칸을 나눠서."""
        j, z = get("jimok"), get("zoning")
        j = _text(j) if j is not None and _fits("jimok", j) else None
        z = _text(z) if z is not None and _fits("zoning", z) else None
        both = get("jimok_zoning")
        if both is not None and (j is None or z is None):
            sj, sz = split_jimok_zoning(both)
            j = j or sj or None
            z = z or (sz if sz and _fits("zoning", sz) else None)
        return j, z

    kv_j, kv_z = jz(lambda f: kv_val(f) if f in kv else None)
    weights_j: Counter = Counter()
    weights_z: Counter = Counter()
    for p in parcels:  # 필지 목록: 면적이 큰 필지의 지목, 지역·지구는 면적 순으로 모두
        pj, pz = jz(lambda f, p=p: _value(p, f)[1])
        size = _area_of(p) or 1
        if pj:
            weights_j[normalize_jimok(pj)] += size
        for part in zoning_parts(pz):
            weights_z[part] += size
    top_j = [k for k, _ in weights_j.most_common()]
    main_j = [k for k in top_j if k not in {"도로", "구거"}] or top_j  # 길·도랑 필지가 대표 지목이 되지 않게
    jimok = normalize_jimok(kv_j) if kv_j else (main_j[0] if main_j else "")
    jimok_mix = ", ".join(top_j[:4]) if not kv_j and len(top_j) > 1 else ""
    zoning = kv_z or ", ".join(k for k, _ in weights_z.most_common())  # 정규화는 normalizer (대표 + 전체 지역·지구)
    zoning_mix = ""

    if not features:  # 표의 '특징/비고' 열
        for p in ([total] if total else []) + parcels[:1]:
            _, v = _value(p, "description")
            if v is not None:
                features = feature_lines(v)
                break
    m = _MULTI_PARCEL.search(base)
    n_parcels = len(parcels) or (int(m.group(1)) + 1 if m else 0)

    desc_parts = []  # 검색용 전체 설명 (용도 태그 + 특징 + 필지 + 지목 구성)
    if tags:
        desc_parts.append(" ".join(tags))
    desc_parts.extend(features)
    if jimok_mix:
        desc_parts.append(f"지목: {jimok_mix}")
    if zoning_mix:
        desc_parts.append(f"용도지역: {zoning_mix}")

    # 엑셀에 적힌 사무실 매물번호 (정보지의 '매물번호' 칸, 또는 표의 '매물번호/관리번호' 열. 필지 순번 '번호' 열은 제외)
    file_no = _text(kv_val("listing_id"))
    if not file_no:
        for p in ([total] if total else []) + parcels:
            file_no = next((_text(v) for k, v in p.items()
                            if re.sub(r"\s+", "", str(k)).lower() in OFFICE_NO_COLUMNS and _text(v)), "")
            if file_no:
                break
    file_no = file_no[:-2] if file_no.endswith(".0") else file_no

    record: dict[str, Any] = {
        "매물번호": _text(kv_val("listing_id")) or "F-" + hashlib.sha1(unit_name.encode()).hexdigest()[:10],
        "_파일매물번호": file_no,
        "소재지": address,
        "면적(㎡)": round(area, 2) if area else None,
        "매매가": price,
        "지목": jimok,
        "용도지역": zoning,
        "도로": _text(kv_val("road_access")),
        "상태": _text(kv_val("status")),
        "등록일": kv_val("registered_date") or modified,
        "특이사항": " / ".join(desc_parts),
        "연락처": _text(kv_val("owner_contact")),
        "_매물폴더": unit_name,
        "_매물특징": format_features(features, n_parcels),
        "_용도태그": " ".join(tags),
        "_지목구성": jimok_mix,
        "_용도지역구성": zoning_mix,
        "_필지수": n_parcels or 1,
    }
    # 공장: 건물면적·보증금·월세 (항목값 -> 합계 줄)
    def first(field: str) -> Any:
        if field in kv:
            return kv[field][1]
        if total:
            return _value(total, field)[1]
        return None
    for out_key, field in (("건물면적(㎡)", "building_area_m2"), ("건물면적(평)", "building_area_pyeong"),
                           ("보증금", "deposit_manwon"), ("월세", "monthly_rent_manwon"),
                           ("거래유형", "deal_type"), ("매물종류", "property_type"), ("지목/용도", "jimok_zoning")):
        v = first(field)
        if v is not None and _text(v):
            record[out_key] = v
    if price is None and "price_per_pyeong" in kv:
        record["평당가"] = kv_val("price_per_pyeong")
    if not parcels and not kv:
        notes.append("엑셀에서 매물 정보를 찾지 못했습니다 (항목 이름 확인 필요)")
    return record, notes


def file_modified_date(path: Path) -> Optional[date]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date()
    except OSError:
        return None
