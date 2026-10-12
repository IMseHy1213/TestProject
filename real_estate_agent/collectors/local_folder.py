"""매물 폴더 수집기 (기본: 프로젝트 폴더 안의 '매물' 폴더).

지원 형식: .xlsx / .xlsm / .xls / .csv / .json / .txt
- 엑셀은 모든 시트를 읽고, 제목 줄이 위에 있어도 헤더 행을 자동으로 찾습니다.
- .txt 는 '항목: 값' 형식, 빈 줄 또는 '---' 로 매물을 구분합니다.
- 하위 폴더/파일 이름에 '네이버', '써브' 등이 들어가면 출처(source)를 자동으로 태깅합니다.
  (네이버 부동산·부동산써브에서 내려받거나 정리한 파일을 하위 폴더에 넣으면 함께 수집됩니다.)
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from typing import Optional

from ..normalizer import map_column
from .base import Collector, RawRecord
from .listing_folder import build_listing_record, file_modified_date, looks_like_listing_name, read_grids

GRID_FORMATS = {".xlsx", ".xlsm", ".xls", ".csv"}

log = logging.getLogger(__name__)

SUPPORTED = {".xlsx", ".xlsm", ".xls", ".csv", ".json", ".txt"}

SOURCE_HINTS = {
    "naver": ["네이버", "naver"],
    "serve": ["써브", "서브", "serve"],
    "public": ["공공", "실거래", "onbid", "온비드", "경매", "공매"],
}


def detect_source(path: Path, root: Path) -> str:
    try:
        rel = str(path.relative_to(root)).lower()
    except ValueError:
        rel = path.name.lower()
    for source, hints in SOURCE_HINTS.items():
        if any(h in rel for h in hints):
            return source
    return "local"


class LocalFolderCollector(Collector):
    name = "local_folder"

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.errors: list[str] = []
        self.files_read: list[str] = []

    def iter_files(self) -> list[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"매물 폴더를 찾을 수 없습니다: {self.data_dir}\n"
                "  --data-dir 옵션이나 .env 의 REA_DATA_DIR 로 경로를 지정하세요."
            )
        return sorted(
            p for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(("~$", "."))
            and p.name.lower() != "readme.txt"  # 폴더 안내문은 매물 파일이 아님
        )

    def listing_unit(self, path: Path) -> Optional[Path]:
        """'매물 하나 = 폴더 하나' 구조면 그 매물 폴더(또는 파일)를 반환, 일반 장부 파일이면 None."""
        try:
            rel_parts = path.relative_to(self.data_dir).parts
        except ValueError:
            return None
        if len(rel_parts) >= 2 and looks_like_listing_name(rel_parts[-2]):
            return path.parent
        if looks_like_listing_name(path.stem) and path.suffix.lower() in GRID_FORMATS:
            return path
        return None

    def collect(self) -> list[RawRecord]:
        self.errors.clear()
        self.files_read.clear()
        records: list[RawRecord] = []
        units: dict[Path, list[Path]] = {}
        ledger_files: list[Path] = []
        for path in self.iter_files():
            unit = self.listing_unit(path)
            if unit is None:
                ledger_files.append(path)
            elif path.suffix.lower() in GRID_FORMATS:
                units.setdefault(unit, []).append(path)

        # 1) 매물 폴더: 폴더 안 엑셀 전체 -> 매물 1건
        for unit, files in sorted(units.items()):
            grids = []
            for f in files:
                try:
                    grids.extend(read_grids(f))
                    self.files_read.append(str(f))
                except Exception as e:
                    self.errors.append(f"{f.name}: 읽기 실패 ({e})")
                    log.warning("파일 읽기 실패 %s: %s", f, e)
            if not grids:
                continue
            region_parts = list(unit.relative_to(self.data_dir).parts[:-1])
            name = unit.stem if unit.is_file() else unit.name
            record, notes = build_listing_record(name, region_parts, grids, file_modified_date(files[0]))
            for n in notes:
                self.errors.append(f"{name}: {n}")
            records.append(RawRecord(data=record, source=detect_source(unit, self.data_dir),
                                     source_file=str(files[0])))

        # 2) 일반 장부 파일: 표 한 줄 = 매물 1건
        for path in ledger_files:
            source = detect_source(path, self.data_dir)
            try:
                rows = read_file(path)
                if not rows and path.suffix.lower() in GRID_FORMATS:
                    rows = self._as_single_listing(path)  # 표가 아니면 '항목 | 값' 정보지로 다시 읽기
            except Exception as e:  # 파일 하나가 깨져도 전체 수집은 계속
                self.errors.append(f"{path.name}: 읽기 실패 ({e})")
                log.warning("파일 읽기 실패 %s: %s", path, e)
                continue
            self.files_read.append(str(path))
            if not rows:
                self.errors.append(f"{path.relative_to(self.data_dir)}: 매물 정보를 찾지 못했습니다 (항목 이름 확인 필요)")
            records.extend(RawRecord(data=r, source=source, source_file=str(path)) for r in rows)
        return records

    def _as_single_listing(self, path: Path) -> list[dict]:
        grids = read_grids(path)
        record, _ = build_listing_record(path.stem, list(path.relative_to(self.data_dir).parts[:-1]), grids,
                                         file_modified_date(path))
        has_core = record.get("소재지") and record.get("면적(㎡)") and record.get("매매가")
        return [record] if has_core else []

    def folder_counts(self, listings) -> dict[str, int]:
        """매물 폴더 바로 아래 폴더별 매물 수 (예: {'음성군': 12, '기타지역': 5, '(최상위)': 3})."""
        counts: dict[str, int] = {}
        for l in listings:
            try:
                parts = Path(l.source_file).relative_to(self.data_dir).parts
            except ValueError:
                continue
            key = parts[0] if len(parts) > 1 else "(최상위)"
            counts[key] = counts.get(key, 0) + 1
        for d in sorted(p.name for p in self.data_dir.iterdir() if p.is_dir()) if self.data_dir.exists() else []:
            counts.setdefault(d, 0)  # 비어 있는 폴더도 0건으로 보여줌
        return counts


def read_file(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return _read_excel(path)
    if ext == ".csv":
        return _read_csv(path)
    if ext == ".json":
        return _read_json(path)
    if ext == ".txt":
        return _read_txt(path)
    return []


def _find_header_row(rows: list[list[Any]], scan: int = 15) -> int | None:
    best, best_hits = None, 1
    for i, row in enumerate(rows[:scan]):
        hits = sum(1 for c in row if c is not None and map_column(c))
        if hits > best_hits:
            best, best_hits = i, hits
    return best


def _rows_to_dicts(rows: list[list[Any]]) -> list[dict[str, Any]]:
    h = _find_header_row(rows)
    if h is None:
        return []
    header = [str(c).strip() if c is not None else f"col{j}" for j, c in enumerate(rows[h])]
    out = []
    for row in rows[h + 1:]:
        if all(c is None or str(c).strip() == "" for c in row):
            continue
        out.append({header[j]: row[j] for j in range(min(len(header), len(row)))})
    return out


def _read_excel(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    out: list[dict[str, Any]] = []
    for sheet_name, df in sheets.items():
        rows = df.where(df.notna(), None).values.tolist()
        for r in _rows_to_dicts(rows):
            r.setdefault("_sheet", sheet_name)
            out.append(r)
    return out


def _read_csv(path: Path) -> list[dict[str, Any]]:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            return _rows_to_dicts([[c if c != "" else None for c in r] for r in rows])
        except UnicodeDecodeError:
            continue
    raise ValueError("인코딩을 알 수 없습니다 (utf-8 / cp949 아님)")


def _read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("listings") or data.get("items") or [data]
    return [d for d in data if isinstance(d, dict)]


def _read_txt(path: Path) -> list[dict[str, Any]]:
    text = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("인코딩을 알 수 없습니다")
    out = []
    for block in re.split(r"\n\s*(?:-{3,}|={3,})?\s*\n", text):
        rec: dict[str, Any] = {}
        for line in block.splitlines():
            m = re.match(r"\s*([^:：]+?)\s*[:：]\s*(.+)$", line)
            if m:
                rec[m.group(1)] = m.group(2).strip()
        if sum(1 for k in rec if map_column(k)) >= 2:
            out.append(rec)
    return out
