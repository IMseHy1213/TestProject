"""매물 엑셀 빠르게 읽기 + 읽은 결과 기억 (프로그램 시작이 느리지 않도록).

1) .xlsx/.xlsm 은 openpyxl 읽기 전용 모드로 값만 읽음 (pandas 보다 약 7배 빠름: 사무실 설명서 1개 140ms -> 18ms).
   .xls(옛 엑셀)만 pandas 로 읽음.
2) 파일마다 (경로, 크기, 수정 시각) 이 같으면 지난번에 읽은 결과를 그대로 씀 -> 바뀐 파일만 다시 읽음.
   기억은 data/ 폴더의 '…_filecache.json' 에 저장 (JSON 이라 열어도 코드가 실행되지 않음, 지워도 다시 만듦).

시트 = (시트 이름, 2차원 셀 목록). 셀 값은 None / 문자열 / 숫자 / 날짜.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

CACHE_VERSION = 1
Sheet = tuple[str, list[list[Any]]]


def _encode(v: Any) -> Any:
    if isinstance(v, datetime):
        return {"$dt": v.isoformat()}
    if isinstance(v, date):
        return {"$d": v.isoformat()}
    if isinstance(v, time):
        return {"$t": v.isoformat()}
    return v


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if "$dt" in v:
            return datetime.fromisoformat(v["$dt"])
        if "$d" in v:
            return date.fromisoformat(v["$d"])
        if "$t" in v:
            return time.fromisoformat(v["$t"])
    return v


def _clean(v: Any) -> Any:
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, str) and v == "":
        return None
    if hasattr(v, "to_pydatetime"):  # pandas Timestamp
        return v.to_pydatetime()
    return v


def _rectangle(rows: list[list[Any]]) -> list[list[Any]]:
    """빈 줄·빈 칸을 채워 네모난 표로 (pandas 로 읽던 것과 같은 모양), 끝의 빈 줄은 뺌."""
    while rows and all(c is None for c in rows[-1]):
        rows.pop()
    width = max((len(r) for r in rows), default=0)
    while width and all(len(r) < width or r[width - 1] is None for r in rows):
        width -= 1
    return [list(r[:width]) + [None] * (width - len(r[:width])) for r in rows]


def read_sheets_uncached(path: Path) -> list[Sheet]:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            out = []
            for ws in wb.worksheets:
                ws.reset_dimensions()  # 크기 정보가 틀리게 저장된 엑셀도 끝까지 읽음
                rows = [[_clean(c) for c in row] for row in ws.iter_rows(values_only=True)]
                out.append((ws.title, _rectangle(rows)))
            return out
        finally:
            wb.close()
    if ext == ".xls":
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        return [(str(name), _rectangle([[_clean(c) for c in r] for r in df.values.tolist()]))
                for name, df in sheets.items()]
    return []


class SheetCache:
    """파일 -> 시트 목록 기억. 같은 프로그램 안에서 여러 스레드가 함께 써도 됨."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, tuple[int, int, list[Sheet]]] = {}
        self._dirty = False
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict) and saved.get("version") == CACHE_VERSION:
                    self._data = {k: (int(m), int(z), [(str(n), [[_decode(c) for c in row] for row in g]) for n, g in sh])
                                  for k, (m, z, sh) in saved.get("files", {}).items()}
            except Exception as e:  # noqa: BLE001 - 깨진 기억 파일은 버리고 새로 만듦
                log.info("엑셀 기억 파일을 새로 만듭니다: %s", e)
                self._data = {}

    @staticmethod
    def _key(path: Path) -> tuple[str, int, int]:
        st = path.stat()
        return str(path.resolve()), st.st_mtime_ns, st.st_size

    def sheets(self, path: Path) -> list[Sheet]:
        key, mtime, size = self._key(path)
        with self._lock:
            hit = self._data.get(key)
            if hit and hit[0] == mtime and hit[1] == size:
                self.hits += 1
                return hit[2]
        sheets = read_sheets_uncached(path)
        with self._lock:
            self._data[key] = (mtime, size, sheets)
            self._dirty = True
            self.misses += 1
        return sheets

    def forget_missing(self, existing: set[str]) -> None:
        """지워진 파일의 기억은 버림."""
        with self._lock:
            gone = [k for k in self._data if k not in existing]
            for k in gone:
                del self._data[k]
            self._dirty = self._dirty or bool(gone)

    def save(self) -> None:
        if not (self.path and self._dirty):
            return
        with self._lock:
            data = {"version": CACHE_VERSION, "saved": datetime.now().isoformat(timespec="seconds"),
                    "files": {k: (m, z, [(n, [[_encode(c) for c in row] for row in g]) for n, g in sh])
                              for k, (m, z, sh) in self._data.items()}}
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("엑셀 기억 파일 저장 실패 (다음 시작 때 다시 읽음): %s", e)


_DEFAULT = SheetCache()  # 기억 파일 없이 (테스트·한 번 쓰는 명령)


def read_sheets(path: Path, cache: Optional[SheetCache] = None) -> list[Sheet]:
    return (cache or _DEFAULT).sheets(Path(path))
