"""매물 엑셀의 '양식(칸 배치)' 만 개인정보 없이 보여 주기 — 지목·용도지역 같은 칸을 프로그램이 못 읽을 때 원인 찾기용.

    python main.py sheet-layout              # 지목·용도지역이 비어 있는 공장 매물의 엑셀을 자동으로 골라 보여줌
    python main.py sheet-layout 파일.xlsx     # 그 파일

- 항목 이름(소재지, 지목, 지역/지구 …)과 짧은 낱말(대, 전, 계획관리지역 …)만 그대로 보여 줍니다.
- 숫자가 들어간 값(지번, 전화번호, 금액, 면적)은 '#', 긴 글은 '(글)', 소유자·연락처·성명 칸 옆/아래 값은 '(가림)' 으로 가립니다.
- 결과는 화면과 logs/양식점검.txt 에 저장됩니다. 보내기 전에 한 번 훑어보고 이름 같은 것이 남아 있으면 지워 주세요.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .collectors.listing_folder import _text, read_grids
from .normalizer import map_column

_PRIVATE_LABEL = re.compile(r"소유|연락|성명|이름|매도인|임대인|임차인|전화|휴대|핸드폰|주민|계좌|담당")
MAX_ROWS = 40
MAX_COLS = 12


def _mask(text: str, private: bool) -> str:
    if not text:
        return ""
    if private:
        return "(가림)"
    if re.search(r"\d", text):
        return "#"
    if len(text) > 14:
        return "(글)"
    return text


def layout_lines(path: Path, max_rows: int = MAX_ROWS) -> list[str]:
    """한 파일의 시트별 칸 배치 (가린 값). ✓ 표시는 프로그램이 알아보는 항목 이름."""
    out = [f"■ 파일: {path.name}"]
    for si, grid in enumerate(read_grids(path)):
        out.append(f"  [시트 {si + 1}]")
        private: set[tuple[int, int]] = set()
        for r, row in enumerate(grid[:max_rows]):
            for c, cell in enumerate(row[:MAX_COLS]):
                if _PRIVATE_LABEL.search(_text(cell)) and len(_text(cell)) <= 12:
                    private.update({(r, c + 1), (r, c + 2)})
                    below = _text(grid[r + 1][c]) if r + 1 < len(grid) and c < len(grid[r + 1]) else ""
                    if re.fullmatch(r"[가-힣]{2,5}", below) and not map_column(below):  # 표 아래 칸의 이름
                        private.add((r + 1, c))
        for r, row in enumerate(grid[:max_rows]):
            cells = []
            for c, cell in enumerate(row[:MAX_COLS]):
                t = _text(cell)
                field = map_column(t) if t and len(t) <= 20 else None
                shown = f"{t}✓" if field else _mask(t, (r, c) in private)
                cells.append(shown)
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                out.append(f"  {r + 1:>3}행: " + " | ".join(c or "·" for c in cells))
    return out


def _same_listing_files(path: Path, limit: int = 4) -> list[Path]:
    from .collectors.listing_folder import looks_like_listing_name

    folder = path.parent
    if not looks_like_listing_name(folder.name):
        return [path]
    files = [path] + sorted(f for f in folder.glob("*") if f.is_file() and f != path
                            and f.suffix.lower() in {".xlsx", ".xlsm", ".xls", ".csv"} and not f.name.startswith("~$"))
    return files[:limit]


def files_needing_check(agent, kind: str = "공장", limit: int = 3) -> list[tuple[Path, Any]]:
    """지목이나 용도지역이 비어 있는 매물의 엑셀 (같은 파일은 한 번만)."""
    seen: dict[Path, Any] = {}
    for l in agent.listings():
        if kind and l.property_type != kind:
            continue
        if l.jimok and l.zoning:
            continue
        p = Path(l.source_file) if l.source_file else None
        if p and p.exists() and p not in seen:
            seen[p] = l
        if len(seen) >= limit:
            break
    return list(seen.items())


def report(agent, path: Optional[str] = None, kind: str = "공장") -> str:
    lines: list[str] = []
    if path:
        targets = [(Path(path), None)]
    else:
        targets = files_needing_check(agent, kind)
        total = [l for l in agent.listings() if (not kind or l.property_type == kind)]
        empty = [l for l in total if not (l.jimok and l.zoning)]
        lines.append(f"{kind or '전체'} 매물 {len(total)}건 중 지목 또는 용도지역이 비어 있는 매물 {len(empty)}건")
        if not targets:
            lines.append("✅ 지목·용도지역이 비어 있는 매물이 없습니다.")
            return "\n".join(lines)
    for p, l in targets:
        if l is not None:
            lines.append(f"\n지금 읽은 값 → 매물번호: {l.office_no or '미등록'} / 지목: {l.jimok or '-'} / "
                         f"용도지역: {l.zoning or '-'} / 거래: {l.deal_type} / 가격: {l.price_text}")
        for f in _same_listing_files(p):  # 매물 하나 = 폴더 하나면 그 폴더의 엑셀 모두 (토지 목록 파일 등)
            try:
                lines += layout_lines(f)
            except Exception as e:  # noqa: BLE001 - 깨진 파일도 다른 파일은 계속
                lines.append(f"■ 파일: {f.name} - 읽기 실패 ({e.__class__.__name__})")
    lines.append("\n✓ = 프로그램이 알아보는 항목 이름, # = 숫자가 든 값, (글) = 긴 글, (가림) = 소유자·연락처 칸")
    return "\n".join(lines)
