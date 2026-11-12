"""매물 폴더를 '매물 종류 → 지역' 순서로 정리 (sample_data 와 같은 모양).

    매물/                                   매물/
    ├─ 음성군/ …                            ├─ 토지/
    ├─ 진천군/ …                 정리 →     │   ├─ 음성군/ …
    ├─ 기타지역/ …                          │   ├─ 진천군/ …
    ├─ 웹입력/ …                            │   ├─ 기타지역/ …
    ├─ 토지매물_입력양식.xlsx               │   ├─ 웹입력/ …
    └─ 공장/매매|매매임대|임대/음성군 …     │   └─ 토지매물_입력양식.xlsx
                                            └─ 공장/ (그대로)

- 매물 폴더 바로 아래에 있는 것 중 '토지', '공장', 안내문(README.txt), '_백업' 을 뺀 나머지를 모두 '토지' 폴더로 옮깁니다.
- '토지' 폴더 안에 같은 이름의 폴더가 이미 있으면 안의 파일을 합치고, 같은 이름의 파일이 이미 있으면 옮기지 않고 알려 줍니다.
- 옮긴 내역은 logs/폴더정리/ 에 저장되어 '되돌리기' 로 원래 자리로 돌려놓을 수 있습니다.
- 매물번호·관리자가 고친 내용·관심/제외 기록은 폴더 위치와 상관없으므로 그대로 이어집니다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

KINDS = ("토지", "공장")
FACTORY_DEALS = ("매매", "매매임대", "임대")
KEEP = {"토지", "공장", "readme.txt", "_백업"}
LOG_FOLDER = "폴더정리"


def is_organized(data_dir: Path) -> bool:
    """'토지' 또는 '공장' 폴더가 있으면 종류별 구조로 봄."""
    data_dir = Path(data_dir)
    return any((data_dir / k).is_dir() for k in KINDS)


def kind_base(data_dir: Path, kind: str = "토지") -> Path:
    """그 종류의 매물을 넣을 폴더: 종류별 구조면 매물/토지 · 매물/공장, 아니면 매물 폴더 그대로."""
    data_dir = Path(data_dir)
    kind = kind if kind in KINDS else "토지"
    return data_dir / kind if is_organized(data_dir) else data_dir


def _skip(p: Path) -> bool:
    return p.name.lower() in KEEP or p.name.startswith((".", "~$"))


@dataclass
class Plan:
    moves: list[tuple[str, str]] = field(default_factory=list)      # (원래 자리, 옮길 자리) 매물 폴더 기준
    conflicts: list[str] = field(default_factory=list)              # 이미 같은 이름의 파일이 있어 못 옮김
    warnings: list[str] = field(default_factory=list)

    @property
    def needed(self) -> bool:
        return bool(self.moves)


def make_plan(data_dir: Path) -> Plan:
    """무엇을 어디로 옮길지 (아무것도 바꾸지 않음)."""
    data_dir = Path(data_dir)
    plan = Plan()
    if not data_dir.is_dir():
        plan.warnings.append(f"매물 폴더가 없습니다: {data_dir}")
        return plan
    for p in sorted(data_dir.iterdir(), key=lambda x: x.name):
        if _skip(p):
            continue
        if "공장" in p.name:  # '공장매물', '공장-매매' 처럼 이름이 다른 공장 폴더는 직접 확인
            plan.warnings.append(f"'{p.name}' 은(는) 공장 폴더로 보여 옮기지 않았습니다. "
                                 f"'공장/매매|매매임대|임대/지역' 모양으로 직접 옮겨 주세요.")
            continue
        _plan_move(data_dir, p, data_dir / "토지" / p.name, plan)
    return plan


def _plan_move(root: Path, src: Path, dst: Path, plan: Plan) -> None:
    if not dst.exists():
        plan.moves.append((src.relative_to(root).as_posix(), dst.relative_to(root).as_posix()))
    elif src.is_dir() and dst.is_dir():  # 같은 이름의 폴더가 이미 있으면 안의 것을 합침
        for child in sorted(src.iterdir(), key=lambda x: x.name):
            _plan_move(root, child, dst / child.name, plan)
    else:
        plan.conflicts.append(f"{src.relative_to(root).as_posix()} (토지 폴더에 같은 이름이 이미 있음)")


@dataclass
class Result:
    moved: list[tuple[str, str]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    log_file: Optional[Path] = None

    def summary(self) -> str:
        text = f"{len(self.moved)}개를 옮겼습니다"
        if self.failed:
            text += f" · {len(self.failed)}개는 옮기지 못했습니다 ({', '.join(self.failed[:3])})"
        return text


def _log_dir(log_dir: Path) -> Path:
    d = Path(log_dir) / LOG_FOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_plan(data_dir: Path, log_dir: Path, plan: Optional[Plan] = None) -> Result:
    """계획대로 옮기고 내역을 logs/폴더정리/ 에 저장. 엑셀에서 열려 있는 파일 등은 건너뛰고 알려 줌."""
    data_dir = Path(data_dir)
    plan = plan or make_plan(data_dir)
    result = Result()
    for src_rel, dst_rel in plan.moves:
        src, dst = data_dir / src_rel, data_dir / dst_rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            result.moved.append((src_rel, dst_rel))
        except OSError as e:  # 윈도에서 엑셀로 열어 둔 파일은 옮길 수 없음
            result.failed.append(f"{src_rel}: {e.__class__.__name__}")
    _remove_empty_dirs(data_dir, {Path(s).parts[0] for s, _ in result.moved})
    if result.moved:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.log_file = _log_dir(log_dir) / f"폴더정리_{stamp}.json"
        result.log_file.write_text(json.dumps({"data_dir": str(data_dir.resolve()), "moved": result.moved},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _remove_empty_dirs(root: Path, tops: set[str]) -> None:
    """합치면서 비게 된 원래 폴더 지우기 (안에 남은 파일이 있으면 그대로)."""
    for top in tops:
        base = root / top
        if not base.is_dir():
            continue
        for d in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [base]:
            try:
                d.rmdir()
            except OSError:
                pass


def last_log(log_dir: Path) -> Optional[Path]:
    d = Path(log_dir) / LOG_FOLDER
    logs = sorted(d.glob("폴더정리_*.json")) if d.is_dir() else []
    return logs[-1] if logs else None


def undo_last(data_dir: Path, log_dir: Path) -> Result:
    """마지막 정리를 되돌림 (옮긴 것을 원래 자리로)."""
    data_dir = Path(data_dir)
    result = Result()
    log = last_log(log_dir)
    if log is None:
        result.failed.append("되돌릴 정리 기록이 없습니다")
        return result
    moved = json.loads(log.read_text(encoding="utf-8")).get("moved", [])
    for src_rel, dst_rel in reversed(moved):
        src, dst = data_dir / src_rel, data_dir / dst_rel
        if not dst.exists() or src.exists():
            result.failed.append(f"{dst_rel} (지금 자리에 없거나 원래 자리에 이미 있음)")
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            result.moved.append((dst_rel, src_rel))
        except OSError as e:
            result.failed.append(f"{dst_rel}: {e.__class__.__name__}")
    _remove_empty_dirs(data_dir, {"토지"})
    if not result.failed:
        log.rename(log.with_suffix(".되돌림.json"))
    result.log_file = log
    return result


def describe(plan: Plan, limit: int = 40) -> list[str]:
    lines = [f"{s}  →  {d}" for s, d in plan.moves[:limit]]
    if len(plan.moves) > limit:
        lines.append(f"… 외 {len(plan.moves) - limit}개")
    return lines
