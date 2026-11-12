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
import os
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
        text = f"파일 {len(self.moved)}개를 옮겼습니다"
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
        # 파일 하나씩 '이름 바꾸기'로 옮김: 엑셀로 열어 둔 파일은 그 자리에 그대로 남고(복사·삭제가 반쯤 되는 일 없음)
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
        for f in files:
            target = dst / f.relative_to(src) if f != src else dst
            if target.exists():
                result.failed.append(f"{f.relative_to(data_dir).as_posix()}: 옮길 자리에 같은 이름의 파일이 있음")
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(f, target)
                result.moved.append((f.relative_to(data_dir).as_posix(), target.relative_to(data_dir).as_posix()))
            except OSError as e:  # 윈도에서 엑셀로 열어 둔 파일은 옮길 수 없음
                result.failed.append(f"{f.relative_to(data_dir).as_posix()}: 열려 있어 옮기지 못함 ({e.__class__.__name__})")
        if src.is_dir():  # 빈 폴더도 새 자리에 만들어 둠
            for d in (p for p in src.rglob("*") if p.is_dir()):
                (dst / d.relative_to(src)).mkdir(parents=True, exist_ok=True)
            dst.mkdir(parents=True, exist_ok=True)
    _remove_empty_dirs(data_dir, {Path(s).parts[0] for s, _ in plan.moves})
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
            os.rename(dst, src) if dst.is_file() else shutil.move(str(dst), str(src))
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


# ---------------------------------------------------------------------- 정리 뒤 빠진 매물 찾기
@dataclass
class Missing:
    listing_id: str
    office_no: str
    address: str
    old_file: str      # 매물 폴더 기준 (빠지기 전에 읽던 파일)
    now_file: str      # 지금 그 파일이 있는 곳 ('' = 못 찾음)
    reason: str


WIN_PATH_LIMIT = 250  # 윈도는 전체 경로가 260자를 넘으면 엑셀을 못 여는 경우가 있음


def _rel(data_dir: Path, path: str) -> Optional[Path]:
    """저장된 파일 경로 -> 매물 폴더 기준 상대 경로 (다른 PC 경로여도 '매물' 다음부터)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(data_dir).resolve())
    except (ValueError, OSError):
        parts = p.parts
        name = Path(data_dir).name
        return Path(*parts[parts.index(name) + 1:]) if name in parts[:-1] else None


def find_missing(agent, since_hours: float = 72) -> list[Missing]:
    """최근 매물 목록에서 빠진 매물(DB 에는 남아 있음)과 빠진 이유.
    정리(폴더 이동) 뒤 매물 수가 줄었을 때 무엇이 왜 빠졌는지 알려 줌."""
    import json as _json
    from datetime import timedelta

    from .cleaner import normalize_address
    from .models import LandListing
    from .normalizer import record_to_listing

    data_dir = Path(agent.settings.data_dir)
    since = (datetime.now() - timedelta(hours=since_hours)).isoformat(timespec="seconds")
    rows = agent.storage.conn.execute(
        "SELECT data FROM listings WHERE active=0 AND last_seen >= ? ORDER BY last_seen DESC", (since,)).fetchall()
    if not rows:
        return []
    gone = [LandListing.from_dict(_json.loads(r["data"])) for r in rows]
    hidden = {l.listing_id for l in getattr(agent, "hidden_listings", [])}
    active = agent.listings()
    by_addr = {}
    for l in active:
        by_addr.setdefault((l.property_type, normalize_address(l.address)), l)
    report = agent.last_sync
    file_errors = list(report.file_errors) if report else []
    issues = list(report.clean.issues) if report else []
    records = None  # 필요할 때만 매물 폴더 전체를 한 번 읽음

    out = []
    for l in gone:
        rel = _rel(data_dir, l.source_file) if l.source_file else None
        old = data_dir / rel if rel else None
        candidates = [old] if old else []
        if rel is not None and rel.parts and rel.parts[0] not in KINDS:
            candidates.append(data_dir / "토지" / rel)
        now = next((c for c in candidates if c and c.exists()), None)
        if now is None and rel is not None and rel.name:  # 다른 곳으로 옮겨졌는지 '폴더/파일' 이름으로 찾기
            tail = rel.parts[-2:]
            now = next((p for p in data_dir.rglob(rel.name) if p.is_file() and p.parts[-len(tail):] == tail), None)
        now_rel = now.relative_to(data_dir).as_posix() if now else ""
        if l.listing_id in hidden:
            reason = "관리자 화면에서 숨긴 매물입니다 ('숨긴 매물'에서 다시 보이게 할 수 있음)"
        elif now is None:
            reason = "파일을 찾을 수 없습니다 (옮기다 실패했거나 지워짐) → '마지막 정리 되돌리기' 를 눌러 보세요"
        elif len(str(now.resolve())) >= WIN_PATH_LIMIT:
            reason = (f"파일 경로가 너무 깁니다 ({len(str(now.resolve()))}자). 윈도는 260자를 넘으면 엑셀을 못 엽니다 "
                      "→ 폴더·파일 이름을 줄여 주세요")
        else:
            same = by_addr.get((l.property_type, normalize_address(l.address)))
            errs = [e for e in file_errors if now.name in e or now.stem in e]
            errs += [f"[{i.level}] {i.message}" for i in issues
                     if i.listing_id in {l.listing_id, l.office_no} and i.level in {"ERROR", "DUPLICATE"}]
            if same is not None:
                reason = f"같은 땅으로 보고 다른 매물({same.office_no or same.listing_id})과 합쳐졌습니다 (중복 제거)"
            elif errs:
                reason = "읽기 오류: " + "; ".join(errs[:2])
            else:
                if records is None:
                    records = agent.collector.collect()
                mine = [r for r in records if Path(r.source_file).resolve() == now.resolve()]
                errors = []
                for r in mine:
                    listing, errs2 = record_to_listing(r.data, r.source_file, r.source)
                    if listing is not None and listing.listing_id == l.listing_id:
                        errors = []
                        break
                    errors += errs2
                if not mine:
                    reason = "이 파일에서 매물을 읽지 못했습니다 (엑셀이 열려 있거나 항목 이름을 못 알아봄)"
                elif errors:
                    reason = "매물 정보 오류: " + "; ".join(errors[:2])
                else:
                    ids = sorted({record_to_listing(r.data, r.source_file, r.source)[0].listing_id
                                  for r in mine if record_to_listing(r.data, r.source_file, r.source)[0]})
                    reason = ("파일은 읽히는데 매물 번호가 달라졌습니다 (지금 번호: " + ", ".join(ids[:3]) + ")"
                              if ids else "원인을 찾지 못했습니다. 매물 정리 보고서를 확인해 주세요")
        out.append(Missing(l.listing_id, l.office_no, l.address, rel.as_posix() if rel else l.source_file,
                           now_rel, reason))
    return out


def count_change(before: set[str], after: set[str]) -> str:
    """정리 전후 매물 번호 비교 문구."""
    lost, gained = before - after, after - before
    if not lost and not gained:
        return f"매물 수 그대로 ({len(after)}건)"
    text = f"매물 수 {len(before)}건 → {len(after)}건"
    if lost:
        text += f" (빠진 매물 {len(lost)}건 - 아래 '빠진 매물' 에서 이유 확인)"
    return text
