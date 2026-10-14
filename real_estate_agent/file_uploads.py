"""관리자 화면에서 매물 파일(.xlsx 등) 여러 개 또는 폴더째 올리기.

- 폴더를 올리면 안의 하위 폴더 구조를 그대로 매물 폴더에 저장합니다.
  예) '음성군' 아래에 '음성군 감곡면 문촌리 417-2 외 36필지(물류부지)' 폴더를 올리면
      매물/음성군/음성군 감곡면 문촌리 417-2 외 36필지(물류부지)/....xlsx  -> 매물 1건으로 읽힘
- 같은 이름의 파일이 이미 있으면: 기본은 둘 다 보관(새 파일 이름 뒤에 _2),
  '교체' 를 고르면 기존 파일을 매물 폴더의 '_백업' 폴더로 옮기고 새 파일로 바꿈. ('_백업' 은 매물로 읽지 않음)
- 매물 파일 형식(.xlsx .xlsm .xls .csv .json .txt)만 저장하고, 사진·PDF 등은 건너뜁니다.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from .collectors.local_folder import SUPPORTED

BACKUP_FOLDER = "_백업"
DEFAULT_FOLDER = "웹업로드"
MAX_FILES = 300
MAX_DEPTH = 6
MAX_FILE_MB = 20


@dataclass
class UploadResult:
    saved: list[str] = field(default_factory=list)      # 매물 폴더 기준 상대 경로
    replaced: list[str] = field(default_factory=list)   # 교체하면서 _백업 으로 옮긴 기존 파일
    renamed: list[str] = field(default_factory=list)    # 이름이 겹쳐 _2 등을 붙인 파일
    skipped: list[str] = field(default_factory=list)    # 형식이 달라 건너뛴 파일 (이유 포함)

    def summary(self) -> str:
        parts = [f"파일 {len(self.saved)}개 저장"]
        if self.replaced:
            parts.append(f"{len(self.replaced)}개는 기존 파일을 '{BACKUP_FOLDER}' 에 보관하고 교체")
        if self.renamed:
            parts.append(f"{len(self.renamed)}개는 이름이 겹쳐 뒤에 번호를 붙임")
        if self.skipped:
            shown = ", ".join(self.skipped[:3]) + (" …" if len(self.skipped) > 3 else "")
            parts.append(f"{len(self.skipped)}개 건너뜀 ({shown})")
        return " · ".join(parts)


def clean_part(part: str) -> str:
    """폴더/파일 이름 한 칸 정리 (한글·괄호 유지, 위험한 글자 제거)."""
    part = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", part).strip(" .")
    return "" if part in {"", ".", ".."} else part[:120]


def clean_relative_path(name: str) -> Optional[PurePosixPath]:
    """브라우저가 보낸 파일 경로('폴더/하위/파일.xlsx') -> 안전한 상대 경로. 쓸 수 없으면 None."""
    raw = [p for p in name.replace("\\", "/").split("/")]
    if any(p.strip() == ".." for p in raw):
        return None
    parts = [clean_part(p) for p in raw]
    parts = [p for p in parts if p]
    if not parts or len(parts) > MAX_DEPTH:
        return None
    return PurePosixPath(*parts)


def destination_folders(data_dir: Path) -> list[str]:
    """올릴 곳으로 고를 수 있는 폴더: 매물 폴더 바로 아래 폴더들."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(p.name for p in data_dir.iterdir()
                  if p.is_dir() and not p.name.startswith((".", "~", BACKUP_FOLDER)))


def resolve_destination(data_dir: Path, dest: str) -> Path:
    """'음성군', '기타지역/문경시', '' (매물 폴더 바로 아래) -> 실제 폴더. 매물 폴더 밖이면 ValueError."""
    data_dir = Path(data_dir).resolve()
    rel = clean_relative_path(dest) if dest.strip() else PurePosixPath()
    if rel is None:
        raise ValueError("올릴 폴더 이름이 올바르지 않습니다.")
    if rel.parts and rel.parts[0].startswith(BACKUP_FOLDER):
        raise ValueError(f"'{BACKUP_FOLDER}' 폴더에는 올릴 수 없습니다.")
    target = (data_dir / Path(*rel.parts)).resolve() if rel.parts else data_dir
    if target != data_dir and data_dir not in target.parents:
        raise ValueError("매물 폴더 밖에는 저장할 수 없습니다.")
    return target


def _free_name(path: Path) -> Path:
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


def save_uploads(data_dir: Path, files: Iterable, dest: str = DEFAULT_FOLDER, replace: bool = False) -> UploadResult:
    """files: werkzeug FileStorage 목록 (filename 에 폴더 경로가 들어 있을 수 있음)."""
    data_dir = Path(data_dir)
    base = resolve_destination(data_dir, dest)
    result = UploadResult()
    files = [f for f in files if f and f.filename]
    if not files:
        raise ValueError("올릴 파일이나 폴더를 선택해 주세요.")
    if len(files) > MAX_FILES:
        raise ValueError(f"한 번에 {MAX_FILES}개까지 올릴 수 있습니다. 폴더를 나눠서 올려 주세요.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = data_dir.resolve()
    for f in files:
        rel = clean_relative_path(f.filename)
        shown = f.filename.replace("\\", "/")
        if rel is None:
            result.skipped.append(f"{shown} (이름 오류)")
            continue
        if rel.name.startswith("~$") or rel.name.startswith(".") or rel.name.lower() in {"thumbs.db", "desktop.ini"}:
            continue  # 엑셀 임시 파일·시스템 파일은 조용히 건너뜀
        if Path(rel.name).suffix.lower() not in SUPPORTED:
            result.skipped.append(rel.name)
            continue
        size = _size(f)
        if size > MAX_FILE_MB * 1024 * 1024:
            result.skipped.append(f"{rel.name} ({MAX_FILE_MB}MB 초과)")
            continue
        target = (base / Path(*rel.parts)).resolve()
        if root not in target.parents:
            result.skipped.append(f"{shown} (매물 폴더 밖)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if replace:
                backup = data_dir / BACKUP_FOLDER / stamp / target.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(backup))
                result.replaced.append(str(target.relative_to(root)))
            else:
                target = _free_name(target)
                result.renamed.append(str(target.relative_to(root)))
        f.save(str(target))
        result.saved.append(str(target.relative_to(root)))
    if not result.saved:
        raise ValueError("저장할 매물 파일이 없습니다. (.xlsx .xls .csv 등 매물 파일만 올릴 수 있습니다)"
                         + (f" 건너뜀: {', '.join(result.skipped[:5])}" if result.skipped else ""))
    return result


def _size(f) -> int:
    try:
        stream = f.stream
        pos = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(pos)
        return size
    except (AttributeError, OSError):
        return 0


def listing_folder(data_dir: Path, source_file: str) -> str:
    """매물의 원본 파일이 있는 폴더 (매물 폴더 기준 상대 경로, 바로 아래면 '')."""
    if not source_file:
        return DEFAULT_FOLDER
    try:
        rel = Path(source_file).resolve().parent.relative_to(Path(data_dir).resolve())
    except ValueError:
        return DEFAULT_FOLDER
    return rel.as_posix() if str(rel) != "." else ""


def files_in(data_dir: Path, folder: str, limit: int = 30) -> list[str]:
    """폴더 안의 매물 파일 목록 (수정 화면 표시용)."""
    try:
        base = resolve_destination(data_dir, folder)
    except ValueError:
        return []
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("~$"))[:limit]
