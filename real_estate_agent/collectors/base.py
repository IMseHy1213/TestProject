from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RawRecord:
    """정규화 전의 원본 레코드 한 건."""

    data: dict[str, Any]
    source: str  # local / naver / serve / public ...
    source_file: str = ""


class Collector(ABC):
    name: str = "collector"

    @abstractmethod
    def collect(self) -> list[RawRecord]:
        """원본 레코드를 수집. 실패한 파일은 건너뛰고 self.errors 에 기록."""
