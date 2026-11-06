"""v1.0 신뢰성 지표 (프로그램이 켜진 뒤 누적): 안전장치가 몇 번 일했는지. /readyz · 관리자 화면에서 확인."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass
class ReliabilityStats:
    guard_fixes: int = 0          # AI 답변 검증 실패 → 도구 결과로 다시 작성
    injections_blocked: int = 0   # 지시 바꾸기 시도 문장 무시
    deadline_hits: int = 0        # 시간 예산 초과 → 규칙 기반으로 마무리
    llm_fallbacks: int = 0        # AI 응답 실패 → 규칙 기반으로 이어서
    tool_retries: int = 0         # 도구 일시 오류(DB 잠김 등) 재시도
    remote_fallbacks: int = 0     # 원격(A2A) Agent 가 응답 없어 이 서버가 대신 처리
    rate_limited: int = 0         # 한 방문자가 너무 자주 물음
    overloaded: int = 0           # 동시에 상담이 너무 많아 잠시 거절

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)
        try:  # v1.1: 지표 + 지금 처리 중인 질문의 trace 에 사건으로 기록
            from ..observability.metrics import RELIABILITY
            from ..observability.tracing import event

            RELIABILITY.inc(n, kind=name)
            event(f"reliability.{name}")
        except Exception:  # noqa: BLE001
            pass

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return asdict(self)

    def reset(self) -> None:
        with self._lock:
            for k in asdict(self):
                setattr(self, k, 0)


STATS = ReliabilityStats()
