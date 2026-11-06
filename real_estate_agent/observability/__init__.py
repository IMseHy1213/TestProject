"""v1.1 Observable: 추적(trace) · 지표(metrics) · 구조화 로그 · 관측 화면.

    observability.setup(settings, "web")   # 웹/API/A2A 서버가 시작할 때 한 번
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .metrics import METRICS
from .tracing import TRACER, current, event, span, traceparent

log = logging.getLogger(__name__)
_PRICES: Optional[dict[str, tuple[float, float]]] = None


def setup(settings, service: str, sqlite: bool = True, json_logs: bool = True) -> None:
    """span 저장(SQLite, 선택 OTLP)과 JSON 로그를 켬. 다시 부르면 새 설정으로 바꿈."""
    from .exporters import OTLPHttpExporter, SQLiteSpanExporter

    TRACER.shutdown()
    TRACER.service = service
    if not TRACER.enabled:
        return
    if sqlite and getattr(settings, "db_path", None):
        try:
            TRACER.exporters.append(SQLiteSpanExporter(settings.db_path))
        except Exception as e:  # noqa: BLE001
            log.warning("span 저장소를 열지 못했습니다: %s", e)
    endpoint = os.getenv("REA_OTLP_ENDPOINT", "").strip()
    if endpoint:
        headers = dict(h.split("=", 1) for h in os.getenv("REA_OTLP_HEADERS", "").split(",") if "=" in h)
        TRACER.exporters.append(OTLPHttpExporter(endpoint, headers))
    if json_logs and os.getenv("REA_LOG_JSON", "1") != "0" and getattr(settings, "log_dir", None):
        from .logs import setup_json_logs

        try:
            setup_json_logs(settings.log_dir, service)
        except OSError as e:
            log.warning("JSON 로그 파일을 열지 못했습니다: %s", e)


def prices() -> dict[str, tuple[float, float]]:
    """AI 사용 단가 (달러 / 100만 토큰, 읽기·쓰기). .env 예: REA_LLM_PRICES=openai:2.5/10,anthropic:3/15
    (단가는 요금제·모델마다 다르므로 직접 확인해 적으세요. 없으면 비용은 계산하지 않고 토큰 수만 셉니다)"""
    global _PRICES
    if _PRICES is None:
        out = {}
        for item in os.getenv("REA_LLM_PRICES", "").split(","):
            try:
                name, rest = item.split(":", 1)
                a, b = rest.split("/", 1)
                out[name.strip().lower()] = (float(a), float(b))
            except ValueError:
                continue
        _PRICES = out
    return _PRICES


def cost_usd(provider: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    p = prices().get((provider or "").lower())
    if not p:
        return None
    return round((input_tokens * p[0] + output_tokens * p[1]) / 1_000_000, 6)


__all__ = ["METRICS", "TRACER", "current", "event", "setup", "span", "traceparent", "cost_usd"]
