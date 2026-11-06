"""v1.1 구조화 로그: 한 줄에 JSON 하나 (시각·수준·모듈·메시지 + trace_id/span_id).

logs/app.jsonl 에 저장 (5MB 마다 새 파일, 5개까지 보관). trace_id 로 '이 질문을 처리하는 동안 남은 로그'를 바로 찾을 수 있음.
전화번호는 기록 전에 가림.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .tracing import current_ids

_PHONE = re.compile(r"(?<!\d)0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = "app"):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        trace_id, span_id = current_ids()
        row = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name, "service": self.service,
               "msg": _PHONE.sub("***", record.getMessage())}
        if trace_id:
            row["trace_id"], row["span_id"] = trace_id, span_id
        if record.exc_info:
            row["exc"] = _PHONE.sub("***", self.formatException(record.exc_info))[-2000:]
        return json.dumps(row, ensure_ascii=False)


_HANDLER: Optional[logging.Handler] = None


def setup_json_logs(log_dir: Path, service: str, level: int = logging.INFO) -> Path:
    """real_estate_agent 로그를 logs/app.jsonl 에도 JSON 으로 남김 (화면 출력은 그대로)."""
    global _HANDLER
    path = Path(log_dir) / "app.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("real_estate_agent")
    if _HANDLER is not None:
        root.removeHandler(_HANDLER)
        _HANDLER.close()
    _HANDLER = logging.handlers.RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    _HANDLER.setFormatter(JsonFormatter(service))
    _HANDLER.setLevel(level)
    root.addHandler(_HANDLER)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return path
