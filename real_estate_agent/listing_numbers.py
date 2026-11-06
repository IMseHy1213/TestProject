"""지금 매물 목록에 없는 매물번호는 화면에 보이지 않게.

예전 상담 기록·알림·AI 답변에 남은 매물번호 중 지금 매물 장부에 없는 것(연습용 샘플 매물, 팔린·지운 매물,
AI 가 지어낸 번호)은 '(지금 없는 매물)' 로 바꿔 보여 줍니다. 손님은 존재하지 않는 매물번호를 보지 않습니다.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# 매물번호 모양: 'E-3002', 'T-1004', 'S.24250409', 'AB-12' (영문 1~3자 + '-' 또는 '.' + 숫자 2~10자리)
NUMBER = re.compile(r"(?<![A-Za-z0-9_.-])([A-Z]{1,3}[-.]\d{2,10})(?![A-Za-z0-9_-])")
MISSING = "(지금 없는 매물)"


def known_numbers(agent) -> set[str]:
    """지금 매물 장부(숨긴 매물 포함)에 있는 매물번호·id 들."""
    from .agent import public_office_no
    from .tool_agent.tools import tool_id

    out: set[str] = set()
    for l in list(agent.listings()) + list(getattr(agent, "hidden_listings", [])):
        out |= {l.listing_id, tool_id(l), l.office_no or "", public_office_no(l.office_no) or ""}
    out.discard("")
    return out


def unknown_in(text: str, known: set[str]) -> list[str]:
    return [m for m in NUMBER.findall(text or "") if m not in known]


def hide_unknown(text: str, known: set[str]) -> str:
    """없는 매물번호를 '(지금 없는 매물)' 로."""
    if not text:
        return text
    return NUMBER.sub(lambda m: m.group(0) if m.group(1) in known else MISSING, text)


def hide_in(value: Any, known: set[str]) -> Any:
    """문자열·목록·사전 안의 없는 매물번호를 모두 가림 (실행 과정의 도구 인자 등)."""
    if isinstance(value, str):
        return hide_unknown(value, known)
    if isinstance(value, list):
        return [hide_in(v, known) for v in value]
    if isinstance(value, dict):
        return {k: hide_in(v, known) for k, v in value.items()}
    return value


def event_listing_exists(payload: dict, known: set[str]) -> bool:
    """실시간 이벤트가 지금 있는 매물에 대한 것인지 (예전 샘플 매물 알림은 다시 보내지 않음)."""
    card = payload.get("listing") if isinstance(payload, dict) else None
    ids: Iterable[str] = [card.get("id"), card.get("office_no")] if isinstance(card, dict) else []
    ids = [i for i in ids if i]
    if not ids and isinstance(payload, dict) and payload.get("listing_id"):
        ids = [payload["listing_id"]]
    return not ids or any(i in known for i in ids)
