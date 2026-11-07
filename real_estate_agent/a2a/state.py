"""Agent 사이에 주고받는 대화 상태 (공개용 매물 id 로 바꿔서 보냄).

분산 구조에서는 Agent 들이 메모리를 함께 쓸 수 없으므로, 총괄 Agent 가 대화 상태(직전 검색 조건·보여준 매물·관심/제외)를
메시지에 실어 보내고, 전문 Agent 는 일을 마친 뒤 바뀐 상태를 돌려줍니다. 그래서 전문 Agent 서버는 상태를 들고 있지 않아도 되고
(재시작·여러 대로 늘리기 쉬움), 손님의 장기 기억(개인 기록)은 웹 서버 밖으로 나가지 않습니다.
매물 id 는 매물번호 괄호 속 이름이 빠진 공개용 id(tool_id)로 바꿔 보냅니다.
"""

from __future__ import annotations

from typing import Any, Callable

from ..tool_agent.memory import dump_ctx, load_ctx
from ..tool_agent.tools import AgentContext, tool_id


def id_maps(agent) -> tuple[Callable[[str], str], Callable[[str], str]]:
    """(실제 id -> 공개 id, 공개 id -> 실제 id). 모르는 id 는 그대로."""
    to_public, to_real = {}, {}
    for l in agent.listings() + list(getattr(agent, "hidden_listings", [])):
        pub = tool_id(l)
        to_public[l.listing_id] = pub
        to_real[pub] = l.listing_id
        to_real[l.listing_id] = l.listing_id
    return (lambda i: to_public.get(i, i)), (lambda i: to_real.get(i, i))


def _convert(raw: dict[str, Any], f: Callable[[str], str]) -> dict[str, Any]:
    out = dict(raw)
    out["rejected"] = {f(k): v for k, v in (raw.get("rejected") or {}).items()}
    for key in ("liked", "seen", "last_shown"):
        out[key] = [f(i) for i in raw.get(key) or []]
    return out


def public_state(agent, ctx: AgentContext) -> dict[str, Any]:
    raw = dump_ctx(ctx)
    raw.pop("run_searches", None)  # 이번 질문에서만 쓰는 값
    return _convert(raw, id_maps(agent)[0])


def private_state(agent, raw: dict[str, Any]) -> AgentContext:
    if not isinstance(raw, dict):
        raise ValueError("state 는 객체여야 합니다")
    ctx = load_ctx(_convert(raw, id_maps(agent)[1]))
    if not isinstance(ctx.last_search, dict) or not isinstance(ctx.rejected, dict):
        raise ValueError("state 형식이 맞지 않습니다")
    ctx.run_searches = []
    return ctx


def copy_into(target: AgentContext, source: AgentContext) -> None:
    """같은 상태 객체를 여러 Agent 가 참조하므로, 새 객체로 바꾸지 않고 내용만 덮어씀."""
    for name in ("rejected", "liked", "seen", "customer", "last_search", "last_shown", "last_search_from_memory",
                 "last_regions", "region_focus"):
        setattr(target, name, getattr(source, name))
