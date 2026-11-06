"""A2A (Agent2Agent) 프로토콜의 메시지 모양 (v0.3 기준, 필요한 만큼만 직접 구현).

    Agent Card   GET /.well-known/agent-card.json   이 Agent 가 누구고 무엇을 할 수 있는지 (이름·기술·주소·인증)
    JSON-RPC 2.0 POST /                              message/send · message/stream(SSE) · tasks/get · tasks/cancel

    Message  {kind:"message", messageId, role:"user"|"agent", parts:[{kind:"text",text} | {kind:"data",data}], contextId}
    Task     {kind:"task", id, contextId, status:{state, message, timestamp}, artifacts:[{artifactId, name, parts}]}
             state: submitted → working → completed / failed / canceled
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

PROTOCOL_VERSION = "0.3.0"      # 우리 클라이언트가 쓰는 형식 (서버는 1.0 형식도 받음)
PROTOCOL_VERSION_1 = "1.0"
CARD_PATH = "/.well-known/agent-card.json"
OLD_CARD_PATH = "/.well-known/agent.json"  # 예전 이름 (호환)
TERMINAL = {"completed", "failed", "canceled", "rejected"}

# JSON-RPC 오류 번호 (A2A 규격)
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
PUSH_NOT_SUPPORTED = -32003
UNSUPPORTED_OPERATION = -32004


class A2AError(Exception):
    """상대 Agent 가 JSON-RPC 오류로 답함 (요청이 잘못됨 등 - 다시 보내도 같은 결과)."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


class A2AUnavailable(Exception):
    """상대 Agent 에 닿지 못함 (꺼져 있음·시간 초과·서버 오류) - 다른 곳에서 대신 처리할 수 있음."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


def text_part(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def data_part(data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "data", "data": data}


def message(role: str, parts: list[dict], context_id: Optional[str] = None, task_id: Optional[str] = None,
            metadata: Optional[dict] = None) -> dict[str, Any]:
    m: dict[str, Any] = {"kind": "message", "messageId": new_id(), "role": role, "parts": parts}
    if context_id:
        m["contextId"] = context_id
    if task_id:
        m["taskId"] = task_id
    if metadata:
        m["metadata"] = metadata
    return m


def get_text(parts: list[dict]) -> str:
    return "\n".join(p.get("text", "") for p in parts or [] if p.get("kind") == "text").strip()


def get_data(parts: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in parts or []:
        if p.get("kind") == "data" and isinstance(p.get("data"), dict):
            out.update(p["data"])
    return out


def task_text(task: dict) -> str:
    """Task 의 답 (산출물 글자, 없으면 상태 메시지)."""
    for a in task.get("artifacts") or []:
        t = get_text(a.get("parts"))
        if t:
            return t
    return get_text(((task.get("status") or {}).get("message") or {}).get("parts"))


def task_data(task: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in task.get("artifacts") or []:
        out.update(get_data(a.get("parts")))
    return out


def rpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id: Any, code: int, msg: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": msg}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ---------------------------------------------------------------------- A2A 1.0 형식 변환
# 1.0 (현재 공식 SDK): method 이름 SendMessage/SendStreamingMessage/GetTask/CancelTask, role 'ROLE_USER',
# part 는 {"text": …} / {"data": …} (kind 없음), 상태 'TASK_STATE_COMPLETED', 스트림 이벤트는 {"statusUpdate": …}.
# 서버 안에서는 0.3 모양으로 처리하고, 1.0 으로 온 요청에만 들어올 때·나갈 때 바꿔 줌.
V1_METHODS = {"SendMessage": "message/send", "SendStreamingMessage": "message/stream", "GetTask": "tasks/get",
              "CancelTask": "tasks/cancel"}


def from_v1_message(m: Any) -> Any:
    if not isinstance(m, dict):
        return m
    parts = []
    for part in m.get("parts") or []:
        if isinstance(part, dict) and "text" in part:
            parts.append({"kind": "text", "text": part["text"]})
        elif isinstance(part, dict) and isinstance(part.get("data"), dict):
            parts.append({"kind": "data", "data": part["data"]})
    out = {"kind": "message", "messageId": m.get("messageId"), "parts": parts,
           "role": {"ROLE_USER": "user", "ROLE_AGENT": "agent"}.get(m.get("role"), m.get("role"))}
    for key in ("contextId", "taskId", "metadata"):
        if m.get(key):
            out[key] = m[key]
    return out


def _v1_parts(parts: list[dict]) -> list[dict]:
    return [{"text": x["text"]} if x.get("kind") == "text" else {"data": x.get("data", {})} for x in parts or []]


def to_v1_message(m: dict) -> dict:
    out = {"messageId": m.get("messageId"), "role": "ROLE_AGENT" if m.get("role") == "agent" else "ROLE_USER",
           "parts": _v1_parts(m.get("parts"))}
    for key in ("contextId", "taskId", "metadata"):
        if m.get(key):
            out[key] = m[key]
    return out


def to_v1_status(st: dict) -> dict:
    out = {"state": "TASK_STATE_" + str(st.get("state", "unspecified")).upper().replace("-", "_")}
    if st.get("timestamp"):
        out["timestamp"] = st["timestamp"]
    if st.get("message"):
        out["message"] = to_v1_message(st["message"])
    return out


def to_v1_artifact(a: dict) -> dict:
    return {"artifactId": a.get("artifactId"), "name": a.get("name", ""), "parts": _v1_parts(a.get("parts"))}


def to_v1_task(task: dict) -> dict:
    return {"id": task["id"], "contextId": task["contextId"], "status": to_v1_status(task["status"]),
            "artifacts": [to_v1_artifact(a) for a in task.get("artifacts") or []],
            "history": [to_v1_message(m) for m in task.get("history") or []]}


def to_v1_event(ev: dict) -> dict:
    kind = ev.get("kind")
    if kind == "task":
        return {"task": to_v1_task(ev)}
    if kind == "status-update":
        return {"statusUpdate": {"taskId": ev["taskId"], "contextId": ev["contextId"], "status": to_v1_status(ev["status"])}}
    if kind == "artifact-update":
        return {"artifactUpdate": {"taskId": ev["taskId"], "contextId": ev["contextId"],
                                   "artifact": to_v1_artifact(ev["artifact"]), "append": bool(ev.get("append")),
                                   "lastChunk": bool(ev.get("lastChunk"))}}
    return ev
