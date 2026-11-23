"""매물 접수: 보여준 매물이 마음에 들지 않거나 따로 찾는 손님이 남기는 문의 → 저장 + 문자 알림.

흐름
  1) 첫 화면 아래 '매물 접수' 폼: 찾는 매물 종류 · 이름 · 전화번호 · 문의 사항 (+ 개인정보 수집·이용 동의)
  2) 검사(빈칸·길이·전화번호 형식·자동 입력 방지·같은 곳에서 너무 자주 보내기 방지) 후 DB 에 저장
  3) 설정된 번호(SMS_TO)로 문자 발송 - 문자 서비스 '솔라피(Solapi)' API 사용 (발신번호 사전 등록 필수)
     문자가 실패해도 접수 내용은 저장되어 관리자 화면 '매물 접수' 에서 보고 다시 보낼 수 있음

문자 설정 (my_api_key.py 또는 .env, GitHub 에 올라가지 않음 / 관리자 화면에서 저장 가능)
  SMS_TO            문자를 받을 번호 (예: 사무실 휴대폰)
  SMS_FROM          보내는 번호 - 솔라피에 '발신번호'로 등록한 번호여야 함 (전기통신사업법)
  SOLAPI_API_KEY    솔라피 API Key
  SOLAPI_API_SECRET 솔라피 API Secret
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

KINDS = ["토지", "공장", "창고", "아파트", "빌라·다세대", "단독·전원주택", "상가", "사무실", "기타"]
PURPOSES = ["찾아요", "내놓아요"]  # 매물을 찾는 손님 / 자기 매물을 우리 사무실에 내놓는 손님
MAX_NAME, MAX_MESSAGE = 20, 500
SOLAPI_URL = "https://api.solapi.com/messages/v4/send"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inquiries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    kind TEXT NOT NULL,
    purpose TEXT NOT NULL DEFAULT '찾아요',    -- 찾아요 / 내놓아요
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    message TEXT NOT NULL,
    sms_status TEXT NOT NULL DEFAULT 'pending',   -- pending / sent / failed / off (문자 설정 없음) / practice
    sms_detail TEXT NOT NULL DEFAULT '',
    handled INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------- 입력 검사
def normalize_phone(raw: str) -> str:
    """'010 1234 5678', '010.1234.5678', '+82 10-1234-5678' -> '010-1234-5678'. 형식이 아니면 ''."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("82"):
        digits = "0" + digits[2:]
    if re.fullmatch(r"01[016789]\d{7,8}", digits):
        return f"{digits[:3]}-{digits[3:-4]}-{digits[-4:]}"
    if re.fullmatch(r"02\d{7,8}", digits):  # 서울
        return f"02-{digits[2:-4]}-{digits[-4:]}"
    if re.fullmatch(r"0[3-6]\d\d{7,8}", digits):  # 지역번호 (043 …)
        return f"{digits[:3]}-{digits[3:-4]}-{digits[-4:]}"
    return ""


@dataclass
class InquiryForm:
    values: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    spam: bool = False


def parse_form(form) -> InquiryForm:
    raw = {k: (form.get(k) or "").strip()
           for k in ("purpose", "kind", "kind_other", "name", "phone", "message", "agree", "website")}
    out = InquiryForm(values={k: raw[k] for k in ("purpose", "kind", "kind_other", "name", "phone", "message")})
    if raw["website"]:  # 사람에게는 보이지 않는 칸 - 채워져 있으면 자동 입력 프로그램
        out.spam = True
        return out
    purpose = raw["purpose"] if raw["purpose"] in PURPOSES else ""
    if not purpose:
        out.errors.append("매물을 찾으시는지, 내놓으시는지 골라 주세요.")
    kind = raw["kind"] if raw["kind"] in KINDS else ""
    if not kind:
        out.errors.append("찾으시는(내놓으시는) 매물 종류를 골라 주세요.")
    elif kind == "기타" and raw["kind_other"]:
        kind = f"기타({raw['kind_other'][:20]})"
    name = re.sub(r"\s+", " ", raw["name"])
    if not name:
        out.errors.append("이름을 적어 주세요.")
    elif len(name) > MAX_NAME:
        out.errors.append(f"이름은 {MAX_NAME}자 안으로 적어 주세요.")
    phone = normalize_phone(raw["phone"])
    if not phone:
        out.errors.append("연락받으실 전화번호를 정확히 적어 주세요. (예: 010-1234-5678)")
    message = raw["message"].replace("\r\n", "\n").strip()
    if not message:
        out.errors.append("문의 사항을 적어 주세요. (지역·예산·면적, 내놓으실 매물의 위치·면적·희망 가격 등)")
    elif len(message) > MAX_MESSAGE:
        out.errors.append(f"문의 사항은 {MAX_MESSAGE}자 안으로 적어 주세요.")
    if raw["agree"] != "1":
        out.errors.append("연락을 위해 이름·전화번호를 받는 것(개인정보 수집·이용)에 동의해 주세요.")
    out.values.update(purpose=purpose or raw["purpose"], kind=kind or raw["kind"], name=name,
                      phone=phone or raw["phone"], message=message)
    return out


class RateLimit:
    """같은 곳(접속 주소)에서 짧은 시간에 너무 많이 보내지 못하게 (장난·자동 입력으로 문자비가 나가지 않도록)."""

    def __init__(self, limit: int = 3, per_sec: float = 600):
        self.limit, self.per_sec = limit, per_sec
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, who: str, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = [t for t in self._hits.get(who, []) if now - t < self.per_sec]
            if len(recent) >= self.limit:
                self._hits[who] = recent
                return False
            recent.append(now)
            self._hits[who] = recent
            return True


# ---------------------------------------------------------------------- 저장
def ensure_table(conn) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(inquiries)")}
    if "purpose" not in cols:  # 예전에 만든 표에 '찾아요/내놓아요' 칸 추가
        conn.execute("ALTER TABLE inquiries ADD COLUMN purpose TEXT NOT NULL DEFAULT '찾아요'")
    conn.commit()


def save(conn, values: dict[str, str]) -> int:
    ensure_table(conn)
    cur = conn.execute("INSERT INTO inquiries (created, purpose, kind, name, phone, message) VALUES (?,?,?,?,?,?)",
                       (datetime.now().isoformat(timespec="seconds"), values.get("purpose") or "찾아요",
                        values["kind"], values["name"], values["phone"], values["message"]))
    conn.commit()
    return int(cur.lastrowid)


def get(conn, inquiry_id: int) -> Optional[dict]:
    ensure_table(conn)
    r = conn.execute("SELECT * FROM inquiries WHERE id=?", (inquiry_id,)).fetchone()
    return dict(r) if r else None


def recent(conn, limit: int = 50) -> list[dict]:
    ensure_table(conn)
    return [dict(r) for r in conn.execute("SELECT * FROM inquiries ORDER BY id DESC LIMIT ?", (limit,))]


def set_status(conn, inquiry_id: int, status: str, detail: str = "") -> None:
    conn.execute("UPDATE inquiries SET sms_status=?, sms_detail=? WHERE id=?", (status, detail[:300], inquiry_id))
    conn.commit()


def set_handled(conn, inquiry_id: int, handled: bool) -> None:
    conn.execute("UPDATE inquiries SET handled=? WHERE id=?", (1 if handled else 0, inquiry_id))
    conn.commit()


def delete(conn, inquiry_id: int) -> None:
    """상담이 끝난 접수의 개인정보(이름·전화번호·문의)를 지움."""
    conn.execute("DELETE FROM inquiries WHERE id=?", (inquiry_id,))
    conn.commit()


def sms_text(row: dict) -> str:
    """문자 내용 (길면 장문 문자로 자동 전환)."""
    when = (row.get("created") or "")[:16].replace("T", " ")
    purpose = row.get("purpose") or "찾아요"
    return (f"[매물 접수 - {purpose}] {row['kind']}\n이름: {row['name']}\n전화: {row['phone']}\n"
            f"문의: {row['message']}\n({when} 홈페이지 접수)")


# ---------------------------------------------------------------------- 문자 보내기
class SmsError(Exception):
    pass


@dataclass
class SmsConfig:
    to: str = ""
    sender: str = ""
    api_key: str = ""
    api_secret: str = ""

    @classmethod
    def from_env(cls) -> "SmsConfig":
        return cls(to=os.getenv("SMS_TO", ""), sender=os.getenv("SMS_FROM", ""),
                   api_key=os.getenv("SOLAPI_API_KEY", ""), api_secret=os.getenv("SOLAPI_API_SECRET", ""))

    def missing(self) -> list[str]:
        names = {"to": "받는 번호(SMS_TO)", "sender": "보내는 번호(SMS_FROM)",
                 "api_key": "솔라피 API Key", "api_secret": "솔라피 API Secret"}
        return [label for k, label in names.items() if not getattr(self, k)]

    @property
    def ready(self) -> bool:
        return not self.missing()


def solapi_auth_header(api_key: str, api_secret: str, now: Optional[datetime] = None, salt: str = "") -> str:
    """솔라피 인증 머리글: HMAC-SHA256(secret, date + salt)."""
    date = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    salt = salt or secrets.token_hex(16)
    sig = hmac.new(api_secret.encode(), (date + salt).encode(), hashlib.sha256).hexdigest()
    return f"HMAC-SHA256 apiKey={api_key}, date={date}, salt={salt}, signature={sig}"


def send_sms(text: str, cfg: Optional[SmsConfig] = None, post=None) -> str:
    """문자 1건 발송. 성공하면 발송 결과 요약, 실패하면 SmsError. post: 시험용 가짜 전송 함수."""
    cfg = cfg or SmsConfig.from_env()
    if not cfg.ready:
        raise SmsError("문자 설정이 없습니다: " + ", ".join(cfg.missing()))
    to, sender = re.sub(r"\D", "", cfg.to), re.sub(r"\D", "", cfg.sender)
    body = {"message": {"to": to, "from": sender, "text": text}}
    headers = {"Authorization": solapi_auth_header(cfg.api_key, cfg.api_secret), "Content-Type": "application/json"}
    if post is None:
        import requests

        post = lambda url, **kw: requests.post(url, timeout=15, **kw)  # noqa: E731
    try:
        resp = post(SOLAPI_URL, json=body, headers=headers)
    except Exception as e:  # 인터넷 끊김 등
        raise SmsError(f"문자 서버에 연결하지 못했습니다 ({type(e).__name__})") from e
    try:
        data: Any = resp.json()
    except Exception:  # noqa: BLE001
        data = {}
    if not 200 <= resp.status_code < 300:
        msg = (data.get("errorMessage") or data.get("message") or "") if isinstance(data, dict) else ""
        raise SmsError(f"문자 발송 실패 ({resp.status_code}) {msg}".strip())
    status = str(data.get("statusMessage") or data.get("statusCode") or "접수됨") if isinstance(data, dict) else "접수됨"
    return status


def notify(conn_factory, inquiry_id: int, practice: bool = False, cfg: Optional[SmsConfig] = None,
           post=None, lock: Optional[threading.RLock] = None) -> str:
    """접수 1건을 문자로 알리고 결과를 저장. conn_factory: DB 연결을 돌려주는 함수(뒤 스레드에서도 쓰도록)."""
    from contextlib import nullcontext

    with lock or nullcontext():
        conn = conn_factory()
        row = get(conn, inquiry_id)
        if row is None:
            return "missing"
        if practice:
            set_status(conn, inquiry_id, "practice", "연습용(샘플) 실행이라 문자를 보내지 않았습니다")
            return "practice"
        cfg = cfg or SmsConfig.from_env()
        if not cfg.ready:
            set_status(conn, inquiry_id, "off", "문자 설정 없음: " + ", ".join(cfg.missing()))
            return "off"
    try:
        detail = send_sms(sms_text(row), cfg, post)
        status = "sent"
    except SmsError as e:
        detail, status = str(e), "failed"
        log.warning("매물 접수 %s 문자 실패: %s", inquiry_id, e)
    with lock or nullcontext():
        set_status(conn_factory(), inquiry_id, status, detail)
    return status


def notify_background(conn_factory, inquiry_id: int, **kw) -> threading.Thread:
    """손님 화면은 바로 '접수되었습니다' 를 보여주고, 문자는 뒤에서 보냄."""
    t = threading.Thread(target=notify, args=(conn_factory, inquiry_id), kwargs=kw, name="inquiry-sms", daemon=True)
    t.start()
    return t
