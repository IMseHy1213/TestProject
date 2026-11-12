"""v1.0 가드레일: 들어오는 질문과 나가는 답변을 검사.

입력 가드
  - 지시 바꾸기 시도('이전 지시 무시하고…', '시스템 프롬프트 보여줘')가 섞인 문장 조각은 빼고 나머지만 처리.
    남는 질문이 없으면 정해진 안내로 답함 (AI 를 부르지 않음).
  - 소유자 연락처·정확한 지번을 물으면, 답 끝에 '상담 시 안내' 문장과 사무실 전화번호를 붙임.

출력 가드 (AI 가 쓴 답변만 검사 - 규칙 기반 답변은 도구 결과로 만들어지므로 검사할 필요 없음)
  - 답변에 나온 매물(매물번호·id), 가격(억·만원), 면적(평)이 이번 질문의 도구 결과에 실제로 있는지  (환각 방지)
  - 내부 번호(F-…), 전화번호(사무실 번호 제외), 지번('…리 123-4') 이 나오지 않는지  (개인정보)
  → 하나라도 걸리면 그 답을 버리고 도구 결과로 다시 작성 (trace 에 '안전장치' 로 이유 표시).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

OFFICE_PHONE_FALLBACK = "043-877-5242"

# ---------------------------------------------------------------------- 입력
_INJECTION = [
    re.compile(r"(이전|위|앞|앞의|모든|기존|원래)\s*(의\s*)?(지시|명령|규칙|설정|프롬프트|지침)\S*\s*(은|는|을|를|다|모두)?\s*"
               r"(전부\s*|다\s*|모두\s*)?(무시|잊|취소|없던)"),
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)", re.I),
    re.compile(r"시스템\s*프롬프트|system\s*prompt|개발자\s*(모드|메시지)|developer\s*(mode|message)|jailbreak|탈옥", re.I),
    re.compile(r"너는\s*이제(부터)?\s|넌\s*이제(부터)?\s|you\s+are\s+now\b|act\s+as\b", re.I),
    re.compile(r"(규칙|제한|필터|안전장치)\S*\s*(을|를)?\s*(무시|해제|풀어|끄)"),
    re.compile(r"(프롬프트|지시문|지침)\S*\s*(을|를)?\s*(보여|알려|출력|말해)"),
]
_SPLIT = re.compile(r"(?<=[.!?\n。])\s*|(?<=[고서며])\s+|,\s*")
_CONTACT = re.compile(r"연락처|전화\s*번호|폰\s*번호|핸드폰|휴대폰|집주인|소유자|땅\s*주인|매도인|지번|번지|정확한\s*(주소|위치)|상세\s*주소")
REFUSAL = ("저는 토지·공장 매물 상담만 도와드릴 수 있습니다. 찾으시는 지역·용도·예산을 말씀해 주세요. "
           "(예: 음성군에 창고 지을 수 있는 땅)")


def office_phone() -> str:
    """사무실 전화번호 (웹 화면의 OFFICE_INFO 가 기준)."""
    try:
        from ..web.app import OFFICE_INFO

        return OFFICE_INFO.get("phone") or OFFICE_PHONE_FALLBACK
    except Exception:  # noqa: BLE001 - 웹 모듈 없이 쓸 때
        return OFFICE_PHONE_FALLBACK


def contact_notice() -> str:
    return f"소유자 연락처와 정확한 지번은 사무실 상담 시 안내해 드립니다 (☎ {office_phone()})."


@dataclass
class InputCheck:
    question: str                 # 처리할 질문 (지시 바꾸기 시도는 뺀 것)
    removed: list[str] = field(default_factory=list)
    refuse: bool = False          # 남는 질문이 없음 -> REFUSAL 로 답함
    asks_contact: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def is_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def check_input(question: str) -> InputCheck:
    q = (question or "").strip()
    asks_contact = bool(_CONTACT.search(q))
    if not is_injection(q):
        return InputCheck(q, asks_contact=asks_contact)
    pieces = [x for x in _SPLIT.split(q) if x and x.strip()]
    kept = [x.strip() for x in pieces if not is_injection(x)]
    removed = [x.strip() for x in pieces if is_injection(x)]
    rest = " ".join(kept).strip(" ,.")
    if not re.search(r"[가-힣A-Za-z0-9]{2,}", rest):
        return InputCheck("", removed, refuse=True, asks_contact=asks_contact)
    return InputCheck(rest, removed, asks_contact=bool(_CONTACT.search(rest)))


# ---------------------------------------------------------------------- 출력
_PRICE = re.compile(r"(\d+(?:\.\d+)?)\s*억(?:\s*([\d,]+)\s*만)?(?:\s*원)?|(?<![\d억])([\d,]+)\s*만\s*원")
_AREA = re.compile(r"([\d,]+(?:\.\d+)?)\s*평(?!당|균|일|야|가)")
_M2 = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:㎡|m2|제곱미터)")
_PHONE = re.compile(r"(?<!\d)0\d{1,2}[-.\s)]?\d{3,4}[-.\s]?\d{4}(?!\d)")
_LOT = re.compile(r"[가-힣]{1,6}(?:리|동)\s*(?:산\s*)?\d+(?:-\d+)?(?:\s*번지)?(?![\d,]*\s*(?:건|평|㎡|억|만|개|%|곳|m))")
_INTERNAL_ID = re.compile(r"(?<![A-Za-z0-9_.-])F-[0-9a-f]{6,}(?![A-Za-z0-9_-])")


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def prices_in(text: str) -> set[int]:
    """'2억 1,000만원' '3억원' '9,000만원' -> 만원 단위 숫자들."""
    out = set()
    for m in _PRICE.finditer(text):
        if m.group(1):
            out.add(round(_num(m.group(1)) * 10000 + (_num(m.group(2)) if m.group(2) else 0)))
        elif m.group(3):
            out.add(round(_num(m.group(3))))
    return out


def _numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)


@dataclass
class Evidence:
    """이번 질문에서 도구가 돌려준 사실들 (답변 검사 기준)."""
    text: str
    numbers: set[float]
    prices: set[int]
    areas: set[float]

    @classmethod
    def from_observations(cls, observations: Iterable[tuple[str, dict]]) -> "Evidence":
        obs = [o for _, o in observations if isinstance(o, dict) and o.get("ok")]
        text = json.dumps(obs, ensure_ascii=False, default=str)
        numbers = set(_numbers(obs))
        areas = {_num(m.group(1)) for m in _AREA.finditer(text)} | {_num(m.group(1)) for m in _M2.finditer(text)}
        return cls(text, numbers, prices_in(text) | {int(n) for n in numbers if n == int(n)}, areas | numbers)


def check_answer(answer: str, evidence: Evidence, agent=None) -> list[str]:
    """문제 목록 (비어 있으면 통과)."""
    problems: list[str] = []
    a = (answer or "").strip()
    if not a:
        return ["빈 답변"]
    phone = office_phone()
    if _INTERNAL_ID.search(a):
        problems.append("내부 번호 노출")
    phones = [m.group(0) for m in _PHONE.finditer(a)]
    if any(re.sub(r"\D", "", p) != re.sub(r"\D", "", phone) for p in phones):
        problems.append("전화번호 노출")
    lots = [m.group(0) for m in _LOT.finditer(a)]
    if lots:
        problems.append("지번 노출: " + ", ".join(lots[:2]))
    prices, areas = set(evidence.prices), set(evidence.areas)
    if agent is not None:
        unseen = []
        for key, l in _listing_keys(agent).items():
            if key in evidence.text:  # 도구로 확인한 매물: 실제 장부의 가격·면적도 근거로 인정
                prices |= {int(l.price_manwon or 0), round(l.price_per_pyeong or 0)}
                areas |= {float(round(l.area_pyeong or 0)), float(round(l.area_m2 or 0))}
            elif re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(key)}(?![A-Za-z0-9_-])", a):
                unseen.append(key)  # 확인하지 않은 매물을 말함 (도구 결과에 없는 매물번호)
        if unseen:
            problems.append("확인하지 않은 매물: " + ", ".join(sorted(unseen)[:3]))
        from ..listing_numbers import known_numbers, unknown_in

        missing = unknown_in(a, known_numbers(agent))
        if missing:  # 매물 장부에 아예 없는 번호 (AI 가 지어냄)
            problems.append("없는 매물번호: " + ", ".join(sorted(set(missing))[:3]))
    bad_prices = sorted(p for p in prices_in(a) if p not in prices)
    if bad_prices:
        from ..models import format_price

        problems.append("도구 결과에 없는 가격: " + ", ".join(format_price(p) for p in bad_prices[:3]))
    bad_areas = []
    for m in list(_AREA.finditer(a)) + list(_M2.finditer(a)):
        v = _num(m.group(1))
        if not any(abs(v - e) <= max(1.0, e * 0.005) for e in areas):
            bad_areas.append(m.group(0))
    if bad_areas:
        problems.append("도구 결과에 없는 면적: " + ", ".join(bad_areas[:3]))
    return problems


_KEYS_CACHE: dict[int, tuple[int, dict]] = {}


def _listing_keys(agent) -> dict:
    """손님에게 말할 수 있는 매물 이름표들 (매물번호, 공개 id). 매물 목록이 바뀌면 다시 만듦."""
    from ..agent import public_office_no
    from ..tool_agent.tools import tool_id

    listings = list(agent.listings()) + list(getattr(agent, "hidden_listings", []))
    cached = _KEYS_CACHE.get(id(agent))
    if cached and cached[0] == hash(tuple(l.listing_id for l in listings)):
        return cached[1]
    out = {}
    for l in listings:
        for k in (public_office_no(l.office_no), tool_id(l)):
            if k and len(k) >= 3:
                out[k] = l
    _KEYS_CACHE[id(agent)] = (hash(tuple(l.listing_id for l in listings)), out)
    return out
