"""손님 문의(자연어) -> CustomerRequirement.

1순위: LLM(Claude / GPT) 으로 구조화 추출
2순위: LLM 이 없거나 실패하면 규칙 기반(정규식) 추출
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Optional

from .llm import LLMRouter, LLMUnavailable
from .models import JIMOK_TYPES, PYEONG_M2, CustomerRequirement
from .normalizer import JIMOK_ALIASES, parse_price_manwon
from .regions import RENAMED_REGIONS, canonical_region, eupmyeon_variants, find_regions, same_place

log = logging.getLogger(__name__)

ZONING_TYPES = [
    "계획관리지역", "생산관리지역", "보전관리지역", "농림지역", "자연환경보전지역",
    "자연녹지지역", "생산녹지지역", "보전녹지지역",
    "제1종전용주거지역", "제2종전용주거지역", "제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
    "준주거지역", "중심상업지역", "일반상업지역", "근린상업지역", "유통상업지역",
    "전용공업지역", "일반공업지역", "준공업지역",
]

PURPOSE_KEYWORDS = {
    "전원주택": ["전원주택", "주택", "집 지", "집을 지", "세컨하우스", "별장", "귀촌", "단독주택"],
    "창고": ["창고", "물류", "야적"],
    "공장": ["공장", "제조"],
    "농사": ["농사", "귀농", "농지", "주말농장", "텃밭", "과수", "경작", "농업"],
    "근린생활": ["카페", "상가", "음식점", "식당", "근생", "근린생활", "매장"],
    "캠핑장": ["캠핑", "글램핑", "야영"],
    "태양광": ["태양광", "발전소"],
    "투자": ["투자", "시세차익", "개발호재", "묻어", "장기보유"],
}

FEATURE_KEYWORDS = [
    "남향", "동향", "서향", "평지", "완경사", "계곡", "강", "하천", "호수", "전망", "조망", "산", "숲",
    "IC", "역세권", "마을", "학교", "토목", "전기", "지하수", "상수도", "정남향", "배산임수", "바다",
    "개발", "택지", "도로확장", "역", "고속도로", "관공서", "조용",
]

# 같은 뜻의 표현 -> 대표 키워드 (매물 설명과 비교할 때는 preference.FEATURE_SYNONYMS 로 다시 넓혀 찾음)
FEATURE_ALIASES = {
    "채광": "채광", "햇빛": "채광", "햇볕": "채광", "일조": "채광", "해가 잘": "채광", "해 잘": "채광",
    "볕이 잘": "채광", "볕 잘": "채광", "양지": "채광", "밝은": "채광",
    "뷰": "조망", "경치": "조망", "평탄": "평지", "개울": "계곡", "나들목": "IC", "한적": "조용",
}

# 이 프로그램은 토지 전용. 다른 매물 종류를 찾으면 안내 문구를 남김
NON_LAND_TYPES = ["아파트", "빌라", "오피스텔", "원룸", "투룸", "쓰리룸", "다세대", "연립", "상가건물", "전세", "월세"]

SYSTEM_PROMPT = f"""당신은 한국 토지 매물 중개 보조 AI 입니다.
손님의 문의 문장에서 토지 검색 조건을 추출합니다.

필드:
- customer_name: 손님 이름 (없으면 "")
- regions: 선호 지역 이름 목록. 시·군은 정식 이름으로 (예: "음성" -> "음성군", "여주" -> "여주시"), 읍·면·리는 그대로 (예: ["음성군", "감곡면"])
- jimok: 원하는 지목 목록. 다음 중에서만: {", ".join(JIMOK_TYPES)}
- zoning: 원하는 용도지역 목록. 다음 중에서만: {", ".join(ZONING_TYPES)}
- budget_min, budget_max: 예산 (단위: 만원, 정수). "3억 정도"는 budget_max 를 약 15% 여유있게(34500) 잡으세요.
  "2억원대"는 2억 이상 3억 미만 (budget_min 20000, budget_max 29999), "5천만원대"는 5000~5999 입니다.
- area_min_m2, area_max_m2: 면적 (㎡). 1평 = 3.305785㎡. "300평 정도"는 ±20%.
- purpose: 다음 중 하나 또는 "": {", ".join(PURPOSE_KEYWORDS)}
- need_road: 도로에 접한 땅이 필요하면 true (맹지 제외). 주택/창고/공장/근린생활 목적이면 기본 true.
- keywords: 원하는 특징 (남향, 계곡, IC 인근 등). "채광 좋은/햇빛 잘 드는"은 "채광", 경치는 "조망"으로 쓰세요.
- excluded_keywords: 원하지 않는 것 (예: "묘지", "고압선", "임야" 등)

모르는 값은 null 또는 빈 목록으로 두고, 추측하지 마세요."""


def _pyeong_or_m2(num: float, unit: str) -> float:
    return num * PYEONG_M2 if "평" in unit else num


class RequirementParser:
    def __init__(self, llm: Optional[LLMRouter] = None, known_regions: Iterable[str] = (),
                 addresses: Iterable[str] = ()):
        self.llm = llm
        self.known_regions = sorted({r for r in known_regions if len(r) >= 2}, key=len, reverse=True)
        self.addresses = list(addresses)  # 보유 매물 주소 ('음성군 대소면' 처럼 어느 읍·면이 어느 군에 속하는지 알기 위해)

    def parse(self, text: str, customer_name: str = "") -> CustomerRequirement:
        req: Optional[CustomerRequirement] = None
        if self.llm and self.llm.available:
            try:
                req = self._parse_llm(text)
            except LLMUnavailable as e:
                log.info("LLM 요구사항 추출 실패, 규칙 기반으로 전환: %s", e)
        if req is None:
            req = parse_rule_based(text, self.known_regions)
        req.raw_text = text
        req.notes = non_land_notes(text)
        if customer_name:
            req.customer_name = customer_name
        req = sanitize(req)
        req.regions = narrow_regions(req.regions, self.addresses)
        # '대소면' 이라고 썼는데 지금 이름이 '대소읍' 이면 알려 줌
        for r in req.regions:
            old = next((v for v in eupmyeon_variants(r) if v != r and v in text and r not in text), None)
            if old:
                req.notes.append(f"'{old}'은(는) 지금 '{r}'(으)로 이름이 바뀌어 {r} 매물을 찾았습니다.")
        return req

    def _parse_llm(self, text: str) -> CustomerRequirement:
        hint = ""
        if self.known_regions:
            hint = "\n\n(참고: 보유 매물 지역명 예시 - " + ", ".join(self.known_regions[:80]) + ")"
        data = self.llm.complete_json(SYSTEM_PROMPT, f"손님 문의:\n{text}{hint}", max_tokens=1500)
        return CustomerRequirement.from_dict(data)

    def apply_feedback(self, req: CustomerRequirement, feedback: str) -> CustomerRequirement:
        """'좀 더 싼 거', '더 넓은 땅', '도로 있는 걸로', '여주 쪽도' 같은 피드백을 조건에 반영."""
        if self.llm and self.llm.available:
            try:
                data = self.llm.complete_json(
                    SYSTEM_PROMPT + "\n\n기존 조건과 손님의 추가 피드백이 주어집니다. 피드백을 반영해 수정된 "
                                    "전체 조건을 같은 형식의 JSON 으로 출력하세요.",
                    f"기존 조건:\n{json.dumps(req.to_dict(), ensure_ascii=False)}\n\n피드백:\n{feedback}",
                    max_tokens=1500,
                )
                new = CustomerRequirement.from_dict({**req.to_dict(), **data})
                new.raw_text = req.raw_text + "\n[피드백] " + feedback
                return sanitize(new)
            except LLMUnavailable as e:
                log.info("LLM 피드백 반영 실패, 규칙 기반으로 전환: %s", e)
        return apply_feedback_rule_based(req, feedback, self.known_regions)


def narrow_regions(regions: list[str], addresses: Iterable[str]) -> list[str]:
    """'음성군 대소면' -> ['대소면']: 같이 적은 읍·면·동·리가 그 시·군 안에 있으면 좁은 쪽만 남김.
    ('여주 양평' 처럼 서로 다른 지역은 그대로 둘 다)"""
    small = [r for r in regions if r[-1:] in "읍면동리"]
    if not small:
        return regions
    addresses = list(addresses)
    drop = set()
    for big in regions:
        if big[-1:] not in "도시군구" or big in small:
            continue
        for s in small:
            if any(big in a and any(same_place(s, tok) for tok in a.split()) for a in addresses) or (
                    not addresses and s.startswith(re.sub(r"(시|군|구)$", "", big))):
                drop.add(big)
                break
    return [r for r in regions if r not in drop]


def sanitize(req: CustomerRequirement) -> CustomerRequirement:
    """LLM 출력값 검증/보정."""
    def _list(v) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()]

    # '음성' -> '음성군', '여주' -> '여주시' (GPT 가 줄여서 돌려줘도 정식 시·군 이름으로)
    regions: list[str] = []
    for r in _list(req.regions):
        c = canonical_region(r)
        if c and c not in regions:
            regions.append(c)
    req.regions = regions
    req.jimok = [JIMOK_ALIASES.get(j, j) for j in _list(req.jimok)]
    req.jimok = [j for j in req.jimok if j in JIMOK_TYPES]
    req.zoning = [z if z.endswith("지역") else z + "지역" for z in _list(req.zoning)]
    req.keywords = _list(req.keywords)
    req.excluded_keywords = _list(req.excluded_keywords)
    for f in ("budget_min", "budget_max"):
        v = getattr(req, f)
        setattr(req, f, int(v) if isinstance(v, (int, float)) and v > 0 else None)
    for f in ("area_min_m2", "area_max_m2"):
        v = getattr(req, f)
        setattr(req, f, float(v) if isinstance(v, (int, float)) and v > 0 else None)
    if req.budget_min and req.budget_max and req.budget_min > req.budget_max:
        req.budget_min, req.budget_max = req.budget_max, req.budget_min
    if req.area_min_m2 and req.area_max_m2 and req.area_min_m2 > req.area_max_m2:
        req.area_min_m2, req.area_max_m2 = req.area_max_m2, req.area_min_m2
    req.purpose = str(req.purpose or "").strip()
    if req.purpose and req.purpose not in PURPOSE_KEYWORDS:
        req.purpose = next((p for p, kws in PURPOSE_KEYWORDS.items() if any(k in req.purpose for k in kws)),
                           req.purpose)
    req.need_road = bool(req.need_road)
    return req


# 지역명/지목 단어 뒤에 올 수 있는 조사·표현
_PARTICLE_END = r"(?=\s|,|·|/|$|이나|나|이랑|랑|과|와|도|이|가|은|는|을|를|만|쪽|으로|로|에)"
_REGION_END = r"(?=\s|,|·|/|$|이나|나|이랑|랑|과|와|도|쪽|근처|인근|일대|에|으로|로|권)"

_NOT_REGION = {"혹시", "역시", "다시", "도시", "당시", "수시", "즉시", "항시", "동시", "표시", "무시", "지시",
               "개시", "게시", "감시", "전시", "평시", "일시", "상시", "아군", "대군", "우리군", "여러군", "몇군", "한군",
               "두군", "세군"}

_MONEY = r"(\d[\d,.]*\s*억(?:\s*\d[\d,.]*\s*(?:천)?)?(?:\s*만)?|\d[\d,.]*\s*천\s*만?|\d[\d,.]*\s*만)\s*원?"
_AREA = r"(\d[\d,.]*)\s*(평|㎡|m2|m²|제곱미터)"


def _money(s: str) -> Optional[int]:
    return parse_price_manwon(s.replace(" ", ""))


def parse_rule_based(text: str, known_regions: Iterable[str] = ()) -> CustomerRequirement:
    t = text
    req = CustomerRequirement()

    m = re.search(r"([가-힣]{2,4})\s*(?:님|고객|손님|사장님|씨)", t)
    if m and m.group(1) not in {"고객", "손님", "사장"}:
        req.customer_name = m.group(1)

    # --- 예산 ---
    m = re.search(_MONEY + r"\s*(?:~|-|에서|부터)\s*" + _MONEY, t)
    band = _price_band(t)  # '2억원대', '5천만원대'
    if band:
        req.budget_min, req.budget_max = band
    elif m:
        req.budget_min, req.budget_max = _money(m.group(1)), _money(m.group(2))
    else:
        m = re.search(_MONEY + r"\s*(이하|이내|까지|안쪽|미만|밑|아래)", t)
        if m:
            req.budget_max = _money(m.group(1))
        m2 = re.search(_MONEY + r"\s*(이상|넘는|초과)", t)
        if m2:
            req.budget_min = _money(m2.group(1))
        if not m and not m2:
            m = re.search(r"(?:예산|자금|가격|금액)\D{0,6}" + _MONEY, t) or re.search(_MONEY + r"\s*(정도|쯤|내외|선)", t)
            if m:
                v = _money(m.group(1))
                if v:
                    req.budget_max = int(v * 1.15)
                    req.budget_min = int(v * 0.6)

    # --- 면적 ---
    area_band = re.search(r"(\d[\d,]*)\s*(평|㎡|m2)\s*대(?![가-힣])", t)  # '200평대' = 200~299평
    m = None if area_band else re.search(_AREA.replace(r"\s*(평|㎡|m2|m²|제곱미터)", r"\s*(평|㎡|m2|m²|제곱미터)?")
                  + r"\s*(?:~|-|에서)\s*" + _AREA, t)
    if area_band:
        n = float(area_band.group(1).replace(",", ""))
        step = 10 ** max(len(str(int(n))) - 1, 1)  # 200평대 -> 100 단위, 50평대 -> 10 단위
        req.area_min_m2 = _pyeong_or_m2(n, area_band.group(2))
        req.area_max_m2 = _pyeong_or_m2(n + step - 1, area_band.group(2))
    elif m:
        unit = m.group(4)
        req.area_min_m2 = _pyeong_or_m2(float(m.group(1).replace(",", "")), m.group(2) or unit)
        req.area_max_m2 = _pyeong_or_m2(float(m.group(3).replace(",", "")), unit)
    else:
        for m in re.finditer(_AREA + r"\s*(이상|넘는|이하|이내|까지|정도|쯤|내외|안팎)?", t):
            v = _pyeong_or_m2(float(m.group(1).replace(",", "")), m.group(2))
            q = m.group(3) or "정도"
            if q in {"이상", "넘는"}:
                req.area_min_m2 = v
            elif q in {"이하", "이내", "까지"}:
                req.area_max_m2 = v
            else:
                req.area_min_m2, req.area_max_m2 = v * 0.8, v * 1.2

    # --- 지목 ---
    for j in JIMOK_TYPES:
        if len(j) >= 2 and j in t and j not in {"도로", "하천", "공원"}:
            req.jimok.append(j)
    for alias, j in JIMOK_ALIASES.items():
        if alias in {"공장", "창고"}:  # '창고 지을 땅' 은 지목이 아니라 목적(purpose)으로 처리
            continue
        if re.search(rf"(?<![가-힣]){alias}{_PARTICLE_END}", t) and j not in req.jimok:
            req.jimok.append(j)
    for m in re.finditer(r"지목\s*[은는이가:]?\s*(전|답|대)(?![가-힣])", t):
        if m.group(1) not in req.jimok:
            req.jimok.append(m.group(1))

    # --- 용도지역 ---
    for z in ZONING_TYPES:
        short = z.replace("지역", "")
        if z in t or (len(short) >= 4 and short in t):
            req.zoning.append(z)

    # --- 목적 ---
    for purpose, kws in PURPOSE_KEYWORDS.items():
        if any(k in t for k in kws):
            req.purpose = purpose
            break

    # --- 도로 ---
    if re.search(r"맹지\s*(는|은)?\s*(제외|말고|빼고|안|싫|불가|x|X)|도로\s*(접|있|붙|끼|변)|진입로", t):
        req.need_road = True
    if req.purpose in {"전원주택", "창고", "공장", "근린생활"}:
        req.need_road = True

    # --- 제외 조건 ---
    for m in re.finditer(r"([가-힣A-Za-z]{1,8})\s*(?:은|는|이|가)?\s*(?:제외|말고|빼고|싫어|싫고|안\s*돼|피해)", t):
        w = re.sub(r"(은|는|이|가|을|를)$", "", m.group(1))
        if not w or w in {"맹지", "도로", "곳", "거", "것", "땅", "데", "건"}:
            continue
        j = JIMOK_ALIASES.get(w, w)
        if j in req.jimok:
            req.jimok.remove(j)
        req.excluded_keywords.append(w)
    for w in ("묘지", "분묘", "고압선", "송전탑", "축사"):
        if w in t and re.search(w + r".{0,6}(없|제외|싫|말고|안)", t) and w not in req.excluded_keywords:
            req.excluded_keywords.append(w)

    # --- 지역 ---
    regions: list[str] = []
    for r in known_regions:
        # 시·군·구는 줄여 써도 인정 ('음성' -> 음성군), 읍·면·동·리는 전체 이름을 써야 인정 ('음성' 이 음성읍이 되지 않도록)
        # 읍·면은 승격 전/후 이름 모두 인정 ('대소면' 이라고 써도 보유 매물의 '대소읍')
        stem = re.sub(r"(시|군|구)$", "", r) if r[-1:] in "시군구" else r
        typed = any(v in t for v in eupmyeon_variants(r))
        if (typed or (len(stem) >= 2 and re.search(rf"{re.escape(stem)}{_REGION_END}", t))) \
                and not any(r in x or same_place(r, x) for x in regions):
            regions.append(r)
    for old, new in RENAMED_REGIONS.items():  # 보유 매물에 아직 없어도 옛 이름은 지금 이름으로
        if old in t and not any(same_place(new, x) for x in regions):
            regions.append(new)
    # 전국 시·도 / 시·군 이름표 (보유 매물에 없는 지역도 지역 조건으로 인정 -> '현재 존재하지 않습니다' 안내)
    for w in find_regions(t):
        if not any(w in x or x in w for x in regions):
            regions.append(w)
    # 이름표에 없더라도 '○○시/○○군' 으로 명시하면 지역 조건으로 인정
    for m in re.finditer(r"(?<![가-힣])([가-힣]{1,3}(?:시|군))" + _REGION_END, t):
        w = m.group(1)
        if w in _NOT_REGION or any(w in x or x in w for x in regions):
            continue
        regions.append(w)
    # '음성군채광이' 처럼 띄어쓰기 없이 붙여 쓴 '○○군' (군으로 끝나는 일반 단어는 드묾)
    for m in re.finditer(r"(?<![가-힣])([가-힣]{1,3}군)(?=[가-힣])", t):
        w = m.group(1)
        if w not in _NOT_REGION and not any(w in x or x in w for x in regions):
            regions.append(w)
    # '음성읍 땅' 처럼 읍·면을 콕 집었으면 그 읍·면만 (음성군 전체로 넓히지 않음)
    for big in [x for x in regions if x[-1:] in "시군"]:
        stem = big[:-1]
        if big not in t and any(x != big and x[-1:] in "읍면동리" and x.startswith(stem) for x in regions):
            regions.remove(big)
    req.regions = [canonical_region(r) for r in regions]

    # --- 특징 키워드 ---
    for k in FEATURE_KEYWORDS:
        if k in t and k not in req.excluded_keywords and not any(k in r for r in req.regions):
            if len(k) == 1 and not re.search(rf"(?<![가-힣]){k}(?![가-힣])", t):
                continue
            req.keywords.append(k)
    for phrase, k in FEATURE_ALIASES.items():
        if phrase in t and k not in req.keywords:
            req.keywords.append(k)
    return req


def _price_band(t: str) -> Optional[tuple[int, int]]:
    """'2억원대' -> (20000, 29999), '2억 5천만원대' -> (25000, 25999), '5천만원대' -> (5000, 5999)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*억\s*(?:(\d+)\s*천)?\s*(?:만)?\s*원?\s*대(?![가-힣])", t)
    if m:
        eok = float(m.group(1))
        if m.group(2):
            lo = int(eok * 10000 + int(m.group(2)) * 1000)
            return lo, lo + 999
        lo = int(eok * 10000)
        return lo, lo + 9999
    m = re.search(r"(\d+)\s*천\s*(?:만)?\s*원?\s*대(?![가-힣])", t)
    if m:
        lo = int(m.group(1)) * 1000
        return lo, lo + 999
    return None


def non_land_notes(text: str) -> list[str]:
    found = [w for w in NON_LAND_TYPES if w in text]
    if not found:
        return []
    return [f"'{', '.join(found)}'은(는) 취급 범위(토지) 밖이라, 토지 매물 중에서 나머지 조건으로 찾습니다."]


def apply_feedback_rule_based(req: CustomerRequirement, feedback: str,
                              known_regions: Iterable[str] = ()) -> CustomerRequirement:
    new = CustomerRequirement.from_dict(req.to_dict())
    f = feedback
    if re.search(r"(싼|저렴|비싸|가격.*(낮|내려)|예산.*(줄|낮))", f):
        base = new.budget_max or None
        if base:
            new.budget_max = int(base * 0.85)
    if re.search(r"(예산.*(늘|올|높)|더 비싸도|좀 더 써)", f):
        if new.budget_max:
            new.budget_max = int(new.budget_max * 1.2)
    if re.search(r"(넓|크|큰)", f):
        if new.area_min_m2:
            new.area_min_m2 *= 1.2
        elif new.area_max_m2:
            new.area_min_m2 = new.area_max_m2
            new.area_max_m2 = None
        if new.area_max_m2:
            new.area_max_m2 *= 1.3
    if re.search(r"(좁|작|작은)", f):
        if new.area_max_m2:
            new.area_max_m2 *= 0.8
        elif new.area_min_m2:
            new.area_max_m2 = new.area_min_m2
            new.area_min_m2 = None
    if re.search(r"(도로|맹지)", f):
        new.need_road = True

    extra = parse_rule_based(f, known_regions)
    for r in extra.regions:
        if r not in new.regions:
            new.regions.append(r)
    if re.search(r"(지역|동네).*(말고|다른|바꿔)", f) and extra.regions:
        new.regions = extra.regions
    for j in extra.jimok:
        if j not in new.jimok:
            new.jimok.append(j)
    for z in extra.zoning:
        if z not in new.zoning:
            new.zoning.append(z)
    for k in extra.keywords:
        if k not in new.keywords:
            new.keywords.append(k)
    for k in extra.excluded_keywords:
        if k not in new.excluded_keywords:
            new.excluded_keywords.append(k)
        if k in new.regions:
            new.regions.remove(k)
    if extra.purpose:
        new.purpose = extra.purpose
    if extra.budget_max and not re.search(r"(싼|저렴|비싸)", f):
        new.budget_max = extra.budget_max
    new.raw_text = req.raw_text + "\n[피드백] " + feedback
    return sanitize(new)
