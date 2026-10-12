"""목적별 '지을 수 있는 땅인가' 판단 (예: 창고를 지을 수 있는 토지).

손님이 "창고 지을 땅", "공장 부지", "전원주택 지을 땅" 처럼 **용도**를 말하면, 그 용도로 쓸 수 없는 땅은
점수가 아무리 높아도 추천하지 않습니다. 판단 순서:

  1) 매물 설명/폴더 이름에 그 용도가 직접 적혀 있음  (예: '창고 건축허가 득', '(물류부지)')  -> 가능
  2) 지목이 이미 그 용도                              (예: 창고용지, 공장용지)           -> 가능
  3) 용도지역이 그 건축물을 허용                       (국토계획법 시행령 별표 기준 요약)   -> 가능 (지목이 전·답·임야면 '전용허가 필요' 안내)
  4) 용도지역 미기재 + 지목이 대·잡종지                                                  -> 가능성 있음 ('용도지역 확인 필요')
  5) 그 외 (농림지역·보전관리지역의 창고 등)                                             -> 불가

※ 실제 건축 가능 여부는 토지이용계획확인원·지자체 조례로 최종 확인해야 합니다. 화면에도 그렇게 안내합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import LandListing

# 용도지역별 허용 (일반적인 기준. 조례에 따라 다를 수 있음)
_INDUSTRIAL = {"전용공업지역", "일반공업지역", "준공업지역"}
_COMMERCIAL = {"중심상업지역", "일반상업지역", "근린상업지역", "유통상업지역"}
_RESIDENTIAL = {"제1종전용주거지역", "제2종전용주거지역", "제1종일반주거지역", "제2종일반주거지역",
                "제3종일반주거지역", "준주거지역"}

RULES: dict[str, dict] = {
    "창고": {
        "words": r"창고|물류|야적|택배|저온저장",
        "jimok": {"창고용지", "공장용지"},
        "zoning": {"계획관리지역", "자연녹지지역", "준주거지역"} | _INDUSTRIAL | (_COMMERCIAL - {"중심상업지역"}),
        "maybe_jimok": {"대", "잡종지"},
    },
    "공장": {
        "words": r"공장|제조|산업용|공장부지",
        "jimok": {"공장용지"},
        "zoning": {"계획관리지역"} | _INDUSTRIAL,
        "maybe_jimok": {"잡종지", "창고용지"},
    },
    "근린생활": {
        "words": r"근린생활|근생|카페|상가|음식점|식당|점포",
        "jimok": {"대"},
        "zoning": {"계획관리지역", "자연녹지지역", "생산관리지역"} | _COMMERCIAL | _RESIDENTIAL | _INDUSTRIAL,
        "maybe_jimok": {"잡종지"},
    },
    "전원주택": {
        "words": r"전원주택|주택\s*건축|주택\s*부지|주택지|택지|세컨하우스|단독주택",
        "jimok": {"대"},
        "zoning": {"계획관리지역", "생산관리지역", "보전관리지역", "자연녹지지역", "생산녹지지역", "보전녹지지역",
                   "농림지역"} | _RESIDENTIAL,
        "maybe_jimok": {"잡종지", "전", "답", "과수원"},
    },
}
NEEDS_CONVERSION = {"전", "답", "과수원", "임야", "목장용지"}  # 다른 용도로 쓰려면 농지·산지 전용허가 필요


@dataclass
class UseCheck:
    ok: bool
    reason: str  # 화면에 보여줄 근거
    caution: str = ""  # 확인이 필요한 점


def check_use(listing: LandListing, purpose: str) -> Optional[UseCheck]:
    """purpose 에 대한 규칙이 없으면 None (판단하지 않음)."""
    rule = RULES.get(purpose)
    if not rule:
        return None
    text = f"{listing.description} {listing.extra.get('_매물폴더', '')} {listing.extra.get('duplicate_sources', '')}"
    negated = re.search(rf"({rule['words']})\s*(불가|안\s*됨|불허|금지|X)", text)
    if re.search(rule["words"], text) and not negated:
        return UseCheck(True, f"매물 정보에 '{purpose}' 용도 명시")
    if negated:
        return UseCheck(False, f"매물 정보에 '{purpose}' 불가 표시")
    if listing.jimok in rule["jimok"]:
        return UseCheck(True, f"지목 '{listing.jimok}'")
    if listing.zoning:
        if listing.zoning in rule["zoning"]:
            caution = (f"지목 '{listing.jimok}' → {purpose} 건축 시 전용허가 필요"
                       if listing.jimok in NEEDS_CONVERSION else "")
            return UseCheck(True, f"'{listing.zoning}'에서 {purpose} 건축 가능", caution)
        return UseCheck(False, f"'{listing.zoning}'에서는 {purpose} 건축이 어려움")
    if listing.jimok in rule["maybe_jimok"]:
        return UseCheck(True, f"지목 '{listing.jimok}'", "용도지역 미기재 - 토지이용계획 확인 필요")
    return UseCheck(False, "용도지역 정보가 없어 판단 불가")
