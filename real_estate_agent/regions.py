"""전국 시·도 / 시·군 이름표.

- 손님 문장에서 지역을 찾을 때: 보유 매물에 없는 지역("부산 땅")도 지역 조건으로 알아봐야
  '해당 조건에 맞는 매물이 없습니다' 라고 정확히 답할 수 있습니다.
- 매물 주소와 비교할 때: '경북' 으로 찾으면 '경상북도 …' 주소나, 시·도 없이 '문경시 …' 로만 적힌 주소도 찾습니다.
"""

from __future__ import annotations

import re

SIDO_ALIASES: dict[str, list[str]] = {
    "서울": ["서울특별시", "서울시", "서울"],
    "부산": ["부산광역시", "부산시", "부산"],
    "대구": ["대구광역시", "대구시", "대구"],
    "인천": ["인천광역시", "인천시", "인천"],
    "광주": ["광주광역시", "광주광역"],  # '광주' 단독은 경기 광주시와 겹쳐 시·군 쪽에서 처리
    "대전": ["대전광역시", "대전시", "대전"],
    "울산": ["울산광역시", "울산시", "울산"],
    "세종": ["세종특별자치시", "세종시", "세종"],
    "경기": ["경기도", "경기"],
    "강원": ["강원특별자치도", "강원도", "강원"],
    "충북": ["충청북도", "충북"],
    "충남": ["충청남도", "충남"],
    "전북": ["전북특별자치도", "전라북도", "전북"],
    "전남": ["전라남도", "전남"],
    "경북": ["경상북도", "경북"],
    "경남": ["경상남도", "경남"],
    "제주": ["제주특별자치도", "제주도"],
}

SIGUNGU: dict[str, list[str]] = {
    "경기": ["수원시", "성남시", "의정부시", "안양시", "부천시", "광명시", "평택시", "동두천시", "안산시", "고양시",
             "과천시", "구리시", "남양주시", "오산시", "시흥시", "군포시", "의왕시", "하남시", "용인시", "파주시",
             "이천시", "안성시", "김포시", "화성시", "광주시", "양주시", "포천시", "여주시", "연천군", "가평군", "양평군"],
    "강원": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시", "홍천군", "횡성군", "영월군",
             "평창군", "정선군", "철원군", "화천군", "양구군", "인제군", "고성군", "양양군"],
    "충북": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군", "진천군", "괴산군", "음성군", "단양군"],
    "충남": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시", "금산군", "부여군",
             "서천군", "청양군", "홍성군", "예산군", "태안군"],
    "전북": ["전주시", "군산시", "익산시", "정읍시", "남원시", "김제시", "완주군", "진안군", "무주군", "장수군",
             "임실군", "순창군", "고창군", "부안군"],
    "전남": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군", "구례군", "고흥군", "보성군",
             "화순군", "장흥군", "강진군", "해남군", "영암군", "무안군", "함평군", "영광군", "장성군", "완도군",
             "진도군", "신안군"],
    "경북": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시",
             "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군",
             "울진군", "울릉군"],
    "경남": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시", "양산시", "의령군", "함안군",
             "창녕군", "고성군", "남해군", "하동군", "산청군", "함양군", "거창군", "합천군"],
    "제주": ["제주시", "서귀포시"],
    "부산": ["기장군"],
    "대구": ["달성군", "군위군"],
    "인천": ["강화군", "옹진군"],
    "울산": ["울주군"],
}

# '시/군' 을 빼고 써도 지역으로 볼 이름 (예: '문경 땅', '밀양 쪽'). 일반 단어와 겹치는 이름은 제외
_AMBIGUOUS_STEMS = {"동해", "남해", "구리", "장수", "영광", "보성", "화성", "광주", "고성", "진도", "부여",
                    "신안", "예산", "청양", "광양", "보은", "의령", "고령", "진안", "영동", "옥천", "청도",
                    "하동", "안성", "양주", "성주", "상주", "무안"}

ALL_SIGUNGU = sorted({n for names in SIGUNGU.values() for n in names}, key=len, reverse=True)
SIGUNGU_TO_SIDO: dict[str, set[str]] = {}
for _sido, _names in SIGUNGU.items():
    for _n in _names:
        SIGUNGU_TO_SIDO.setdefault(_n, set()).add(_sido)

# 지역 이름 뒤에 올 수 있는 말: 조사·방향 표현, 그리고 띄어쓰기 없이 붙여 쓴 땅 관련 단어 ('음성창고', '여주토지')
_END = (r"(?=\s|,|·|/|$|이나|나|이랑|랑|과|와|도|쪽|근처|인근|일대|에|으로|로|권|의|지역|땅|토지|부지|매물|임야|"
        r"농지|논|밭|대지|창고|공장|전원|주택|물류|카페|상가|축사|태양광|캠핑|[가-힣]{0,2}면|[가-힣]{0,2}읍)")


def _stem(name: str) -> str:
    return re.sub(r"(시|군)$", "", name)


def sido_of(alias_or_name: str) -> str | None:
    """'경상북도' / '경북' -> '경북'"""
    for key, aliases in SIDO_ALIASES.items():
        if alias_or_name in aliases or alias_or_name == key:
            return key
    return None


def find_regions(text: str) -> list[str]:
    """문장에서 시·도 / 시·군 이름을 찾음. 예) '경북 문경 땅' -> ['경북', '문경시']"""
    found: list[str] = []
    compact = text
    for name in ALL_SIGUNGU:
        stem = _stem(name)
        hit = name in compact or (
            len(stem) >= 2 and stem not in _AMBIGUOUS_STEMS
            and re.search(rf"(?<![가-힣]){re.escape(stem)}{_END}", compact)
        )
        if hit and name not in found:
            found.append(name)
    for key, aliases in SIDO_ALIASES.items():
        for a in sorted(aliases, key=len, reverse=True):
            if re.search(rf"(?<![가-힣]){re.escape(a)}{_END}", compact) or (len(a) > 3 and a in compact):
                if key == "광주" and "광주시" in found:
                    continue
                if key not in found:
                    found.append(key)
                break
    # '세종시' 처럼 시·도이면서 '○○시' 로 잡힌 경우 정리
    return found


def address_in_region(address: str, region: str) -> bool:
    """매물 주소가 손님이 말한 지역에 속하는지."""
    addr = address.replace(" ", "")
    r = region.replace(" ", "")
    if not r:
        return True
    if r in addr:
        return True
    sido = sido_of(r)
    if sido:  # 시·도: 별칭 또는 소속 시·군이 주소에 있으면 일치
        if any(a in addr for a in SIDO_ALIASES[sido] if len(a) >= 2 and a != "광주광역"):
            return True
        return any(n in addr for n in SIGUNGU.get(sido, []))
    stem = _stem(r)
    if r in SIGUNGU_TO_SIDO and len(stem) >= 2:  # '문경시' 요청 <-> 주소 '문경 …' 처럼 시/군을 뺀 표기
        return bool(re.search(re.escape(stem) + r"(시|군)?(?=[가-힣]{1,4}(읍|면|동)|$)", addr))
    return False


def canonical_region(name: str) -> str:
    """'음성' / '음성시'(잘못된 표기) / '음성 군' -> '음성군', '여주' -> '여주시', '경상북도' -> '경북'.
    시·군 이름표에 없으면(읍·면·리 이름 등) 그대로 둠."""
    n = re.sub(r"\s+", "", str(name or ""))
    if not n:
        return n
    if n in SIGUNGU_TO_SIDO:
        return n
    sido = sido_of(n)
    if sido:
        return sido
    stem = _stem(n)
    matches = [c for c in ALL_SIGUNGU if _stem(c) == stem]
    if len(set(matches)) == 1 and len(stem) >= 2:
        return matches[0]
    if len(matches) > 1:  # '고성' 처럼 여러 곳 -> 원래 쓴 표기가 정확하면 그것, 아니면 그대로
        return n
    return n
