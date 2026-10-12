"""피드백 학습 (학습형 추천).

손님이 매물을 '거절'하거나 '관심(좋아요)' 표시할 때마다 두 가지를 배웁니다.

1) 항목 중요도 (가중치)  - 온라인 학습
   거절한 매물이 다른 후보보다 '잘했던' 항목은 중요도를 낮추고, '못했던' 항목은 높입니다.
   예) 가격은 싼데 면적이 작은 땅을 거절  ->  가격 중요도 ↓, 면적 중요도 ↑
   관심 표시한 매물은 반대로 강점 항목의 중요도를 높입니다.
   거절 사유("너무 비싸요", "도로가 별로")가 있으면 해당 항목을 직접 크게 올립니다.

2) 속성 선호  - 지목·용도지역·특징(남향, 계곡 ...) 별 좋아요/싫어요 횟수
   (좋아요 - 싫어요) / (전체 + 2) 비율로 최대 ±6점 가감 (데이터가 적을 때 과하게 반응하지 않도록 보정)

손님별 모델은 SQLite 에 저장되어 다음 상담에도 이어지고,
사무실 전체 피드백을 모은 '공통 모델'이 처음 오는 손님의 기본값 역할을 합니다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

BASE_WEIGHTS = {
    "region": 15, "price": 25, "area": 20, "jimok": 15, "zoning": 10, "road": 5, "keyword": 5, "fresh": 5,
}
CRITERIA_KO = {
    "region": "지역", "price": "가격", "area": "면적", "jimok": "지목", "zoning": "용도지역",
    "road": "도로", "keyword": "특징(채광·조망 등)", "fresh": "최신 매물",
}
LEARNING_RATE = 0.6
REASON_BOOST = 1.4  # 거절 사유로 명시된 항목은 중요도 40% ↑
MULT_MIN, MULT_MAX = 0.4, 3.0
ATTR_BONUS_MAX = 6.0

# 거절/피드백 사유 -> 항목
REASON_RULES: list[tuple[str, str]] = [
    (r"비싸|가격|예산|싼|저렴|돈", "price"),
    (r"좁|작|넓|크|면적|평수", "area"),
    (r"멀|위치|지역|동네|거리|교통", "region"),
    (r"도로|맹지|진입|길", "road"),
    (r"지목|임야|논|밭|농지|대지", "jimok"),
    (r"용도지역|관리지역|녹지|건축|허가", "zoning"),
    (r"채광|햇빛|일조|어둡|그늘|북향|남향|조망|전망|경사|평지|계곡|소음", "keyword"),
    (r"오래|예전|최근", "fresh"),
]

# 특징 키워드 동의어 (매물 설명에서 찾을 때 함께 인정)
FEATURE_SYNONYMS: dict[str, list[str]] = {
    "채광": ["채광", "남향", "정남향", "남동향", "남서향", "일조", "햇빛", "햇볕", "양지", "볕이 잘", "볕 잘"],
    "남향": ["남향", "정남향", "남동향", "남서향"],
    "조망": ["조망", "전망", "뷰", "view"],
    "평지": ["평지", "평탄"],
    "완경사": ["완경사", "경사 완만"],
    "계곡": ["계곡", "개울", "냇가"],
    "강": ["강", "강변", "하천", "천변", "북한강", "남한강"],
    "IC": ["IC", "나들목", "톨게이트"],
    "역세권": ["역세권", "역 도보", "역 인근", "역까지", "역 차량"],
    "조용": ["조용", "한적", "프라이빗"],
    "전기": ["전기"],
    "지하수": ["지하수"],
    "상수도": ["상수도"],
    "토목": ["토목"],
}


def feature_hit(keyword: str, text: str) -> bool:
    words = FEATURE_SYNONYMS.get(keyword, [keyword])
    low = text.lower()
    return any(w.lower() in low for w in words)


def listing_features(text: str) -> list[str]:
    return [k for k in FEATURE_SYNONYMS if feature_hit(k, text)]


@dataclass
class PreferenceModel:
    multipliers: dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in BASE_WEIGHTS})
    # {"jimok": {"전": [좋아요, 싫어요]}, "zoning": {...}, "feature": {...}}
    attrs: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {"jimok": {}, "zoning": {}, "feature": {}})
    n_feedback: int = 0

    # ------------------------------------------------------------ 계산
    def weights(self, prior: Optional["PreferenceModel"] = None) -> dict[str, float]:
        """BASE 가중치에 학습된 배수를 곱하고 합이 100 이 되도록 정규화."""
        raw = {}
        for k, base in BASE_WEIGHTS.items():
            m = self.multipliers.get(k, 1.0)
            if prior is not None:
                m *= math.sqrt(prior.multipliers.get(k, 1.0))  # 공통 모델은 절반 강도로 반영
            raw[k] = base * m
        total = sum(raw.values()) or 1.0
        return {k: v * 100 / total for k, v in raw.items()}

    def attribute_bonus(self, jimok: str, zoning: str, text: str) -> tuple[float, list[str]]:
        bonus, notes = 0.0, []
        items = [("jimok", jimok), ("zoning", zoning)] + [("feature", f) for f in listing_features(text)]
        for kind, value in items:
            if not value:
                continue
            likes, dislikes = self.attrs.get(kind, {}).get(value, [0, 0])
            if likes + dislikes == 0:
                continue
            b = (likes - dislikes) / (likes + dislikes + 2) * (ATTR_BONUS_MAX / 2)
            bonus += b
            if abs(b) >= 0.9:
                notes.append(f"이전 피드백상 '{value}' {'선호' if b > 0 else '비선호'}")
        bonus = max(-ATTR_BONUS_MAX, min(ATTR_BONUS_MAX, bonus))
        return bonus, notes

    # ------------------------------------------------------------ 학습
    def learn(self, action: str, factors: dict[str, float], others: Iterable[dict[str, float]],
              jimok: str = "", zoning: str = "", text: str = "", reason: str = "",
              requested: Iterable[str] = (), protected: Iterable[str] = ()) -> list[str]:
        """action: 'reject' 또는 'like'. 바뀐 내용을 사람이 읽을 수 있는 문장으로 반환.

        requested : 손님이 직접 말한 조건 항목 (예: {"region", "price", "keyword"}).
                    거절 한 번으로 손님이 '직접 요구한 항목'의 중요도가 크게 떨어지지 않도록 감쇠.
        protected : 손님이 직접 원한 속성값 (예: {"채광", "대"}). 거절해도 '비선호'로 배우지 않음.
        """
        others = [o for o in others if o]
        sign = -1.0 if action == "reject" else 1.0
        requested, protected = set(requested), set(protected)
        reason_crits = reason_criteria(reason)
        before = dict(self.multipliers)

        if others:
            for k in BASE_WEIGHTS:
                if k not in factors:
                    continue
                mean = sum(o.get(k, factors[k]) for o in others) / len(others)
                step = sign * LEARNING_RATE * (factors[k] - mean)  # 이 매물이 다른 후보보다 그 항목에서 얼마나 나았나
                if step < 0 and (k in requested or k in reason_crits):
                    step *= 0.3
                self.multipliers[k] = self.multipliers.get(k, 1.0) * math.exp(step)

        for crit in reason_crits:  # "너무 비싸요" -> 가격 중요도 ↑
            self.multipliers[crit] = self.multipliers.get(crit, 1.0) * REASON_BOOST

        for k in self.multipliers:
            self.multipliers[k] = max(MULT_MIN, min(MULT_MAX, self.multipliers[k]))

        # 속성 선호: 좋아요는 매물의 모든 속성에 +1.
        # 거절은 이유를 모르므로 +0.5 만, 사유가 지목/용도지역/특징을 가리키면 그 종류에만 +1.
        kinds_by_reason = {"jimok": "jimok", "zoning": "zoning", "keyword": "feature"}
        blamed = {kinds_by_reason[c] for c in reason_crits if c in kinds_by_reason}
        items = [("jimok", jimok), ("zoning", zoning)] + [("feature", f) for f in listing_features(text)]
        for kind, value in items:
            if not value:
                continue
            if action == "like":
                self.attrs.setdefault(kind, {}).setdefault(value, [0.0, 0.0])[0] += 1
                continue
            if value in protected or (kind == "feature" and _related_to_any(value, protected)):
                continue  # 손님이 원한 특징과 그 동의어(채광 ↔ 남향)는 비선호로 배우지 않음
            if reason_crits and kind not in blamed:
                continue  # "너무 비싸요" 로 거절했으면 지목/특징은 탓하지 않음
            if kind == "feature" and reason_crits and not feature_hit(value, reason):
                continue  # "채광이 별로" 라면 평지·IC 같은 다른 특징은 탓하지 않음
            self.attrs.setdefault(kind, {}).setdefault(value, [0.0, 0.0])[1] += 1.0 if kind in blamed else 0.5
        self.n_feedback += 1
        return describe_change(before, self.multipliers)

    # ------------------------------------------------------------ 표시/저장
    def describe(self, prior: Optional["PreferenceModel"] = None) -> str:
        w = self.weights(prior)
        parts = [f"{CRITERIA_KO[k]} {w[k]:.0f}%" for k in sorted(w, key=lambda k: -w[k])]
        lines = [f"학습한 피드백 {self.n_feedback}건 | 중요도: " + ", ".join(parts)]
        likes = [v for kind in self.attrs.values() for v, (l, d) in kind.items() if l - d >= 1]
        dislikes = [v for kind in self.attrs.values() for v, (l, d) in kind.items() if d - l >= 1]
        if likes:
            lines.append("  선호: " + ", ".join(likes))
        if dislikes:
            lines.append("  비선호: " + ", ".join(dislikes))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"multipliers": self.multipliers, "attrs": self.attrs, "n_feedback": self.n_feedback}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreferenceModel":
        m = cls()
        m.multipliers.update({k: float(v) for k, v in d.get("multipliers", {}).items() if k in BASE_WEIGHTS})
        for kind, values in d.get("attrs", {}).items():
            m.attrs[kind] = {k: [float(v[0]), float(v[1])] for k, v in values.items()}
        m.n_feedback = int(d.get("n_feedback", 0))
        return m


def _related_to_any(feature: str, keywords: Iterable[str]) -> bool:
    words = set(FEATURE_SYNONYMS.get(feature, [feature]))
    return any(words & set(FEATURE_SYNONYMS.get(k, [k])) for k in keywords)


def reason_criteria(reason: str) -> list[str]:
    if not reason:
        return []
    return [crit for pattern, crit in REASON_RULES if re.search(pattern, reason)]


def describe_change(before: dict[str, float], after: dict[str, float]) -> list[str]:
    out = []
    for k in BASE_WEIGHTS:
        b, a = before.get(k, 1.0), after.get(k, 1.0)
        if a / b >= 1.12:
            out.append(f"{CRITERIA_KO[k]} 중요도 ↑")
        elif a / b <= 0.89:
            out.append(f"{CRITERIA_KO[k]} 중요도 ↓")
    return out
