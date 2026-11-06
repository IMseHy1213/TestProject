"""v1.2 온라인 학습 시뮬레이션 (python main.py online-sim) - 아직 실제 손님 평가가 없을 때 알고리즘이 맞게 도는지 확인.

(A) 추천 순서 학습: 취향이 정해진 가상 손님이 매 라운드 상위 3건을 보고 ♡(취향에 맞음) / ✕(+이유) 를 누름.
    '학습 켬'과 '제외만'(✕ 한 매물만 빠지고 배우지는 않음)을 비교해, 학습 자체의 효과만 봄.
    지표: 상위 3건 중 취향에 맞는 비율 (precision@3)
(B) '자동' 상담 방식(톰슨 샘플링): 방식별 만족 확률을 '가정'하고 대화 N번을 흉내 냄.
    반반 나누는 고정 A/B 테스트와 비교해 👍 수와, 더 좋은 방식을 고른 비율이 어떻게 바뀌는지.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from .bandit import ThompsonBandit
from .learning import pref_key


@dataclass
class Taste:
    name: str
    likes: Callable[[Any], bool]
    reason: str


TASTES = [
    Taste("넓은 땅을 원함 (400평 이상)", lambda l: l.area_pyeong >= 400, "너무 좁아요"),
    Taste("싼 땅을 원함 (2억 5천 이하)", lambda l: l.price_manwon <= 25000, "너무 비싸요"),
    Taste("평당가 싼 땅 (평당 80만원 이하)", lambda l: l.price_per_pyeong <= 80, "너무 비싸요"),
]


def _run_taste(agent, taste: Taste, learn: bool, rounds: int, query: str) -> list[float]:
    from ..tool_agent import AgentContext, ToolRegistry

    agent.storage.save_preference(pref_key(f"sim:{taste.name}:{learn}"), _fresh())
    ctx = AgentContext(pref_key=pref_key(f"sim:{taste.name}:{learn}") if learn else "")
    reg = ToolRegistry(agent, ctx)
    curve = []
    for _ in range(rounds):
        r = reg.run("search_listings", {"query": query, "limit": 3})
        shown = r["result"]["매물"] if r["ok"] else []
        listings = [agent.find_listing(reg.resolve(x["id"])) for x in shown]
        hits = [taste.likes(l) for l in listings]
        curve.append(round(sum(hits) / max(1, len(hits)), 3))
        liked = next((l for l, h in zip(listings, hits) if h), None)
        disliked = next((l for l, h in zip(listings, hits) if not h), None)
        if liked:
            reg.run("record_feedback", {"listing_id": liked.listing_id, "action": "like"})
        if disliked:
            reg.run("record_feedback", {"listing_id": disliked.listing_id, "action": "reject", "reason": taste.reason})
    return curve


def _fresh():
    from ..preference import PreferenceModel

    return PreferenceModel()


def simulate_learning(agent, rounds: int = 6, query: str = "땅 추천해줘") -> list[dict]:
    from ..storage import OFFICE_MODEL

    saved_office = agent.storage.load_preference(OFFICE_MODEL)  # 시뮬레이션이 사무실 공통 모델을 바꾸지 않게
    out = []
    try:
        for taste in TASTES:
            agent._office_model = _fresh()
            off = _run_taste(agent, taste, False, rounds, query)
            agent._office_model = _fresh()
            on = _run_taste(agent, taste, True, rounds, query)
            out.append({"taste": taste.name, "learning": on, "exclude_only": off})
    finally:
        agent.storage.save_preference(OFFICE_MODEL, saved_office)
        agent._office_model = None
        sim_keys = [pref_key(f"sim:{t.name}:{b}") for t in TASTES for b in (True, False)]
        agent.storage.conn.executemany("DELETE FROM preferences WHERE name = ?", [(k,) for k in sim_keys])
        agent.storage.conn.commit()
    return out


def simulate_bandit(p_true: dict[str, float], conversations: int = 500, seed: int = 1, runs: int = 20) -> dict:
    """방식별 만족 확률(가정) → 톰슨 샘플링 vs 반반 A/B. runs 번 반복한 평균."""
    arms = list(p_true)
    best = max(arms, key=p_true.get)
    totals = {"thompson": 0.0, "ab": 0.0}
    share_first, share_last, p_best = 0.0, 0.0, 0.0
    first_n, last_n = min(50, conversations), min(100, conversations)
    for run in range(runs):
        rng = random.Random(seed * 1000 + run)
        conn = sqlite3.connect(":memory:")
        b = ThompsonBandit(conn, arms=arms, rng=random.Random(seed * 7 + run))
        picks = []
        for i in range(conversations):
            arm = b.choose()
            reward = 1 if rng.random() < p_true[arm] else 0
            b.update(arm, reward)
            totals["thompson"] += reward
            picks.append(arm)
            ab_arm = arms[i % len(arms)]
            totals["ab"] += 1 if rng.random() < p_true[ab_arm] else 0
        share_first += sum(1 for a in picks[:first_n] if a == best) / first_n
        share_last += sum(1 for a in picks[-last_n:] if a == best) / last_n
        p_best += next(s["p_best"] for s in b.stats(1000) if s["arm"] == best)
    return {"p_true": p_true, "best": best, "conversations": conversations, "runs": runs,
            "thompson_up": round(totals["thompson"] / runs, 1), "ab_up": round(totals["ab"] / runs, 1),
            "oracle_up": round(p_true[best] * conversations, 1),
            "best_share_first": round(100 * share_first / runs, 1), "best_share_last": round(100 * share_last / runs, 1),
            "p_best_final": round(100 * p_best / runs, 1), "first_n": first_n, "last_n": last_n}


def markdown(learning: list[dict], bandit: dict) -> str:
    rounds = len(learning[0]["learning"]) if learning else 0
    lines = ["# v1.2 온라인 학습 시뮬레이션", "",
             "## (A) 손님 반응(♡·✕)으로 추천 순서 학습", "",
             f"가상 손님이 매 라운드 상위 3건 중 취향에 맞는 매물 ♡, 안 맞는 매물 ✕(+이유). 값 = 상위 3건 중 취향에 맞는 비율.", "",
             "| 손님 취향 | 방식 | " + " | ".join(f"{i + 1}회" for i in range(rounds)) + " |",
             "|---|---|" + "---|" * rounds]
    for r in learning:
        for key, label in (("exclude_only", "제외만"), ("learning", "**온라인 학습**")):
            lines.append(f"| {r['taste']} | {label} | " + " | ".join(f"{v * 100:.0f}%" for v in r[key]) + " |")
    lines += ["", "- 제외만: ✕ 한 매물이 빠지기만 함 (취향은 배우지 않음). 온라인 학습: 반응마다 중요도·선호 속성을 고쳐 다음 검색에 반영.",
              "- 한계: 처음부터 잘 맞는 취향(위 '싼 땅')은 배울 여지가 적고, ♡ 만 쌓이면 우연히 겹친 특징(예: 넓이)을 배워 "
              "잠깐 흔들릴 수 있음 → ✕ 가 함께 오면 바로잡힘.", "",
              "## (B) '자동' 상담 방식 (톰슨 샘플링) vs 반반 A/B", "",
              "방식별 만족 확률은 **가정**입니다 (실제 값은 운영하면서 손님 평가로 알게 됨): "
              + ", ".join(f"{k} {v * 100:.0f}%" for k, v in bandit["p_true"].items()), "",
              f"| 항목 | 값 (대화 {bandit['conversations']}번 × {bandit['runs']}회 평균) |", "|---|---|",
              f"| 처음 {bandit['first_n']}번 중 더 좋은 방식({bandit['best']})을 고른 비율 | {bandit['best_share_first']}% |",
              f"| 마지막 {bandit['last_n']}번 중 더 좋은 방식을 고른 비율 | **{bandit['best_share_last']}%** |",
              f"| 👍 수: 톰슨 샘플링 / 반반 A/B / (정답을 미리 알 때) | **{bandit['thompson_up']}** / {bandit['ab_up']} / {bandit['oracle_up']} |",
              f"| 끝났을 때 '{bandit['best']}가 가장 좋을 확률' 추정 | {bandit['p_best_final']}% |", "",
              "- 처음에는 두 방식을 골고루 써 보고(탐색), 👍가 쌓이면 더 좋은 쪽을 주로 고름(활용) → 반반 A/B 보다 손님 만족이 많음."]
    return "\n".join(lines)
