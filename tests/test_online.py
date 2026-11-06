"""v1.2 Online: 손님 반응으로 바로 배우는 추천 순서(온라인 학습), 답변 평가, '자동' 상담 방식(톰슨 샘플링)."""

import json
import re
import shutil
import sqlite3
from pathlib import Path
import random

import pytest

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.observability import TRACER
from real_estate_agent.online.bandit import ThompsonBandit
from real_estate_agent.online.learning import pref_key
from real_estate_agent.online.simulate import simulate_bandit, simulate_learning
from real_estate_agent.preference import PreferenceModel, feature_stats, numeric_features
from real_estate_agent.tool_agent import AgentContext, ToolRegistry
from real_estate_agent.tool_agent.service import AgentService, Busy

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"


@pytest.fixture
def agent(tmp_path):
    a = RealEstateAgent(Settings(data_dir=SAMPLE, db_path=tmp_path / "on.db", log_dir=tmp_path / "logs",
                                 llm_providers=["rule"]))
    a.sync()
    return a


# ---------------------------------------------------------------- 온라인 로지스틱 회귀
def test_numeric_taste_learns_from_single_reactions(agent):
    stats = feature_stats(agent.listings())
    big = max(agent.listings(), key=lambda l: l.area_pyeong)
    small = min(agent.listings(), key=lambda l: l.area_pyeong)
    m = PreferenceModel()
    assert m.numeric_bonus(numeric_features(big, stats)) == 0  # 배우기 전에는 영향 없음
    m.learn_numeric(numeric_features(big, stats), liked=True)
    m.learn_numeric(numeric_features(small, stats), liked=False)
    assert m.numeric["area"] > 0.3
    assert m.numeric_bonus(numeric_features(big, stats)) > m.numeric_bonus(numeric_features(small, stats))
    assert PreferenceModel.from_dict(json.loads(json.dumps(m.to_dict()))).numeric == m.numeric


def test_feedback_reorders_next_search(agent):
    ctx = AgentContext(pref_key=pref_key("web:visitor-1"))
    reg = ToolRegistry(agent, ctx)
    first = reg.run("search_listings", {"query": "양평 땅", "limit": 5})["result"]
    assert "학습_반영" not in first
    big = max(first["매물"], key=lambda x: x["면적_평"])
    small = min(first["매물"], key=lambda x: x["면적_평"])
    assert reg.run("record_feedback", {"listing_id": big["id"], "action": "like"})["ok"]
    r = reg.run("record_feedback", {"listing_id": small["id"], "action": "reject", "reason": "너무 좁아요"})
    assert r["result"]["배운_점"]
    second = reg.run("search_listings", {"query": "양평 땅", "limit": 5})["result"]
    avg = lambda res: sum(x["면적_평"] for x in res["매물"][:3]) / 3  # noqa: E731
    assert avg(second) > avg(first) and small["id"] not in [x["id"] for x in second["매물"]]
    assert "반영" in second["학습_반영"]
    names = [r[0] for r in agent.storage.conn.execute("SELECT name FROM preferences")]
    assert ctx.pref_key in names and "__office__" in " ".join(names) or any("office" in n or "공통" in n for n in names)
    assert not any("visitor-1" in n for n in names)  # 방문자 id 는 바꾼 값만


def test_without_pref_key_nothing_is_learned(agent):
    reg = ToolRegistry(agent, AgentContext())
    shown = reg.run("search_listings", {"query": "양평 땅", "limit": 3})["result"]["매물"]
    r = reg.run("record_feedback", {"listing_id": shown[0]["id"], "action": "like"})
    assert r["ok"] and r["result"]["배운_점"] is None


def test_simulated_customers_learn_faster_than_exclusion(agent):
    rows = {r["taste"]: r for r in simulate_learning(agent, rounds=4)}
    wide = rows["넓은 땅을 원함 (400평 이상)"]
    assert wide["learning"][1] > wide["exclude_only"][1]
    for r in rows.values():  # 기본 순서가 틀리던 취향에서는 학습이 이김
        if r["exclude_only"][0] < 1:
            assert sum(r["learning"]) > sum(r["exclude_only"])
    assert sum(sum(r["learning"]) for r in rows.values()) > sum(sum(r["exclude_only"]) for r in rows.values())
    assert not agent.storage.conn.execute("SELECT 1 FROM preferences WHERE name LIKE 'agent:%'").fetchall()


# ---------------------------------------------------------------- 톰슨 샘플링
def test_bandit_choose_update_and_change_vote():
    b = ThompsonBandit(sqlite3.connect(":memory:"), arms=["single", "multi"], rng=random.Random(1))
    arm = b.choose()
    assert {s["arm"]: s["pulls"] for s in b.stats()}[arm] == 1
    b.update("multi", 1)
    b.update("multi", 0, previous=1)  # 👍 → 👎 로 바꿈
    multi = next(s for s in b.stats() if s["arm"] == "multi")
    assert (multi["wins"], multi["losses"]) == (0, 1)
    for _ in range(30):
        b.update("single", 1)
    stats = {s["arm"]: s for s in b.stats()}
    assert stats["single"]["p_best"] > 0.95 and stats["single"]["low"] > stats["multi"]["high"] - 0.5
    assert sum(1 for _ in range(50) if b.choose() == "single") > 45


def test_bandit_beats_fixed_ab():
    r = simulate_bandit({"single": 0.6, "multi": 0.8}, conversations=300, runs=10)
    assert r["best_share_last"] > r["best_share_first"] and r["best_share_last"] > 80
    assert r["thompson_up"] > r["ab_up"] and r["thompson_up"] <= r["oracle_up"] + 15


# ---------------------------------------------------------------- 서비스: 자동 방식·평가·카드 반응
def test_auto_mode_is_sticky_and_ratings_feed_bandit(agent):
    s = AgentService(agent)
    cid, t1 = s.ask("v1", None, "양평 땅 2개 비교해줘", mode="auto")
    arm = t1["variant"].split(":", 1)[1]
    assert t1["variant"].startswith("auto:") and t1["mode"] == arm and t1["id"]
    _, t2 = s.ask("v1", cid, "그 중에 넓은 건?", mode="auto")
    assert t2["variant"] == t1["variant"]  # 한 대화 안에서는 같은 방식
    s.rate("v1", cid, t1["id"], 1)
    s.rate("v1", cid, t2["id"], 0, "원하는 매물이 아니에요")
    stats = {x["arm"]: x for x in s.bandit.stats()}
    assert (stats[arm]["wins"], stats[arm]["losses"]) == (1, 1)
    s.rate("v1", cid, t2["id"], 1)  # 평가 바꾸기
    stats = {x["arm"]: x for x in s.bandit.stats()}
    assert (stats[arm]["wins"], stats[arm]["losses"]) == (2, 0)
    summary = s.ratings.summary()
    assert summary["n"] == 2 and summary["rate"] == 100.0
    assert s.store.load(cid)["turns"][1]["rating"] == 1
    with pytest.raises(ValueError):
        s.rate("someone-else", cid, t1["id"], 1)
    with pytest.raises(ValueError):
        s.rate("v1", cid, "nope", 1)


def test_rating_joins_the_answer_trace(agent):
    TRACER.ring.clear()
    s = AgentService(agent)
    cid, t = s.ask("v1", None, "양평 땅")
    s.rate("v1", cid, t["id"], 0, "답이 틀렸어요")
    fb = [x for x in TRACER.trace(t["trace_id"]) if x.name == "feedback.rating"]
    assert fb and fb[0].parent_id == t["span_id"] and fb[0].attrs["reason"] == "답이 틀렸어요"
    assert s.ratings.recent_negative()[0]["trace_id"] == t["trace_id"]


def test_rating_while_running_is_busy(agent):
    s = AgentService(agent)
    cid, t = s.ask("v1", None, "양평 땅")
    conv = s.store.load(cid)
    s.store.save(cid, "v1", conv["history"], conv["ctx"], conv["turns"], "running", "다음 질문")
    with pytest.raises(Busy):
        s.rate("v1", cid, t["id"], 1)


def test_listing_feedback_learns_and_excludes(agent):
    s = AgentService(agent)
    cid, t = s.ask("v1", None, "양평 땅 3개")
    from real_estate_agent.tool_agent.tools import tool_id

    target = agent.find_listing(t["listing_ids"][0])
    r = s.listing_feedback("v1", cid, tool_id(target), "reject", "너무 비싸요")
    assert r["기록"] == "reject" and r["배운_점"]
    _, t2 = s.ask("v1", cid, "양평 땅 3개")
    assert target.listing_id not in t2["listing_ids"]
    assert target.listing_id in s.memory.load("v1")["rejected"]  # 다음 방문에도 기억
    with pytest.raises(ValueError):
        s.listing_feedback("v1", cid, "없는-매물", "like")


# ---------------------------------------------------------------- 웹·API
@pytest.fixture
def web(tmp_path, monkeypatch):
    from real_estate_agent.web.app import create_app

    monkeypatch.setenv("REA_ADMIN_PASSWORD", "office-pass-1")
    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    app = create_app(Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "logs",
                              llm_providers=["rule"]))
    app.config["TESTING"] = True
    return app


def _csrf(c, path="/agent"):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def test_web_rating_card_feedback_and_admin_page(web):
    c = web.test_client()
    html = c.get("/agent").get_data(as_text=True)
    assert "상담 방식이 뭔가요?" in html and 'value="auto"' in html
    c.post("/agent/ask", data={"q": "양평 땅 3개", "csrf": _csrf(c), "mode": "auto"})
    html = c.get("/agent").get_data(as_text=True)
    assert "👍 도움됐어요" in html and "♡ 관심" in html and "(자동 →" in html
    turn = re.search(r'name="turn" value="([0-9a-f]{12})"', html).group(1)
    pid = re.search(r'name="id" value="([^"]+)"', html).group(1)
    r = c.post("/agent/rate", data={"turn": turn, "rating": "0", "reason": "너무 느려요", "csrf": _csrf(c)})
    assert r.status_code == 302 and r.headers["Location"].endswith(f"#turn-{turn}") or "#last" in r.headers["Location"]
    c.post("/agent/listing-feedback", data={"id": pid, "action": "reject", "reason": "너무 좁아요", "csrf": _csrf(c)})
    html = c.get("/agent").get_data(as_text=True)
    assert "다음 검색부터" in html and "✕ 다음부터 제외" in html and "알려 주셔서 감사합니다" in html
    assert "관리자 로그인" in c.get("/admin/online").get_data(as_text=True)
    c.post("/admin/login", data={"password": "office-pass-1", "csrf": _csrf(c, "/admin")})
    page = c.get("/admin/online").get_data(as_text=True)
    assert "톰슨 샘플링" in page and "너무 느려요" in page and "가장 좋을 확률" in page
    assert "043-877-5242" in page and "학습한 피드백" in page  # 사무실 정보 footer 와 공통 모델이 섞이지 않음


def test_api_feedback_endpoints(agent):
    from fastapi.testclient import TestClient

    from real_estate_agent.api import create_api

    api = TestClient(create_api(agent=agent))
    h = {"X-Visitor-Id": "visitor-online-1"}
    body = api.post("/api/v1/agent/ask", json={"question": "양평 땅 3개", "mode": "auto"}, headers=h).json()
    assert body["turn_id"] and body["mode"] in {"single", "multi"}
    r = api.post("/api/v1/agent/feedback", headers=h, json={"conversation_id": body["conversation_id"],
                                                            "turn_id": body["turn_id"], "rating": "up"})
    assert r.status_code == 200 and r.json()["rating"] == "up"
    r = api.post("/api/v1/agent/listing-feedback", headers=h, json={
        "conversation_id": body["conversation_id"], "listing_id": body["listings"][0]["id"], "action": "like"})
    assert r.status_code == 200 and r.json()["learned"] is not None
    assert api.post("/api/v1/agent/feedback", json={"conversation_id": "x", "turn_id": "y", "rating": "up"}).status_code == 400
    bad = api.post("/api/v1/agent/feedback", headers={"X-Visitor-Id": "visitor-other-9"},
                   json={"conversation_id": body["conversation_id"], "turn_id": body["turn_id"], "rating": "down"})
    assert bad.status_code == 404  # 남의 대화는 평가할 수 없음
