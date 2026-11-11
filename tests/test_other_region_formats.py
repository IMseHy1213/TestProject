"""'음성군에 창고 지을 땅 있어?' → '다른 지역에는 없어?' 는 음성군을 빼고 찾아야 함.
실제 매물 장부처럼 주소가 '충북 음성 대소면 …', '충청북도음성군금왕읍 …' 처럼 적혀 있어도."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.config import Settings
from real_estate_agent.tool_agent import AgentContext, ToolAgent
from real_estate_agent.tool_agent.multi import MultiAgent


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fmt")
    data = tmp / "매물"
    head = ["매물번호", "소재지", "지목", "용도지역", "면적(평)", "매매가", "도로", "특이사항"]
    rows = {"음성군": [], "진천군": [], "횡성군": []}
    formats = ["충북 음성 대소면 오류리 {}", "충청북도음성군금왕읍 무극리 {}", "음성 감곡면 오향리 {}", "충북음성군 맹동면 {}"]
    for i in range(12):  # 음성군이 대부분 (실제 사무실처럼)
        rows["음성군"].append([f"EU-{100 + i}", formats[i % 4].format(10 + i), "잡종지", "계획관리지역", 300 + 20 * i,
                              f"{2 + i % 3}억", "6m 도로 접함", "창고 부지, 대형차 진입"])
    for i in range(3):
        rows["진천군"].append([f"JC-{200 + i}", f"충청북도 진천군 덕산읍 용몽리 {i + 1}", "잡종지", "계획관리지역", 400,
                              "2억 5천", "6m 도로 접함", "창고 부지"])
    rows["횡성군"].append(["HS-300", "강원특별자치도 횡성군 우천면 1", "잡종지", "계획관리지역", 500, "2억", "6m 도로 접함",
                         "창고 부지"])
    for region, items in rows.items():
        (data / region).mkdir(parents=True)
        wb = Workbook()
        ws = wb.active
        ws.append(head)
        for r in items:
            ws.append(r)
        wb.save(data / region / "장부.xlsx")
    a = RealEstateAgent(Settings(data_dir=data, db_path=tmp / "f.db", log_dir=tmp / "l", llm_providers=["rule"]))
    a.sync()
    return a


def region_of(agent, lid):
    addr = agent.find_listing(lid).address
    return "음성군" if "음성" in addr else ("진천군" if "진천" in addr else "횡성군")


@pytest.mark.parametrize("runner_cls", [ToolAgent, MultiAgent])
def test_other_region_excludes_eumseong(agent, runner_cls):
    t = runner_cls(agent, ctx=AgentContext(), log_dir="")
    r1, h = t.run("음성군에 창고 지을 땅 있어?")
    assert r1.listing_ids and {region_of(agent, i) for i in r1.listing_ids} == {"음성군"}
    r2, h = t.run("다른 지역에는 없어?", h)
    regions = {region_of(agent, i) for i in r2.listing_ids}
    assert r2.listing_ids and "음성군" not in regions and regions <= {"진천군", "횡성군"}, (r2.listing_ids, r2.answer)
    assert "음성군 말고" in r2.answer or "음성군 제외" in r2.answer
    r3, _ = t.run("정말 다른 지역에는 없어?", h)  # 다시 확인해도 음성군은 계속 빠짐
    assert "음성군" not in {region_of(agent, i) for i in r3.listing_ids}


def test_region_reader_understands_address_formats(agent):
    from real_estate_agent.agent import mask_address
    from real_estate_agent.tool_agent.tools import shown_region

    eumseong = [l for l in agent.listings() if l.listing_id.startswith("EU-")]
    assert {shown_region(l) for l in eumseong} == {"음성군"}
    assert {mask_address(l.address) for l in eumseong} == {"충청북도 음성군 ***"}
