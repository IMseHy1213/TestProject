"""시작 속도: 엑셀은 빠르게 읽고 바뀐 파일만 다시 읽음, AI 패키지는 키가 있을 때만 필요할 때 불러옴."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook

from real_estate_agent.agent import RealEstateAgent
from real_estate_agent.collectors.excel_cache import SheetCache, read_sheets_uncached
from real_estate_agent.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def _ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.append(["매물번호", "소재지", "지목", "면적(평)", "매매가", "등록일"])
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_cache_rereads_only_changed_files(tmp_path):
    data = tmp_path / "매물"
    _ledger(data / "토지" / "음성군" / "a.xlsx", [["A-1", "충청북도 음성군 대소면 오류리 1", "대", 300, "3억", datetime(2026, 1, 2)]])
    _ledger(data / "토지" / "진천군" / "b.xlsx", [["B-1", "충청북도 진천군 덕산읍 용몽리 2", "전", 500, "2억", datetime(2026, 2, 3)]])
    s = Settings(data_dir=data, db_path=tmp_path / "x.db", log_dir=tmp_path / "l", llm_providers=["rule"])
    a = RealEstateAgent(s)
    a.sync()
    assert a.collector.cache.misses == 2 and (tmp_path / "x_filecache.json").exists()

    b = RealEstateAgent(s)  # 프로그램을 다시 켬
    b.sync()
    assert (b.collector.cache.misses, b.collector.cache.hits) == (0, 2)
    assert {l.office_no for l in b.listings()} == {"A-1", "B-1"}
    assert next(l for l in b.listings() if l.office_no == "A-1").registered_date.isoformat() == "2026-01-02"

    _ledger(data / "토지" / "음성군" / "a.xlsx", [["A-1", "충청북도 음성군 대소면 오류리 1", "대", 300, "2억 8천", None]])
    (data / "토지" / "진천군" / "b.xlsx").unlink()
    c = RealEstateAgent(s)
    c.sync()
    assert c.collector.cache.misses == 1  # 바뀐 파일만 다시 읽음
    assert [(l.office_no, l.price_manwon) for l in c.listings()] == [("A-1", 28000)]
    saved = json.loads((tmp_path / "x_filecache.json").read_text(encoding="utf-8"))
    assert len(saved["files"]) == 1  # 지운 파일의 기억은 버림


def test_broken_cache_file_is_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    assert SheetCache(p)._data == {}


def test_fast_reader_keeps_layout(tmp_path):
    path = tmp_path / "s.xlsx"
    wb = Workbook()
    ws = wb.active
    ws["B2"], ws["D4"], ws["C5"] = "가격", 123, datetime(2020, 1, 1)
    wb.save(path)
    (name, grid), = read_sheets_uncached(path)
    assert grid[1][1] == "가격" and grid[3][3] == 123 and grid[4][2] == datetime(2020, 1, 1)
    assert grid[0] == [None, None, None, None] and len(grid) == 5  # 빈 첫 줄·첫 칸도 그대로 (예전 읽기와 같은 모양)


def test_ai_packages_load_only_when_needed():
    """키가 없으면 openai·anthropic 을 불러오지 않음, 키가 있어도 처음 쓸 때까지 미룸 (시작이 몇 초 빨라짐)."""
    code = ("import os, sys; os.environ.pop('OPENAI_API_KEY', None); os.environ.pop('ANTHROPIC_API_KEY', None);"
            "from real_estate_agent.llm.router import LLMRouter; r = LLMRouter(['openai','anthropic','rule']);"
            "print(int('openai' in sys.modules), int('anthropic' in sys.modules), len(r.clients));"
            "os.environ['OPENAI_API_KEY']='sk-test'; r = LLMRouter(['openai','rule']);"
            "print(int('openai' in sys.modules), len(r.clients)); r.clients[0].client; print(int('openai' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    lines = out.stdout.split()
    assert lines[:3] == ["0", "0", "0"], out.stderr
    assert lines[3:5] == ["0", "1"]  # 키가 있어도 아직 안 불러옴
    assert lines[5] in {"1", "0"}  # openai 가 설치돼 있으면 처음 쓸 때 불러옴
