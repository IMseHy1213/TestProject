"""API 키를 코드 파일(my_api_key.py)에 넣어 쓰는 기능 테스트 (실제 키·네트워크 불필요)."""

import re
import shutil
from pathlib import Path

import pytest

import real_estate_agent.config as config
from real_estate_agent.setup_tools import save_key_to_code_file

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "토지"
FAKE_KEY = "sk-proj-" + "A1b2C3d4" * 6


@pytest.fixture
def key_file(tmp_path, monkeypatch):
    f = tmp_path / "my_api_key.py"
    monkeypatch.setattr(config, "KEY_FILE", f)
    monkeypatch.setattr(config, "KEY_FILE_EXAMPLE", config.PROJECT_ROOT / "my_api_key.example.py")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    return f


def test_example_file_has_empty_key():
    text = (config.PROJECT_ROOT / "my_api_key.example.py").read_text(encoding="utf-8")
    assert 'OPENAI_API_KEY = ""' in text  # 저장소에는 빈 키만


def test_key_file_is_git_ignored():
    assert "my_api_key.py" in (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_load_keys_from_code_file(key_file):
    key_file.write_text(f'OPENAI_API_KEY = "{FAKE_KEY}"\nOPENAI_MODEL = "gpt-5"\n', encoding="utf-8")
    assert config.read_key_file() == {"OPENAI_API_KEY": FAKE_KEY, "OPENAI_MODEL": "gpt-5"}
    assert "my_api_key.py" in config.load_keys()
    import os
    assert os.environ["OPENAI_API_KEY"] == FAKE_KEY and os.environ["OPENAI_MODEL"] == "gpt-5"


def test_broken_key_file_does_not_crash(key_file):
    key_file.write_text("OPENAI_API_KEY = sk-without-quotes\nthis is not python (", encoding="utf-8")
    assert config.read_key_file() == {}


def test_first_run_creates_key_file(key_file):
    assert not key_file.exists()
    config.load_keys()
    assert key_file.exists() and 'OPENAI_API_KEY = ""' in key_file.read_text(encoding="utf-8")


def test_save_key_validates(key_file):
    with pytest.raises(ValueError):
        save_key_to_code_file('sk-abc"; import os; os.system("x") #')
    with pytest.raises(ValueError):
        save_key_to_code_file("hello")
    save_key_to_code_file(FAKE_KEY, "gpt-5.1")
    text = key_file.read_text(encoding="utf-8")
    assert f'OPENAI_API_KEY = "{FAKE_KEY}"' in text and 'OPENAI_MODEL = "gpt-5.1"' in text
    assert len(re.findall(r"^OPENAI_API_KEY =", text, re.M)) == 1  # 덮어쓰기 (중복 설정 줄 없음)


def test_admin_key_form_and_reason(tmp_path, key_file, monkeypatch):
    from real_estate_agent.web.app import create_app

    data = tmp_path / "토지"
    shutil.copytree(SAMPLE, data)
    monkeypatch.setenv("REA_ADMIN_PASSWORD", "pw")
    settings = config.Settings(data_dir=data, db_path=tmp_path / "w.db", log_dir=tmp_path / "l",
                               llm_providers=["openai", "rule"])
    app = create_app(settings)
    agent = app.config["AGENT"]
    c = app.test_client()
    html = c.get("/").get_data(as_text=True)
    assert "AI 미연결" in html and "OPENAI_API_KEY" in html  # 이유 표시

    def tok(path="/"):
        return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)

    c.post("/admin/login", data={"password": "pw", "csrf": tok("/admin")})
    monkeypatch.setattr(type(agent), "test_llm_each", lambda self: [(True, "GPT(gpt-test) 연결 성공: 확인")])
    r = c.post("/admin/ai-key", data={"key": FAKE_KEY, "model": "", "csrf": tok("/admin")}, follow_redirects=True)
    assert "연결 성공" in r.get_data(as_text=True)
    assert agent.llm.available and agent.llm.mode() == "ai"  # 재시작 없이 연결됨
    assert FAKE_KEY in key_file.read_text(encoding="utf-8")
    assert "AI(GPT)가 문장을 이해합니다" in c.get("/").get_data(as_text=True)
    r = c.post("/admin/ai-key", data={"key": "wrong", "csrf": tok("/admin")}, follow_redirects=True)
    assert "저장하지 못했습니다" in r.get_data(as_text=True)


def test_auto_reconnect_when_key_added_later(tmp_path, key_file, monkeypatch):
    from real_estate_agent.agent import RealEstateAgent

    settings = config.Settings(data_dir=SAMPLE, db_path=tmp_path / "a.db", log_dir=tmp_path / "l",
                               llm_providers=["openai", "rule"])
    agent = RealEstateAgent(settings)
    assert agent.llm.mode() == "off"
    key_file.write_text(f'OPENAI_API_KEY = "{FAKE_KEY}"\n', encoding="utf-8")  # 서버 켠 뒤에 키를 넣음
    agent._llm_built_at -= 1000  # 몇 분 지난 것으로
    agent.maybe_reconnect_llm()
    assert agent.llm.mode() == "ai"


def test_save_claude_key_too(key_file, monkeypatch):
    import os

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # 테스트가 끝나면 원래대로
    claude = "sk-ant-" + "b" * 40
    with pytest.raises(ValueError):  # Claude 키를 GPT 칸에 넣은 실수
        save_key_to_code_file(claude)
    with pytest.raises(ValueError):
        save_key_to_code_file("", anthropic_key="sk-ant-short")
    save_key_to_code_file(FAKE_KEY)
    save_key_to_code_file("", anthropic_key=claude)  # GPT 키는 그대로 두고 Claude 키만 추가
    text = key_file.read_text(encoding="utf-8")
    assert f'OPENAI_API_KEY = "{FAKE_KEY}"' in text and f'ANTHROPIC_API_KEY = "{claude}"' in text
    assert os.environ["ANTHROPIC_API_KEY"] == claude
