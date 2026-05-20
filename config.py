"""全局配置模块——LLM API 配置、路径配置、评测参数"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).parent


class Settings(BaseSettings):
    # ── LLM Provider ──
    llm_provider: Literal["openai"] = "openai"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.deepseek.com"
    openai_model_name: str = "deepseek-v4-flash"

    # ── Evaluation ──
    max_turns: int = 8
    eval_temperature: float = 0.1
    sim_temperature: float = 0.8
    num_evals_per_scenario: int = 1  # 多次采样取平均

    # ── Paths ──
    project_root: Path = _PROJECT_ROOT
    prompt_dir: Path = _PROJECT_ROOT / "prompts"
    data_dir: Path = _PROJECT_ROOT / "data"
    output_dir: Path = _PROJECT_ROOT / "outputs"

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
