"""全局配置模块——LLM API 配置、路径配置、评测参数"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # ── LLM Provider ──
    llm_provider: Literal["openai", "dashscope"] = "openai"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model_name: str = "gpt-4o"
    dashscope_api_key: str = ""
    dashscope_model_name: str = "qwen-max"

    # ── Evaluation ──
    max_turns: int = 8
    eval_temperature: float = 0.1
    sim_temperature: float = 0.7
    num_evals_per_scenario: int = 1  # 多次采样取平均

    # ── Paths ──
    project_root: Path = Path(__file__).parent
    prompt_dir: Path = Path(__file__).parent / "prompts"
    data_dir: Path = Path(__file__).parent / "data"
    output_dir: Path = Path(__file__).parent / "outputs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
