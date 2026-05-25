"""全局配置模块——LLM API 配置、路径配置、评测参数"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).parent


class Settings(BaseSettings):
    # ── LLM Provider ──
    llm_provider: Literal["openai", "dashscope"] = "openai"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.deepseek.com"
    openai_model_name: str = "deepseek-v4-flash"

    # ── Model Separation（防止循环自评）──
    target_model_name: str = "deepseek-v4-flash"  # 被测模型（System Under Test）
    target_temperature: float = 0.3  # 被测模型温度（稍高以模拟真实场景）

    # ── Evaluation ──
    max_turns: int = 8
    eval_temperature: float = 0.3  # 多评委评测温度（>0使σ有意义）
    sim_temperature: float = 0.9
    num_evals_per_scenario: int = 3  # 多评委数量

    # ── Paths ──
    project_root: Path = _PROJECT_ROOT
    prompt_dir: Path = _PROJECT_ROOT / "prompts"
    data_dir: Path = _PROJECT_ROOT / "data"
    output_dir: Path = _PROJECT_ROOT / "outputs"

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
