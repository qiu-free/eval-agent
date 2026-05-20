"""任务指令解析模块——将自然语言指令解析为结构化评测要素"""

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import settings


class TaskRubric:
    """结构化的评测要素"""

    def __init__(
        self,
        task_goal: str,
        must_do: list[str],
        must_not_do: list[str],
        constraints: dict,
        success_criteria: list[str],
    ):
        self.task_goal = task_goal
        self.must_do = must_do
        self.must_not_do = must_not_do
        self.constraints = constraints
        self.success_criteria = success_criteria

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRubric":
        return cls(
            task_goal=data.get("task_goal", ""),
            must_do=data.get("must_do", []),
            must_not_do=data.get("must_not_do", []),
            constraints=data.get("constraints", {}),
            success_criteria=data.get("success_criteria", []),
        )

    def to_dict(self) -> dict:
        return {
            "task_goal": self.task_goal,
            "must_do": self.must_do,
            "must_not_do": self.must_not_do,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
        }

    def __str__(self) -> str:
        parts = [f"任务目标: {self.task_goal}"]
        if self.must_do:
            parts.append(f"必须做: {', '.join(self.must_do)}")
        if self.must_not_do:
            parts.append(f"禁止做: {', '.join(self.must_not_do)}")
        return " | ".join(parts)


class Scenario:
    """单个测试场景"""

    def __init__(
        self,
        persona_id: str,
        persona_name: str,
        persona_description: str,
        behavior: list[str],
        test_goal: str,
    ):
        self.persona_id = persona_id
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.behavior = behavior
        self.test_goal = test_goal

    @property
    def persona_prompt(self) -> str:
        """生成用户画像描述文本"""
        lines = [
            f"用户类型：{self.persona_name}",
            f"行为特点：{self.persona_description}",
            "行为模式：",
        ]
        for b in self.behavior:
            lines.append(f"- {b}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "description": self.persona_description,
            "behavior": self.behavior,
            "test_goal": self.test_goal,
        }


class ScenarioBuilder:
    """场景构建器——解析任务指令并构建测试场景"""

    def __init__(self):
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs = {"api_key": settings.openai_api_key}
            if settings.llm_provider == "dashscope":
                kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                kwargs["base_url"] = settings.openai_api_base
            self._client = OpenAI(timeout=60.0, **kwargs)
        return self._client

    def parse_instruction(self, instruction: str) -> TaskRubric:
        """将自然语言任务指令解析为结构化 TaskRubric"""
        prompt_path = settings.prompt_dir / "rubric_extractor.txt"
        template_text = prompt_path.read_text(encoding="utf-8")

        prompt = template_text.replace("{task_instruction}", instruction)

        response = self._get_client().chat.completions.create(
            model=settings.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.eval_temperature,
        )

        content = response.choices[0].message.content.strip()

        # 更健壮的 Markdown code fence 剥离
        content = re.sub(r'^```(?:json)?\s*([\s\S]*?)```\s*$', r'\1', content.strip(), flags=re.DOTALL)

        # JSON 解析带异常保护
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 回退：尝试修复常见问题后重试
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {
                    "task_goal": "解析失败，请重试",
                    "must_do": [],
                    "must_not_do": [],
                    "constraints": {},
                    "success_criteria": [],
                }
        return TaskRubric.from_dict(data)

    def load_scenarios(self, persona_ids: Optional[list[str]] = None) -> list[Scenario]:
        """从 scenarios.json 加载用户画像，可按 ID 过滤"""
        scenarios_path = settings.data_dir / "scenarios.json"
        data = json.loads(scenarios_path.read_text(encoding="utf-8"))

        scenarios = []
        for p in data["personas"]:
            if persona_ids and p["id"] not in persona_ids:
                continue
            scenarios.append(Scenario(
                persona_id=p["id"],
                persona_name=p["name"],
                persona_description=p["description"],
                behavior=p["behavior"],
                test_goal=p["test_goal"],
            ))
        return scenarios
