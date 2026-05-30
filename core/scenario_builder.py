"""任务指令解析模块——将自然语言指令解析为结构化评测要素"""

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import settings
from core.llm_utils import safe_llm_call, get_llm_client


class CallFlowStep:
    """单个对话流程步骤"""

    def __init__(
        self,
        step_id: str,
        title: str,
        description: str = "",
        sub_steps: list[dict] = None,
        reference_script: str = "",
    ):
        self.step_id = step_id
        self.title = title
        self.description = description
        self.sub_steps = sub_steps or []
        self.reference_script = reference_script

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "sub_steps": self.sub_steps,
            "reference_script": self.reference_script,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CallFlowStep":
        return cls(
            step_id=data.get("step_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            sub_steps=data.get("sub_steps", []),
            reference_script=data.get("reference_script", ""),
        )


class TaskRubric:
    """结构化的评测要素"""

    def __init__(
        self,
        task_goal: str,
        must_do: list[str] = None,
        must_not_do: list[str] = None,
        constraints: dict = None,
        success_criteria: list[str] = None,
        opening_line: str = "",
        call_flow: list[CallFlowStep] = None,
        knowledge_points: dict[str, str] = None,
        role: str = "",
        raw_instruction: str = "",
    ):
        self.task_goal = task_goal
        self.must_do = must_do or []
        self.must_not_do = must_not_do or []
        # 终极兜底：构造函数中清洗数值字段
        self.constraints = {}
        for ck, cv in (constraints or {}).items():
            if ck in ("max_words_per_turn", "max_turns") and isinstance(cv, str):
                try:
                    self.constraints[ck] = int(cv)
                except (ValueError, TypeError):
                    self.constraints[ck] = 0
            else:
                self.constraints[ck] = cv
        self.success_criteria = success_criteria or []
        # ── 结构化指令新增字段 ──
        self.opening_line = opening_line          # 开场白模板
        self.call_flow = call_flow or []           # 对话流程步骤列表
        self.knowledge_points = knowledge_points or {}  # FAQ 知识库 {"问题关键词": "标准答案"}
        self.role = role                            # AI 扮演的角色
        self.raw_instruction = raw_instruction      # 原始指令文本（兼容旧模式）

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRubric":
        call_flow = []
        for step_data in data.get("call_flow", []):
            call_flow.append(CallFlowStep.from_dict(step_data))

        # 安全清理约束中的数值字段（LLM 可能返回文字描述）
        constraints = {}
        for k, v in (data.get("constraints") or {}).items():
            if k in ("max_words_per_turn", "max_turns") and isinstance(v, str):
                try:
                    constraints[k] = int(v)
                except (ValueError, TypeError):
                    constraints[k] = 0
            else:
                constraints[k] = v

        return cls(
            task_goal=data.get("task_goal", ""),
            must_do=data.get("must_do", []),
            must_not_do=data.get("must_not_do", []),
            constraints=constraints,
            success_criteria=data.get("success_criteria", []),
            opening_line=data.get("opening_line", ""),
            call_flow=call_flow,
            knowledge_points=data.get("knowledge_points", {}),
            role=data.get("role", ""),
            raw_instruction=data.get("raw_instruction", ""),
        )

    def to_dict(self) -> dict:
        return {
            "task_goal": self.task_goal,
            "must_do": self.must_do,
            "must_not_do": self.must_not_do,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "opening_line": self.opening_line,
            "call_flow": [s.to_dict() for s in self.call_flow],
            "knowledge_points": self.knowledge_points,
            "role": self.role,
            "raw_instruction": self.raw_instruction,
        }

    @property
    def has_structured_instruction(self) -> bool:
        """是否包含结构化指令（Call Flow / FAQ / Opening Line）"""
        return bool(self.call_flow or self.knowledge_points or self.opening_line)

    @property
    def call_flow_summary(self) -> str:
        """生成流程摘要文本"""
        if not self.call_flow:
            return ""
        lines = []
        for step in self.call_flow:
            lines.append(f"步骤{step.step_id}: {step.title}")
            if step.reference_script:
                lines.append(f"  参考话术: {step.reference_script[:80]}...")
            for sub in step.sub_steps:
                lines.append(f"  - {sub.get('title', '')}: {sub.get('detail', '')[:60]}")
        return "\n".join(lines)

    def __str__(self) -> str:
        parts = [f"任务目标: {self.task_goal}"]
        if self.role:
            parts.append(f"角色: {self.role}")
        if self.must_do:
            parts.append(f"必须做: {', '.join(self.must_do)}")
        if self.must_not_do:
            parts.append(f"禁止做: {', '.join(self.must_not_do)}")
        if self.opening_line:
            parts.append(f"开场白: {self.opening_line[:50]}...")
        if self.call_flow:
            parts.append(f"流程步骤: {len(self.call_flow)}步")
        if self.knowledge_points:
            parts.append(f"FAQ条目: {len(self.knowledge_points)}条")
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
        pass

    def _get_client(self) -> OpenAI:
        return get_llm_client()

    def parse_instruction(self, instruction: str) -> TaskRubric:
        """将自然语言任务指令解析为结构化 TaskRubric"""
        prompt_path = settings.prompt_dir / "rubric_extractor.txt"
        template_text = prompt_path.read_text(encoding="utf-8")

        prompt = template_text.replace("{task_instruction}", instruction)

        client = self._get_client()

        def _call():
            resp = client.chat.completions.create(
                model=settings.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.eval_temperature,
            )
            return resp.choices[0].message.content.strip()

        content = safe_llm_call(_call, label="指令解析", fallback=json.dumps({
            "task_goal": "解析临时失败，请重试",
            "must_do": [], "must_not_do": [], "constraints": {}, "success_criteria": [],
        }))

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

    def parse_structured_instruction(self, data: dict) -> TaskRubric:
        """解析结构化指令（JSON格式，支持 Role/Task/OpeningLine/CallFlow/FAQ/Constraints）

        输入格式示例:
        {
            "role": "美团外卖骑手的站长",
            "task": "致电骑手通知合同签署并提醒配送",
            "opening_line": "你好，请问是${rider_name}吗？我是站长...",
            "call_flow": [
                {"step_id": "1", "title": "告知合同生效", "reference_script": "...",
                 "sub_steps": [{"title": "...", "detail": "..."}]}
            ],
            "knowledge_points": {"飞毛腿是什么": "飞毛腿是...", "如何取消": "..."},
            "constraints": {"max_words_per_turn": 30, "tone": "口语化", "forbidden_phrases": ["好的"]}
        }
        """
        # ── 提取任务目标 ──
        task_goal = data.get("task", "") or data.get("task_goal", "")
        role = data.get("role", "")

        # ── 提取开场白 ──
        opening_line = data.get("opening_line", "")

        # ── 提取流程步骤 ──
        call_flow = []
        for step_data in data.get("call_flow", []):
            call_flow.append(CallFlowStep.from_dict(step_data))

        # ── 提取 FAQ 知识库 ──
        knowledge_points = data.get("knowledge_points", {}) or data.get("faq", {})

        # ── 提取约束 ──
        constraints = data.get("constraints", {})

        # ── 从约束和流程中推导 must_do / must_not_do ──
        must_do = []
        must_not_do = []

        # 流程步骤作为必须完成项
        for step in call_flow:
            must_do.append(f"完成步骤{step.step_id}: {step.title}")

        # 约束中的禁止短语
        forbidden = constraints.get("forbidden_phrases", [])
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        for phrase in forbidden:
            must_not_do.append(f"禁止说'{phrase}'")

        # 约束中的语气要求
        tone = constraints.get("tone", "")
        if tone:
            must_do.append(f"语气要求: {tone}")

        # 字数限制（安全转换，支持"无限制"等文字描述）
        max_words = constraints.get("max_words_per_turn", 0)
        if max_words:
            try:
                constraints["max_words_per_turn"] = int(max_words)
            except (ValueError, TypeError):
                constraints["max_words_per_turn"] = 0

        # ── 成功标准 ──
        success_criteria = []
        if call_flow:
            success_criteria.append(f"完成全部 {len(call_flow)} 个流程步骤")
        if opening_line:
            success_criteria.append("使用指定开场白开始对话")
        if knowledge_points:
            success_criteria.append("问题回答与知识库一致")

        return TaskRubric(
            task_goal=task_goal,
            role=role,
            opening_line=opening_line,
            call_flow=call_flow,
            knowledge_points=knowledge_points,
            constraints=constraints,
            must_do=must_do,
            must_not_do=must_not_do,
            success_criteria=success_criteria,
        )

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
