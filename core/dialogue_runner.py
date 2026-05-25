"""多轮对话执行器——控制用户模拟器与被测模型的对话流程"""

from dataclasses import dataclass, field
from typing import Optional

import re

from openai import OpenAI

from config import settings
from core.user_simulator import UserSimulator
from core.scenario_builder import Scenario, TaskRubric


@dataclass
class Turn:
    """单轮对话"""
    user: str
    assistant: str
    turn_number: int


@dataclass
class DialogResult:
    """一次对话的运行结果"""
    scenario: Scenario
    turns: list[Turn] = field(default_factory=list)
    finished_reason: str = "max_turns"


# 被测模型的 System Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你正在打电话给用户，需要完成以下任务：

{task_instruction}

## 通话要点
- 像真人电话客服一样说话，用口语，带过渡词
- 每轮先说一句话回应用户，再说你的推进内容
- 用户热情就推进，犹豫就解释，拒绝就温和挽回一次然后尊重
- 严格禁止：违反任务中的禁止事项
- 任务完成就礼貌结束，别拖沓

{opening_line_section}

{call_flow_section}

{faq_section}

{constraints_section}"""


class DialogueRunner:
    """多轮对话执行器"""

    def __init__(self):
        self._client = None
        self.user_simulator = UserSimulator()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs = {"api_key": settings.openai_api_key}
            if settings.llm_provider == "dashscope":
                kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                kwargs["base_url"] = settings.openai_api_base
            self._client = OpenAI(timeout=300.0, **kwargs)
        return self._client

    def run_dialog(
        self,
        scenario: Scenario,
        rubric: TaskRubric,
        max_turns: Optional[int] = None,
    ) -> DialogResult:
        """运行一次完整的用户模拟器↔被测模型对话

        Args:
            scenario: 用户画像场景
            rubric: 任务评测要素
            max_turns: 最大对话轮次

        Returns:
            DialogResult 包含完整对话记录
        """
        max_turns = max_turns or rubric.constraints.get("max_turns") or settings.max_turns
        if isinstance(max_turns, str):
            max_turns = int(max_turns)
        result = DialogResult(scenario=scenario)

        # 对话历史
        dialog_history: list[dict] = []

        for turn_num in range(1, max_turns + 1):
            # ── 用户模拟器生成用户消息 ──
            user_msg = self.user_simulator.generate_response(
                persona_prompt=scenario.persona_prompt,
                test_goal=scenario.test_goal,
                dialog_history=dialog_history,
                rubric=rubric,
            )

            # 检查结束信号（至少2轮后才允许结束）
            if turn_num >= 3 and re.search(r'<END>', user_msg, re.IGNORECASE):
                result.finished_reason = "end_signal"
                break

            # ── 被测模型生成回复 ──
            assistant_msg = self._call_target_model(
                rubric, dialog_history, user_msg
            )

            # 记录本轮
            turn = Turn(
                user=user_msg,
                assistant=assistant_msg,
                turn_number=turn_num,
            )
            result.turns.append(turn)

            dialog_history.append({"role": "user", "content": user_msg})
            dialog_history.append({"role": "assistant", "content": assistant_msg})

            # 检查是否自然结束
            if self._is_natural_end(assistant_msg, dialog_history):
                result.finished_reason = "natural_end"
                break

        return result

    def _build_instruction(self, rubric: TaskRubric) -> str:
        """从 TaskRubric 构建任务指令文本"""
        parts = []
        if rubric.role:
            parts.append(f"你的角色：{rubric.role}")
        parts.append(f"核心目标：{rubric.task_goal}")
        if rubric.must_do:
            parts.append(f"必须完成：{'；'.join(rubric.must_do)}")
        if rubric.must_not_do:
            parts.append(f"禁止行为：{'；'.join(rubric.must_not_do)}")
        if rubric.constraints.get("tone"):
            parts.append(f"语气要求：{rubric.constraints['tone']}")
        return "\n".join(parts)

    def _build_system_prompt(self, rubric: TaskRubric) -> str:
        """构建完整的 System Prompt，包含结构化指令的各部分"""
        instruction = self._build_instruction(rubric)

        # 开场白 section
        opening_section = ""
        if rubric.opening_line:
            opening_section = (
                "## 开场白要求\n"
                f"首轮对话必须使用以下开场白模板（${...}为变量，需替换为实际值）：\n"
                f"> {rubric.opening_line}"
            )

        # 流程 section
        flow_section = ""
        if rubric.call_flow:
            lines = ["## 通话流程（必须按顺序执行）"]
            for step in rubric.call_flow:
                lines.append(f"\n### 步骤{step.step_id}: {step.title}")
                if step.description:
                    lines.append(f"{step.description}")
                if step.reference_script:
                    lines.append(f"参考话术: {step.reference_script}")
                for sub in step.sub_steps:
                    lines.append(f"- {sub.get('title', '')}: {sub.get('detail', '')}")
            flow_section = "\n".join(lines)

        # FAQ section
        faq_section = ""
        if rubric.knowledge_points:
            lines = ["## 知识库（FAQ）—— 回答用户问题必须严格参考以下内容"]
            for question, answer in rubric.knowledge_points.items():
                lines.append(f"- **{question}**: {answer}")
            faq_section = "\n".join(lines)

        # 约束 section
        constraints_section = ""
        constraint_lines = []
        if rubric.must_not_do:
            constraint_lines.append("禁止行为：")
            for item in rubric.must_not_do:
                constraint_lines.append(f"- {item}")
        forbidden = rubric.constraints.get("forbidden_phrases", [])
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        if forbidden:
            constraint_lines.append(f"禁止词汇/短语：{'、'.join(forbidden)}")
        max_words = rubric.constraints.get("max_words_per_turn", 0)
        if max_words:
            constraint_lines.append(f"每次回复控制在约 {max_words} 字以内")
        if constraint_lines:
            constraints_section = "## 严格约束\n" + "\n".join(constraint_lines)

        return SYSTEM_PROMPT_TEMPLATE.format(
            task_instruction=instruction,
            opening_line_section=opening_section,
            call_flow_section=flow_section,
            faq_section=faq_section,
            constraints_section=constraints_section,
        )

    def _call_target_model(
        self, rubric: TaskRubric, history: list[dict], user_msg: str
    ) -> str:
        """调用被测模型生成回复"""
        system = self._build_system_prompt(rubric)
        messages = [{"role": "system", "content": system}]

        # 添加上文（最近 10 轮）
        for h in history[-10:]:
            messages.append(
                {"role": h["role"], "content": h["content"]}
            )

        messages.append({"role": "user", "content": user_msg})

        response = self._get_client().chat.completions.create(
            model=settings.target_model_name,
            messages=messages,
            temperature=settings.target_temperature,
            max_tokens=250,
        )

        return response.choices[0].message.content.strip()

    def _is_natural_end(self, assistant_msg: str, history: list[dict]) -> bool:
        """判断对话是否自然结束"""
        end_phrases = ["再见", "感谢", "谢谢", "祝您", "先这样"]
        has_end_phrase = any(p in assistant_msg for p in end_phrases)
        # 至少有 1 轮完整对话（user + assistant）后再检查结束
        if has_end_phrase and len(history) >= 2:
            return True
        return False
