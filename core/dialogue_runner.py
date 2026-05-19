"""多轮对话执行器——控制用户模拟器与被测模型的对话流程"""

from dataclasses import dataclass, field
from typing import Optional

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
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的数字人外呼助手。请严格遵循以下任务指令：

{task_instruction}

要求：
1. 严格遵守任务指令，完成核心目标任务。
2. 不得违反禁止事项。
3. 保持礼貌、自然、简洁的对话风格。
4. 根据用户态度灵活推进任务。
5. 如果任务已经完成或用户明确拒绝继续，可以礼貌结束对话。"""


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
            self._client = OpenAI(**kwargs)
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
        max_turns = max_turns or rubric.constraints.get("max_turns", settings.max_turns)
        result = DialogResult(scenario=scenario)

        # 构造被测模型使用的任务指令文本
        task_instruction = self._build_instruction(rubric)

        # 对话历史
        dialog_history: list[dict] = []

        for turn_num in range(1, max_turns + 1):
            # ── 用户模拟器生成用户消息 ──
            user_msg = self.user_simulator.generate_response(
                persona_prompt=scenario.persona_prompt,
                test_goal=scenario.test_goal,
                dialog_history=dialog_history,
            )

            # 检查结束信号
            if user_msg.strip() == "<END>":
                result.finished_reason = "end_signal"
                break

            # ── 被测模型生成回复 ──
            assistant_msg = self._call_target_model(
                task_instruction, dialog_history, user_msg
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
        parts = [f"核心目标：{rubric.task_goal}"]
        if rubric.must_do:
            parts.append(f"必须完成：{'；'.join(rubric.must_do)}")
        if rubric.must_not_do:
            parts.append(f"禁止行为：{'；'.join(rubric.must_not_do)}")
        if rubric.constraints.get("tone"):
            parts.append(f"语气要求：{rubric.constraints['tone']}")
        return "\n".join(parts)

    def _call_target_model(
        self, instruction: str, history: list[dict], user_msg: str
    ) -> str:
        """调用被测模型生成回复"""
        system = SYSTEM_PROMPT_TEMPLATE.format(task_instruction=instruction)
        messages = [{"role": "system", "content": system}]

        # 添加上文（最近 10 轮）
        for h in history[-10:]:
            messages.append(
                {"role": h["role"], "content": h["content"]}
            )

        messages.append({"role": "user", "content": user_msg})

        response = self._get_client().chat.completions.create(
            model=settings.openai_model_name,
            messages=messages,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def _is_natural_end(self, assistant_msg: str, history: list[dict]) -> bool:
        """判断对话是否自然结束"""
        end_phrases = ["再见", "感谢", "谢谢", "祝您", "先这样"]
        has_end_phrase = any(p in assistant_msg for p in end_phrases)
        if has_end_phrase and len(history) >= 2:
            return True
        return False
