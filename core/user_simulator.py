"""用户模拟器——根据用户画像生成多轮对话中的用户回复"""

from openai import OpenAI

from config import settings
from core.scenario_builder import TaskRubric
from core.llm_utils import safe_llm_call, get_llm_client


class UserSimulator:
    """用户模拟器：扮演不同类型的用户与模型对话"""

    def __init__(self):
        pass

    def _get_client(self) -> OpenAI:
        return get_llm_client()

    def generate_response(
        self,
        persona_prompt: str,
        test_goal: str,
        dialog_history: list[dict],
        rubric: TaskRubric = None,
    ) -> str:
        """根据用户画像和对话历史生成用户的下一条回复

        Args:
            persona_prompt: 用户画像描述文本
            test_goal: 该场景的测试目标
            dialog_history: 对话历史 [{"role": "user"/"assistant", "content": "..."}, ...]
            rubric: 任务评估要素（可选，用于生成针对性测试问题）

        Returns:
            用户的下一条回复文本。如果对话结束，返回 "<END>"
        """
        prompt_path = settings.prompt_dir / "user_simulator.txt"
        template_text = prompt_path.read_text(encoding="utf-8")

        # 格式化对话历史
        history_lines = []
        for turn in dialog_history:
            role = "用户" if turn["role"] == "user" else "客服"
            history_lines.append(f"{role}: {turn['content']}")
        history_str = "\n".join(history_lines) if history_lines else "（对话尚未开始）"

        prompt = template_text.replace("{persona}", persona_prompt)
        prompt = prompt.replace("{dialog_history}", history_str)

        # ── 结构化指令增强：注入 FAQ 和流程感知 ──
        if rubric and rubric.has_structured_instruction:
            extra_context = self._build_simulator_context(rubric, dialog_history)
            prompt = prompt + "\n\n" + extra_context

        client = self._get_client()

        def _call():
            resp = client.chat.completions.create(
                model=settings.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.sim_temperature,
            )
            return resp.choices[0].message.content.strip()

        content = safe_llm_call(_call, label="用户模拟器", fallback="<END>")
        return content

    def _build_simulator_context(
        self, rubric: TaskRubric, dialog_history: list[dict]
    ) -> str:
        """为模拟器生成结构化的测试上下文

        基于 Call Flow 和 FAQ，引导模拟器提出针对性问题
        """
        parts = []

        # FAQ 感知：引导用户在合适时机提问 FAQ 中的问题
        if rubric.knowledge_points:
            faq_questions = list(rubric.knowledge_points.keys())
            # 检查历史中是否已经问过
            asked_questions = set()
            for turn in dialog_history:
                if turn["role"] == "user":
                    for q in faq_questions:
                        if any(kw in turn["content"] for kw in q.split()):
                            asked_questions.add(q)

            unanswered = [q for q in faq_questions if q not in asked_questions]
            if unanswered:
                parts.append(
                    "## 知识库测试\n"
                    f"你可以在后续对话中自然地问以下问题来测试客服的知识准确性：\n"
                    + "\n".join(f"- {q}" for q in unanswered[:3])
                )

        # 流程感知：引导模拟器推动流程进展
        if rubric.call_flow:
            steps_summary = []
            for step in rubric.call_flow:
                steps_summary.append(f"步骤{step.step_id}: {step.title}")
            parts.append(
                "## 通话流程\n"
                "客服应按照以下流程推进对话。你可以配合流程进展，"
                "在每步完成后给简短回应再进入下一步：\n"
                + "\n".join(steps_summary)
            )

        # 约束感知：引导模拟器测试边界
        constraints = rubric.constraints
        if constraints:
            constraint_hints = []
            max_words = constraints.get("max_words_per_turn", 0)
            if max_words:
                constraint_hints.append(f"注意客服是否每次回复都在{max_words}字以内")
            forbidden = constraints.get("forbidden_phrases", [])
            if isinstance(forbidden, str):
                forbidden = [forbidden]
            if forbidden:
                constraint_hints.append(f"注意客服是否使用了禁止词汇：{'、'.join(forbidden)}")
            if constraint_hints:
                parts.append("## 约束测试\n" + "\n".join(f"- {h}" for h in constraint_hints))

        return "\n\n".join(parts) if parts else ""
