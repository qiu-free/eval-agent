"""用户模拟器——根据用户画像生成多轮对话中的用户回复"""

from openai import OpenAI

from config import settings


class UserSimulator:
    """用户模拟器：扮演不同类型的用户与模型对话"""

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

    def generate_response(
        self,
        persona_prompt: str,
        test_goal: str,
        dialog_history: list[dict],
    ) -> str:
        """根据用户画像和对话历史生成用户的下一条回复

        Args:
            persona_prompt: 用户画像描述文本
            test_goal: 该场景的测试目标
            dialog_history: 对话历史 [{"role": "user"/"assistant", "content": "..."}, ...]

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
        prompt = prompt.replace("{test_goal}", test_goal)
        prompt = prompt.replace("{dialog_history}", history_str)

        response = self._get_client().chat.completions.create(
            model=settings.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.sim_temperature,
        )

        content = response.choices[0].message.content.strip()
        return content
