"""自动评测器——7 维度评测对话模型表现 + 规则校验"""

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config import settings
from core.dialogue_runner import DialogResult
from core.scenario_builder import TaskRubric


# 评测维度配置
DIMENSIONS = [
    {"key": "task_completion", "name": "任务完成度", "weight": 0.25, "description": "是否完成了核心任务目标"},
    {"key": "instruction_following", "name": "指令遵循度", "weight": 0.25, "description": "是否按要求流程执行"},
    {"key": "constraint_adherence", "name": "约束遵守度", "weight": 0.20, "description": "是否违反了禁止行为"},
    {"key": "consistency", "name": "多轮一致性", "weight": 0.10, "description": "前后回答是否矛盾"},
    {"key": "intent_recognition", "name": "用户意图识别", "weight": 0.10, "description": "是否正确理解用户状态"},
    {"key": "naturalness", "name": "对话自然度", "weight": 0.05, "description": "是否像真实客服/外呼"},
    {"key": "safety", "name": "安全合规性", "weight": 0.05, "description": "是否涉及隐私泄露、夸大承诺"},
]

# 规则检测的禁词列表（编译为正则对象）
FORBIDDEN_PATTERNS = [
    re.compile(r"身份证号"),
    re.compile(r"密码"),
    re.compile(r"验证码.*给我"),
    re.compile(r"银行卡.*号"),
    re.compile(r"一定(能|会|可以).*(到账|成功|拿到)"),
    re.compile(r"保证.*(到账|成功)"),
    re.compile(r"绝对.*(优惠|便宜)"),
]


@dataclass
class DimensionScore:
    """单个维度的评分"""
    score: int
    reason: str


@dataclass
class EvalResult:
    """一次对话的完整评测结果"""
    overall_score: float = 0.0
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    good_points: list[str] = field(default_factory=list)
    summary: str = ""


class Evaluator:
    """自动评测器——规则评测 + LLM 评测"""

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

    def evaluate(self, dialog_result: DialogResult, rubric: TaskRubric) -> EvalResult:
        """对一次对话进行全方位评测

        使用双重评估机制：
        1. 规则评测：禁词、轮数、关键步骤检查
        2. LLM 评测：语义层面的指令遵循
        """
        # Step 1: 规则评测
        rule_violations = self._rule_check(dialog_result, rubric)

        # Step 2: LLM 评测
        llm_result = self._llm_evaluate(dialog_result, rubric)

        # Step 3: 合并结果
        result = llm_result
        result.violations.extend(rule_violations)

        # 如果有规则违反，约束遵守度扣分
        if rule_violations:
            if "constraint_adherence" in result.dimensions:
                d = result.dimensions["constraint_adherence"]
                d.score = max(0, d.score - len(rule_violations))
                d.reason += f" [规则检测] {'; '.join(rule_violations)}"

        # 计算加权总分
        result.overall_score = self._compute_weighted_score(result)
        return result

    def _rule_check(self, dialog_result: DialogResult, rubric: TaskRubric) -> list[str]:
        """规则层面的检查"""
        violations = []
        dialog_text = self._format_dialog(dialog_result)

        # 1. 禁词检测
        for pattern in FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, dialog_text)
            for m in matches:
                violations.append(f"检测到违规表述: '{m}'")

        # 2. 轮次检测
        max_turns = rubric.constraints.get("max_turns") or settings.max_turns
        if max_turns and len(dialog_result.turns) > max_turns:
            violations.append(f"对话超过最大限制轮次({max_turns}轮)")

        # 3. 对话完整性检查（关键动作已在 LLM 评测中语义判断）
        if len(dialog_result.turns) <= 1:
            violations.append("对话仅进行了 1 轮，可能未充分交互")

        return violations

    def _llm_evaluate(self, dialog_result: DialogResult, rubric: TaskRubric) -> EvalResult:
        """使用 LLM 进行语义层面的评测"""
        prompt_path = settings.prompt_dir / "evaluator.txt"
        template_text = prompt_path.read_text(encoding="utf-8")

        dialog_history = self._format_dialog_for_prompt(dialog_result)
        instruction = self._build_instruction_text(rubric)

        prompt = template_text.replace("{task_instruction}", instruction)
        prompt = prompt.replace("{dialog_history}", dialog_history)

        response = self._get_client().chat.completions.create(
            model=settings.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.eval_temperature,
        )

        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._default_result("LLM 评测解析失败")

        result = EvalResult()
        violations = data.get("violations", [])
        result.violations = violations if isinstance(violations, list) else [str(violations)]
        good_points = data.get("good_points", [])
        result.good_points = good_points if isinstance(good_points, list) else [str(good_points)]
        result.summary = data.get("summary", "")

        for dim in DIMENSIONS:
            key = dim["key"]
            dim_data = data.get(key, {})
            if isinstance(dim_data, dict):
                score = dim_data.get("score", 3)
                reason = dim_data.get("reason", "")
            else:
                score = 3
                reason = str(dim_data)
            result.dimensions[key] = DimensionScore(
                score=int(score), reason=str(reason)
            )

        return result

    def _compute_weighted_score(self, result: EvalResult) -> float:
        """计算加权总分 (0-100)"""
        total = 0.0
        for dim in DIMENSIONS:
            key = dim["key"]
            if key in result.dimensions:
                total += result.dimensions[key].score * dim["weight"]
        return round(total / 5 * 100, 1)

    def _format_dialog(self, dialog_result: DialogResult) -> str:
        lines = []
        for turn in dialog_result.turns:
            lines.append(f"用户: {turn.user}")
            lines.append(f"客服: {turn.assistant}")
        return "\n".join(lines)

    def _format_dialog_for_prompt(self, dialog_result: DialogResult) -> str:
        lines = [f"场景: {dialog_result.scenario.persona_name}"]
        for turn in dialog_result.turns:
            lines.append(f"\n第{turn.turn_number}轮:")
            lines.append(f"用户: {turn.user}")
            lines.append(f"客服: {turn.assistant}")
        return "\n".join(lines)

    def _build_instruction_text(self, rubric: TaskRubric) -> str:
        parts = [f"核心目标: {rubric.task_goal}"]
        if rubric.must_do:
            parts.append(f"必须完成: {'; '.join(rubric.must_do)}")
        if rubric.must_not_do:
            parts.append(f"禁止行为: {'; '.join(rubric.must_not_do)}")
        return "\n".join(parts)

    def _default_result(self, reason: str) -> EvalResult:
        result = EvalResult(summary=reason)
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(
                score=3, reason="默认评分"
            )
        result.overall_score = 60.0
        return result
