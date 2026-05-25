"""自动评测器——7 维度评测对话模型表现 + 规则校验"""

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config import settings
from core.dialogue_runner import DialogResult
from core.scenario_builder import TaskRubric


# 评测维度配置（权重合计=1.0）
DIMENSIONS = [
    {"key": "task_completion", "name": "任务完成度", "weight": 0.20, "description": "是否完成了核心任务目标"},
    {"key": "instruction_following", "name": "指令遵循度", "weight": 0.20, "description": "是否按要求流程执行"},
    {"key": "call_flow_completion", "name": "流程完成度", "weight": 0.10, "description": "Call Flow 各步骤是否完整执行"},
    {"key": "constraint_adherence", "name": "约束遵守度", "weight": 0.15, "description": "是否违反了禁止行为"},
    {"key": "consistency", "name": "多轮一致性", "weight": 0.10, "description": "前后回答是否矛盾"},
    {"key": "intent_recognition", "name": "用户意图识别", "weight": 0.08, "description": "是否正确理解用户状态"},
    {"key": "faq_accuracy", "name": "知识准确性", "weight": 0.05, "description": "FAQ回答是否与知识库一致"},
    {"key": "opening_compliance", "name": "开场合规度", "weight": 0.05, "description": "开场白是否与指定模板一致"},
    {"key": "naturalness", "name": "对话自然度", "weight": 0.04, "description": "是否像真实客服/外呼"},
    {"key": "safety", "name": "安全合规性", "weight": 0.03, "description": "是否涉及隐私泄露、夸大承诺"},
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
    token_usage: int = 0
    # 违规定位：{违规描述: [(轮次, "user/assistant", 句子)]}
    violation_locations: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)


@dataclass
class MultiJudgeResult:
    """多评委一致性评分结果"""
    overall_mean: float = 0.0      # 平均分
    overall_std: float = 0.0       # 标准差（一致性指标）
    dimension_means: dict[str, float] = field(default_factory=dict)
    dimension_stds: dict[str, float] = field(default_factory=dict)
    individual_results: list[EvalResult] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    good_points: list[str] = field(default_factory=list)
    summary: str = ""
    num_judges_used: int = 3       # 实际使用的评委数（含仲裁）
    arbitration_triggered: bool = False  # 是否触发了仲裁


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
            self._client = OpenAI(timeout=300.0, **kwargs)
        return self._client

    def evaluate(self, dialog_result: DialogResult, rubric: TaskRubric) -> EvalResult:
        """对一次对话进行全方位评测

        使用多重评估机制：
        1. 规则评测：禁词、轮数、字数、禁止短语检查
        2. 结构化评测：开场白合规 / 流程完成度 / FAQ准确性（如适用）
        3. LLM 评测：语义层面的指令遵循
        """
        # Step 1: 规则评测
        rule_violations = self._rule_check(dialog_result, rubric)

        # Step 2: LLM 评测
        llm_result = self._llm_evaluate(dialog_result, rubric)

        # Step 3: 合并结果
        result = llm_result
        result.violations.extend(rule_violations)

        # Step 4: 结构化评测（仅当有结构化指令时）
        if rubric.has_structured_instruction:
            # 开场白合规检查
            if rubric.opening_line:
                opening_score, opening_reason = self._check_opening_line(
                    dialog_result, rubric
                )
                result.dimensions["opening_compliance"] = DimensionScore(
                    score=opening_score, reason=opening_reason,
                )

            # 流程完成度检查
            if rubric.call_flow:
                cf_score, cf_reason = self._check_call_flow(
                    dialog_result, rubric
                )
                result.dimensions["call_flow_completion"] = DimensionScore(
                    score=cf_score, reason=cf_reason,
                )

            # FAQ 准确性检查
            if rubric.knowledge_points:
                faq_score, faq_reason = self._check_faq_accuracy(
                    dialog_result, rubric
                )
                result.dimensions["faq_accuracy"] = DimensionScore(
                    score=faq_score, reason=faq_reason,
                )

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
            for m in pattern.finditer(dialog_text):
                violations.append(f"检测到违规表述: '{m.group(0)[:50]}'")

        # 2. 轮次检测
        max_turns = rubric.constraints.get("max_turns") or settings.max_turns
        if isinstance(max_turns, str):
            max_turns = int(max_turns)
        if max_turns and len(dialog_result.turns) > max_turns:
            violations.append(f"对话超过最大限制轮次({max_turns}轮)")

        # 3. 对话完整性检查
        if len(dialog_result.turns) <= 1:
            violations.append("对话仅进行了 1 轮，可能未充分交互")

        # 4. 字数限制检测（结构化指令专用）
        max_words = rubric.constraints.get("max_words_per_turn", 0)
        if isinstance(max_words, str):
            max_words = int(max_words)
        if max_words:
            for turn in dialog_result.turns:
                word_count = len(turn.assistant)
                if word_count > max_words:
                    violations.append(
                        f"第{turn.turn_number}轮回复超字数限制 "
                        f"({word_count}字 > {max_words}字限制)"
                    )

        # 5. 禁止短语检测（来自结构化约束）
        forbidden_phrases = rubric.constraints.get("forbidden_phrases", [])
        if isinstance(forbidden_phrases, str):
            forbidden_phrases = [forbidden_phrases]
        for phrase in forbidden_phrases:
            for turn in dialog_result.turns:
                if phrase in turn.assistant:
                    violations.append(
                        f"第{turn.turn_number}轮使用了禁止短语: '{phrase}'"
                    )

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

        # 健壮JSON解析：剥离markdown fence + 修复常见问题 + 最多3次重试
        data = None
        for attempt in range(3):
            cleaned = content
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            try:
                data = json.loads(cleaned)
                break
            except json.JSONDecodeError:
                if attempt < 2:
                    retry_prompt = f"你输出的JSON格式无效。请只输出严格JSON（用```json```包裹）。\n之前输出：{content[-200:]}\n请重新输出完整的评测JSON。"
                    resp2 = self._get_client().chat.completions.create(
                        model=settings.openai_model_name,
                        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": content}, {"role": "user", "content": retry_prompt}],
                        temperature=settings.eval_temperature,
                    )
                    content = resp2.choices[0].message.content.strip()
                else:
                    return self._default_result("LLM评测JSON解析失败(3次重试)")

        if data is None:
            return self._default_result("LLM评测JSON解析失败")

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
                score = int(dim_data.get("score", 3))
                reason = dim_data.get("reason", "")
            else:
                score = 3
                reason = str(dim_data) if dim_data else ""
            if not reason:
                reason = f"基于对话内容的{key}评分"
            result.dimensions[key] = DimensionScore(
                score=min(5, max(0, score)), reason=str(reason),
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
                score=2, reason=f"评测解析异常: {reason}"
            )
        result.overall_score = 40.0
        return result

    # ── 结构化评测方法 ──

    def _check_opening_line(
        self, dialog_result: DialogResult, rubric: TaskRubric
    ) -> tuple[int, str]:
        """检查模型首轮回复是否使用了指定的开场白

        Returns:
            (score 0-5, reason)
        """
        if not dialog_result.turns:
            return 0, "无对话记录，无法检查开场白"

        first_turn = dialog_result.turns[0]
        opening = rubric.opening_line

        # 去除模板变量（如 ${rider_name}）后做模糊匹配
        clean_opening = re.sub(r'\$\{[^}]+\}', '', opening).strip()
        clean_response = first_turn.assistant.strip()

        # 计算关键词覆盖率
        opening_keywords = self._extract_keywords(clean_opening)
        matched = [kw for kw in opening_keywords if kw in clean_response]
        coverage = len(matched) / len(opening_keywords) if opening_keywords else 0

        if coverage >= 0.8:
            return 5, f"开场白与模板高度一致（关键词覆盖率 {coverage:.0%}）：{first_turn.assistant[:60]}..."
        elif coverage >= 0.5:
            return 3, f"开场白部分匹配模板（关键词覆盖率 {coverage:.0%}），缺失: {set(opening_keywords) - set(matched)}"
        elif coverage > 0:
            return 2, f"开场白与模板差异较大（覆盖率仅 {coverage:.0%}），实际: {first_turn.assistant[:60]}..."
        else:
            return 1, f"开场白未匹配模板关键词。期望: {clean_opening[:60]}... 实际: {first_turn.assistant[:60]}..."

    def _check_call_flow(
        self, dialog_result: DialogResult, rubric: TaskRubric
    ) -> tuple[int, str]:
        """检查对话流程是否覆盖了指定步骤

        使用 LLM 做语义判断：哪些步骤在对话中得到了体现

        Returns:
            (score 0-5, reason)
        """
        if not rubric.call_flow:
            return 5, "无指定流程步骤"

        # 构造步骤描述
        steps_desc = []
        for step in rubric.call_flow:
            desc = f"步骤{step.step_id}: {step.title}"
            if step.reference_script:
                desc += f" (参考话术: {step.reference_script[:60]})"
            steps_desc.append(desc)

        dialog_text = self._format_dialog(dialog_result)

        prompt = f"""你是一个对话流程审核员。检查以下客服对话是否完成了指定的通话流程步骤。

## 流程步骤
{chr(10).join(steps_desc)}

## 对话记录
{dialog_text}

## 任务
对每个步骤判断是否在对话中被执行（considering 语义相似度，不要求字面一样）。
输出 JSON 格式（不要其他内容）：
{{
  "steps_completed": ["1", "2"],       // 已完成的步骤ID列表
  "steps_partial": ["3"],              // 部分完成的步骤ID列表
  "steps_missed": ["4", "5"],          // 未执行的步骤ID列表
  "overall_assessment": "总体评价（一句话）"
}}"""

        try:
            response = self._get_client().chat.completions.create(
                model=settings.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
            data = json.loads(content)

            completed = len(data.get("steps_completed", []))
            partial = len(data.get("steps_partial", []))
            missed = len(data.get("steps_missed", []))
            total = len(rubric.call_flow)

            effective = completed + partial * 0.5
            ratio = effective / total if total > 0 else 1

            if ratio >= 0.9:
                score = 5
            elif ratio >= 0.7:
                score = 4
            elif ratio >= 0.5:
                score = 3
            elif ratio >= 0.3:
                score = 2
            else:
                score = 1

            reason_parts = [f"完成 {completed}/{total} 步骤"]
            if partial:
                reason_parts.append(f"部分完成 {partial} 步骤")
            if missed:
                reason_parts.append(f"缺失步骤: {', '.join(data['steps_missed'])}")
            reason = "；".join(reason_parts)
            if data.get("overall_assessment"):
                reason += f"。{data['overall_assessment']}"

            return score, reason

        except Exception:
            return 3, "流程完成度评测失败，默认3分"

    def _check_faq_accuracy(
        self, dialog_result: DialogResult, rubric: TaskRubric
    ) -> tuple[int, str]:
        """检查模型回答是否与知识库（FAQ）一致

        使用 LLM 对比客服回答与知识库标准答案

        Returns:
            (score 0-5, reason)
        """
        if not rubric.knowledge_points:
            return 5, "无知识库条目"

        # 构造知识库文本
        faq_lines = []
        for question, answer in rubric.knowledge_points.items():
            faq_lines.append(f"Q: {question}\nA: {answer}")

        dialog_text = self._format_dialog(dialog_result)

        prompt = f"""你是一个知识准确性审核员。检查客服在对话中的回答是否与知识库一致。

## 知识库
{chr(10).join(faq_lines)}

## 对话记录
{dialog_text}

## 任务
判断客服回答是否与知识库内容一致。考虑：
1. 事实是否准确
2. 是否有遗漏关键信息
3. 是否有编造或错误信息

输出 JSON 格式（不要其他内容）：
{{
  "accurate_answers": 3,      // 与知识库一致的回答数
  "inaccurate_answers": 1,    // 与知识库不一致/矛盾的回答数
  "irrelevant_answers": 2,    // 不涉及知识库的回答数
  "inaccuracy_details": [     // 不准确的详情
    {{"question": "用户问了什么", "expected": "知识库答案", "actual": "客服答案", "issue": "问题描述"}}
  ],
  "overall_assessment": "总体评价（一句话）"
}}"""

        try:
            response = self._get_client().chat.completions.create(
                model=settings.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
            data = json.loads(content)

            accurate = data.get("accurate_answers", 0)
            inaccurate = data.get("inaccurate_answers", 0)
            total = accurate + inaccurate

            if total == 0:
                return 3, "对话未涉及知识库内容，无法评估准确性"

            accuracy = accurate / total if total > 0 else 1

            if accuracy >= 0.95:
                score = 5
            elif accuracy >= 0.8:
                score = 4
            elif accuracy >= 0.6:
                score = 3
            elif accuracy >= 0.4:
                score = 2
            else:
                score = 1

            reason = f"知识准确性 {accuracy:.0%}（{accurate}/{total} 准确）"
            if data.get("inaccuracy_details"):
                details = "; ".join(
                    d.get("issue", "")[:50] for d in data["inaccuracy_details"]
                )
                reason += f"。不准确处: {details}"
            if data.get("overall_assessment"):
                reason += f"。{data['overall_assessment']}"

            return score, reason

        except Exception:
            return 3, "FAQ准确性评测失败，默认3分"

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词（用于模糊匹配）"""
        # 去除标点，按长度≥2的连续汉字/英文提取
        cleaned = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
        words = [w for w in cleaned.split() if len(w) >= 2]
        # 去重保持顺序
        seen = set()
        result = []
        for w in words:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result

    def multi_judge_evaluate(
        self, dialog_result: DialogResult, rubric: TaskRubric, num_judges: int = 3
    ) -> MultiJudgeResult:
        """多评委一致性评分——多次评测取均值+标准差，含仲裁机制"""
        import statistics

        results = []
        for _ in range(num_judges):
            results.append(self.evaluate(dialog_result, rubric))

        # ── 仲裁机制：如果一致性差（σ>10），自动加评委 ──
        max_arbitration = 1  # 最多仲裁1轮（速度优先）
        for arb_round in range(max_arbitration):
            overall_scores = [r.overall_score for r in results]
            if len(overall_scores) < 2:
                break
            current_std = statistics.stdev(overall_scores)
            if current_std < 10:  # 一致性可接受
                break
            # 加一个评委
            results.append(self.evaluate(dialog_result, rubric))

        # 计算各维度的均值和标准差
        dim_means = {}
        dim_stds = {}
        for dim in DIMENSIONS:
            key = dim["key"]
            scores = [r.dimensions[key].score for r in results if key in r.dimensions]
            dim_means[key] = statistics.mean(scores) if scores else 0
            dim_stds[key] = statistics.stdev(scores) if len(scores) > 1 else 0

        overall_scores = [r.overall_score for r in results]
        overall_mean = statistics.mean(overall_scores)
        overall_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0

        # 合并违规项和亮点（去重）
        all_violations = []
        all_good = []
        seen_v = set()
        seen_g = set()
        for r in results:
            for v in r.violations:
                if v not in seen_v:
                    all_violations.append(v)
                    seen_v.add(v)
            for g in r.good_points:
                if g not in seen_g:
                    all_good.append(g)
                    seen_g.add(g)

        return MultiJudgeResult(
            overall_mean=round(overall_mean, 1),
            overall_std=round(overall_std, 1),
            dimension_means={k: round(v, 1) for k, v in dim_means.items()},
            dimension_stds={k: round(v, 1) for k, v in dim_stds.items()},
            individual_results=results,
            violations=all_violations,
            good_points=all_good,
            summary=results[-1].summary if results else "",
            num_judges_used=len(results),
            arbitration_triggered=len(results) > num_judges,
        )

    def locate_violations(
        self, dialog_result: DialogResult, rubric: TaskRubric
    ) -> dict[str, list[tuple[int, str, str]]]:
        """违规定位——找到违规行为发生在哪一轮哪句话"""
        locations = {}

        for pattern in FORBIDDEN_PATTERNS:
            for turn in dialog_result.turns:
                for role, text in [("assistant", turn.assistant), ("user", turn.user)]:
                    for m in pattern.finditer(text):
                        key = f"违规: '{m.group(0)}'"
                        if key not in locations:
                            locations[key] = []
                        locations[key].append((turn.turn_number, role, text[:60]))

        return locations
