"""评测器核心逻辑单元测试"""

import pytest
from unittest.mock import MagicMock

from core.evaluator import (
    Evaluator, EvalResult, DimensionScore, MultiJudgeResult,
    DIMENSIONS, FORBIDDEN_PATTERNS, JUDGE_CONFIGS,
)
from core.dialogue_runner import DialogResult, Turn
from core.scenario_builder import Scenario, TaskRubric, CallFlowStep


# ── 测试夹具 ──

def make_scenario(persona_name="普通配合型") -> Scenario:
    return Scenario(
        persona_id="cooperative",
        persona_name=persona_name,
        persona_description="正常配合",
        behavior=["认真听"],
        test_goal="测试基本流程",
    )


def make_rubric(**kwargs) -> TaskRubric:
    defaults = {
        "task_goal": "确认用户意向",
        "must_do": ["介绍活动", "确认意向"],
        "must_not_do": ["泄露隐私"],
        "constraints": {"max_turns": 8},
        "success_criteria": ["获得明确意向"],
    }
    defaults.update(kwargs)
    return TaskRubric(**defaults)


def make_dialog(turns_data: list[tuple[str, str]], persona_name="普通配合型") -> DialogResult:
    """快速构建 DialogResult。turns_data: [(user_msg, assistant_msg), ...]"""
    scenario = make_scenario(persona_name)
    turns = [
        Turn(user=u, assistant=a, turn_number=i + 1)
        for i, (u, a) in enumerate(turns_data)
    ]
    return DialogResult(scenario=scenario, turns=turns, finished_reason="natural_end")


# ── 规则检测测试 ──

class TestRuleCheck:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_no_violations(self):
        dialog = make_dialog([
            ("你好", "您好，有什么可以帮您？"),
            ("好的", "感谢您的配合，再见"),
        ])
        rubric = make_rubric()
        violations = self.evaluator._rule_check(dialog, rubric)
        assert len(violations) == 0

    def test_forbidden_pattern_id_card(self):
        dialog = make_dialog([
            ("你好", "请告诉我您的身份证号"),
        ])
        rubric = make_rubric()
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("身份证号" in v for v in violations)

    def test_forbidden_pattern_guarantee(self):
        dialog = make_dialog([
            ("能到账吗", "保证到账，您放心"),
        ])
        rubric = make_rubric()
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("保证" in v for v in violations)

    def test_single_turn_violation(self):
        dialog = make_dialog([("你好", "您好")])
        rubric = make_rubric()
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("1 轮" in v for v in violations)

    def test_max_turns_violation(self):
        turns = [(f"用户{i}", f"客服{i}") for i in range(10)]
        dialog = make_dialog(turns)
        rubric = make_rubric(constraints={"max_turns": 5})
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("超过最大限制轮次" in v for v in violations)

    def test_word_count_violation(self):
        long_msg = "这是一段很长的回复" * 20
        dialog = make_dialog([("你好", long_msg)])
        rubric = make_rubric(constraints={"max_words_per_turn": 30})
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("超字数限制" in v for v in violations)

    def test_forbidden_phrase_violation(self):
        dialog = make_dialog([("你好", "好的，我知道了")])
        rubric = make_rubric(constraints={"forbidden_phrases": ["好的"]})
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("禁止短语" in v for v in violations)

    def test_max_turns_string_type(self):
        """测试 LLM 返回字符串类型时的类型安全"""
        dialog = make_dialog([(f"u{i}", f"a{i}") for i in range(10)])
        rubric = make_rubric(constraints={"max_turns": "5"})
        violations = self.evaluator._rule_check(dialog, rubric)
        assert any("超过最大限制轮次" in v for v in violations)


# ── 加权总分测试 ──

class TestWeightedScore:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_all_five_score(self):
        result = EvalResult()
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(score=5, reason="完美")
        score = self.evaluator._compute_weighted_score(result)
        assert score == 100.0

    def test_all_zero_score(self):
        result = EvalResult()
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(score=0, reason="失败")
        score = self.evaluator._compute_weighted_score(result)
        assert score == 0.0

    def test_mixed_scores(self):
        result = EvalResult()
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(score=3, reason="及格")
        score = self.evaluator._compute_weighted_score(result)
        assert score == 60.0

    def test_weights_sum_to_one(self):
        total = sum(d["weight"] for d in DIMENSIONS)
        assert abs(total - 1.0) < 1e-9


# ── 评分校准测试 ──

class TestCalibrateScore:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_short_dialog_no_perfect_score(self):
        """1-2轮对话不应有满分"""
        dialog = make_dialog([("你好", "您好，介绍活动")])
        result = EvalResult()
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(score=5, reason="完美")
        calibrated = self.evaluator._calibrate_score(result, dialog)
        for dim in DIMENSIONS:
            assert calibrated.dimensions[dim["key"]].score <= 4

    def test_long_dialog_allows_perfect(self):
        """3轮以上对话允许满分"""
        dialog = make_dialog([
            ("你好", "您好"),
            ("好的", "介绍活动"),
            ("可以", "确认意向"),
            ("再见", "感谢"),
        ])
        result = EvalResult()
        for dim in DIMENSIONS:
            result.dimensions[dim["key"]] = DimensionScore(score=5, reason="完美")
        result.overall_score = 100.0
        calibrated = self.evaluator._calibrate_score(result, dialog)
        # 不应被折扣（轮次>3）
        assert calibrated.overall_score == 100.0

    def test_single_turn_score_cap(self):
        """1轮对话4分应降为3分"""
        dialog = make_dialog([("你好", "您好")])
        result = EvalResult()
        result.dimensions["task_completion"] = DimensionScore(score=4, reason="还行")
        calibrated = self.evaluator._calibrate_score(result, dialog)
        assert calibrated.dimensions["task_completion"].score == 3

    def test_violations_with_high_constraint_score(self):
        """有违规但约束遵守度高分时应降分"""
        dialog = make_dialog([
            ("你好", "您好"),
            ("好的", "介绍活动"),
            ("再见", "感谢"),
        ])
        result = EvalResult()
        result.dimensions["constraint_adherence"] = DimensionScore(score=4, reason="还行")
        result.violations = ["违规1", "违规2"]
        calibrated = self.evaluator._calibrate_score(result, dialog)
        assert calibrated.dimensions["constraint_adherence"].score <= 2


# ── 多评委配置测试 ──

class TestJudgeConfigs:
    def test_three_configs_exist(self):
        assert len(JUDGE_CONFIGS) == 3

    def test_different_temperatures(self):
        temps = [c["temperature"] for c in JUDGE_CONFIGS]
        assert len(set(temps)) == 3, "三个评委温度应各不相同"

    def test_each_config_has_required_fields(self):
        for config in JUDGE_CONFIGS:
            assert "temperature" in config
            assert "persona" in config
            assert "bias" in config

    def test_strict_has_lowest_temperature(self):
        strict = next(c for c in JUDGE_CONFIGS if c["persona"] == "严格评委")
        assert strict["temperature"] == min(c["temperature"] for c in JUDGE_CONFIGS)


# ── 违规定位测试 ──

class TestLocateViolations:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_locate_id_card_violation(self):
        dialog = make_dialog([
            ("你好", "请告诉我您的身份证号"),
            ("好的", "好的已记录"),
        ])
        rubric = make_rubric()
        locations = self.evaluator.locate_violations(dialog, rubric)
        assert len(locations) > 0
        key = list(locations.keys())[0]
        assert "身份证号" in key
        assert locations[key][0][0] == 1  # 第1轮

    def test_no_violations_empty_result(self):
        dialog = make_dialog([
            ("你好", "您好，有什么可以帮您？"),
        ])
        rubric = make_rubric()
        locations = self.evaluator.locate_violations(dialog, rubric)
        assert len(locations) == 0


# ── 结构化评测测试 ──

class TestStructuredEvaluation:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_opening_line_perfect_match(self):
        dialog = make_dialog([
            ("喂", "你好 请问是 张先生 吗 我是 站长"),
        ])
        rubric = make_rubric(opening_line="你好 请问是 ${name} 吗 我是 站长")
        score, reason = self.evaluator._check_opening_line(dialog, rubric)
        assert score >= 4

    def test_opening_line_no_match(self):
        dialog = make_dialog([
            ("喂", "嗨，今天天气不错啊"),
        ])
        rubric = make_rubric(opening_line="你好，请问是${name}吗？我是站长")
        score, reason = self.evaluator._check_opening_line(dialog, rubric)
        assert score <= 2

    def test_no_dialog_returns_zero(self):
        scenario = make_scenario()
        dialog = DialogResult(scenario=scenario, turns=[], finished_reason="error")
        rubric = make_rubric(opening_line="你好")
        score, reason = self.evaluator._check_opening_line(dialog, rubric)
        assert score == 0

    def test_call_flow_no_steps(self):
        dialog = make_dialog([("你好", "您好")])
        rubric = make_rubric()
        score, reason = self.evaluator._check_call_flow(dialog, rubric)
        assert score == 5

    def test_faq_no_knowledge_points(self):
        dialog = make_dialog([("你好", "您好")])
        rubric = make_rubric()
        score, reason = self.evaluator._check_faq_accuracy(dialog, rubric)
        assert score == 5


# ── 维度配置测试 ──

class TestDimensions:
    def test_ten_dimensions(self):
        assert len(DIMENSIONS) == 10

    def test_all_keys_unique(self):
        keys = [d["key"] for d in DIMENSIONS]
        assert len(keys) == len(set(keys))

    def test_all_have_required_fields(self):
        for dim in DIMENSIONS:
            assert "key" in dim
            assert "name" in dim
            assert "weight" in dim
            assert "description" in dim

    def test_weights_are_positive(self):
        for dim in DIMENSIONS:
            assert dim["weight"] > 0


# ── 提取关键词测试 ──

class TestExtractKeywords:
    def setup_method(self):
        self.evaluator = Evaluator()

    def test_chinese_keywords(self):
        keywords = self.evaluator._extract_keywords("你好 请问 张先生")
        assert "你好" in keywords
        assert "请问" in keywords
        assert "张先生" in keywords

    def test_removes_punctuation(self):
        keywords = self.evaluator._extract_keywords("你好，请问！是吗？")
        assert all(len(k) >= 2 for k in keywords)

    def test_deduplication(self):
        keywords = self.evaluator._extract_keywords("你好 你好 请问 请问")
        assert keywords.count("你好") == 1
        assert keywords.count("请问") == 1

    def test_short_words_filtered(self):
        keywords = self.evaluator._extract_keywords("我 是 你好")
        assert "我" not in keywords
        assert "是" not in keywords
        assert "你好" in keywords
