"""场景构建器单元测试"""

import json
import pytest
from core.scenario_builder import TaskRubric, CallFlowStep, Scenario, ScenarioBuilder


class TestTaskRubric:
    def test_from_dict_minimal(self):
        data = {"task_goal": "测试目标"}
        rubric = TaskRubric.from_dict(data)
        assert rubric.task_goal == "测试目标"
        assert rubric.must_do == []
        assert rubric.must_not_do == []

    def test_from_dict_full(self):
        data = {
            "task_goal": "确认意向",
            "must_do": ["介绍活动"],
            "must_not_do": ["泄露隐私"],
            "constraints": {"max_turns": 8},
            "success_criteria": ["获得意向"],
            "opening_line": "你好",
            "call_flow": [{"step_id": "1", "title": "步骤1"}],
            "knowledge_points": {"问题": "答案"},
            "role": "客服",
            "raw_instruction": "原始指令",
        }
        rubric = TaskRubric.from_dict(data)
        assert rubric.task_goal == "确认意向"
        assert rubric.must_do == ["介绍活动"]
        assert rubric.role == "客服"
        assert len(rubric.call_flow) == 1
        assert rubric.knowledge_points == {"问题": "答案"}

    def test_has_structured_instruction_true(self):
        rubric = TaskRubric(task_goal="test", call_flow=[CallFlowStep("1", "步骤1")])
        assert rubric.has_structured_instruction is True

    def test_has_structured_instruction_false(self):
        rubric = TaskRubric(task_goal="test")
        assert rubric.has_structured_instruction is False

    def test_to_dict_roundtrip(self):
        data = {
            "task_goal": "测试",
            "must_do": ["动作1"],
            "must_not_do": [],
            "constraints": {},
            "success_criteria": [],
            "opening_line": "",
            "call_flow": [],
            "knowledge_points": {},
            "role": "",
            "raw_instruction": "",
        }
        rubric = TaskRubric.from_dict(data)
        result = rubric.to_dict()
        assert result["task_goal"] == "测试"
        assert result["must_do"] == ["动作1"]


class TestCallFlowStep:
    def test_from_dict(self):
        data = {"step_id": "1", "title": "步骤1", "reference_script": "话术"}
        step = CallFlowStep.from_dict(data)
        assert step.step_id == "1"
        assert step.title == "步骤1"
        assert step.reference_script == "话术"

    def test_to_dict(self):
        step = CallFlowStep("1", "步骤1", "描述", [], "话术")
        d = step.to_dict()
        assert d["step_id"] == "1"
        assert d["title"] == "步骤1"

    def test_defaults(self):
        step = CallFlowStep("1", "步骤1")
        assert step.description == ""
        assert step.sub_steps == []
        assert step.reference_script == ""


class TestScenario:
    def test_persona_prompt(self):
        scenario = Scenario(
            persona_id="cooperative",
            persona_name="普通配合型",
            persona_description="正常配合",
            behavior=["认真听", "正常回答"],
            test_goal="测试基本流程",
        )
        prompt = scenario.persona_prompt
        assert "普通配合型" in prompt
        assert "认真听" in prompt

    def test_to_dict(self):
        scenario = Scenario("id", "name", "desc", ["行为"], "目标")
        d = scenario.to_dict()
        assert d["persona_id"] == "id"
        assert d["persona_name"] == "name"


class TestParseStructuredInstruction:
    def test_basic_parse(self):
        builder = ScenarioBuilder()
        data = {
            "role": "客服",
            "task": "介绍活动",
            "opening_line": "你好",
            "call_flow": [{"step_id": "1", "title": "步骤1"}],
            "knowledge_points": {"问题": "答案"},
            "constraints": {"tone": "礼貌", "forbidden_phrases": ["好的"], "max_words_per_turn": "30"},
        }
        rubric = builder.parse_structured_instruction(data)
        assert rubric.task_goal == "介绍活动"
        assert rubric.role == "客服"
        assert rubric.opening_line == "你好"
        assert len(rubric.call_flow) == 1
        assert rubric.knowledge_points == {"问题": "答案"}
        assert "语气要求: 礼貌" in rubric.must_do
        assert "禁止说'好的'" in rubric.must_not_do
        assert rubric.constraints["max_words_per_turn"] == 30

    def test_success_criteria_generated(self):
        builder = ScenarioBuilder()
        data = {
            "task": "测试",
            "opening_line": "你好",
            "call_flow": [{"step_id": "1", "title": "步骤1"}, {"step_id": "2", "title": "步骤2"}],
            "knowledge_points": {"q": "a"},
        }
        rubric = builder.parse_structured_instruction(data)
        assert len(rubric.success_criteria) == 3
        assert any("2" in s for s in rubric.success_criteria)
