"""FastAPI 入口（可选）"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.scenario_builder import ScenarioBuilder
from core.dialogue_runner import DialogueRunner
from core.evaluator import Evaluator
from core.report_generator import ReportGenerator

app = FastAPI(title="EvalAgent API", version="0.1.0")

scenario_builder = ScenarioBuilder()
dialogue_runner = DialogueRunner()
evaluator = Evaluator()
report_gen = ReportGenerator()


class EvalRequest(BaseModel):
    task_instruction: str
    personas: list[str] = ["cooperative", "rejecting", "inquiring"]
    max_turns: int = 8


class EvalResponse(BaseModel):
    status: str
    results: list[dict]


@app.get("/")
def root():
    return {"service": "EvalAgent", "status": "running"}


@app.get("/scenarios")
def list_scenarios():
    """列出所有可用的用户画像场景"""
    scenarios = scenario_builder.load_scenarios()
    return {"scenarios": [s.to_dict() for s in scenarios]}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """运行一次完整的多轮对话评测"""
    # 1. 解析指令
    rubric = scenario_builder.parse_instruction(req.task_instruction)

    # 2. 加载场景
    scenarios = scenario_builder.load_scenarios(req.personas)

    results = []
    for scenario in scenarios:
        # 运行对话
        dialog_result = dialogue_runner.run_dialog(
            scenario=scenario,
            rubric=rubric,
            max_turns=req.max_turns,
        )

        # 评测
        eval_result = evaluator.evaluate(dialog_result, rubric)

        # 保存报告
        report_gen.save_report(
            dialog_result, eval_result, rubric,
            req.task_instruction, scenario.persona_name,
        )

        results.append({
            "scenario": scenario.to_dict(),
            "dialog": {
                "turns": len(dialog_result.turns),
                "finished_reason": dialog_result.finished_reason,
                "records": [
                    {"turn": t.turn_number, "user": t.user, "assistant": t.assistant}
                    for t in dialog_result.turns
                ],
            },
            "evaluation": {
                "overall_score": eval_result.overall_score,
                "dimensions": {
                    dim["name"]: {
                        "score": eval_result.dimensions[dim["key"]].score,
                        "reason": eval_result.dimensions[dim["key"]].reason,
                    }
                    for dim in [
                        {"key": "task_completion", "name": "任务完成度"},
                        {"key": "instruction_following", "name": "指令遵循度"},
                        {"key": "constraint_adherence", "name": "约束遵守度"},
                        {"key": "consistency", "name": "多轮一致性"},
                        {"key": "intent_recognition", "name": "用户意图识别"},
                        {"key": "naturalness", "name": "对话自然度"},
                        {"key": "safety", "name": "安全合规性"},
                    ]
                    if dim["key"] in eval_result.dimensions
                },
                "violations": eval_result.violations,
                "summary": eval_result.summary,
            },
        })

    return EvalResponse(status="success", results=results)
