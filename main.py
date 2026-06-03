"""FastAPI 入口（可选）"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from core.scenario_builder import ScenarioBuilder
from core.dialogue_runner import DialogueRunner
from core.evaluator import Evaluator, DIMENSIONS, DimensionScore
from core.report_generator import ReportGenerator

app = FastAPI(title="EvalAgent API", version="0.1.0")

scenario_builder = ScenarioBuilder()
dialogue_runner = DialogueRunner()
evaluator = Evaluator()
report_gen = ReportGenerator()


class EvalRequest(BaseModel):
    task_instruction: str
    personas: list[str] = ["cooperative", "rejecting", "inquiring", "distracting", "adversarial", "ambiguous"]
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
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="API Key 未配置，请在 .env 中设置")

    try:
        # 1. 解析指令
        rubric = scenario_builder.parse_instruction(req.task_instruction)

        # 2. 加载场景
        scenarios = scenario_builder.load_scenarios(req.personas)
        if not scenarios:
            raise HTTPException(status_code=400, detail="未找到有效的用户画像场景")

        results = []
        for scenario in scenarios:
            try:
                # 运行对话
                dialog_result = dialogue_runner.run_dialog(
                    scenario=scenario,
                    rubric=rubric,
                    max_turns=req.max_turns,
                )

                # 多评委评测（与 Streamlit 端一致）
                mj_result = evaluator.multi_judge_evaluate(dialog_result, rubric, num_judges=3)
                eval_result = mj_result.individual_results[-1]
                violation_locs = evaluator.locate_violations(dialog_result, rubric)

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
                        "overall_score": mj_result.overall_mean,
                        "overall_std": mj_result.overall_std,
                        "num_judges": mj_result.num_judges_used,
                        "arbitration_triggered": mj_result.arbitration_triggered,
                        "dimensions": {
                            dim["name"]: {
                                "score": eval_result.dimensions.get(dim["key"], DimensionScore(0, "")).score,
                                "reason": eval_result.dimensions.get(dim["key"], DimensionScore(0, "")).reason,
                            }
                            for dim in DIMENSIONS
                        },
                        "violations": mj_result.violations,
                        "good_points": mj_result.good_points,
                        "summary": mj_result.summary,
                    },
                    "violation_locations": {
                        k: [{"turn": t, "role": r, "text": s} for t, r, s in v]
                        for k, v in violation_locs.items()
                    } if violation_locs else {},
                })
            except Exception as e:
                results.append({
                    "scenario": {"persona_name": scenario.persona_name, "persona_id": scenario.persona_id},
                    "dialog": {"turns": 0, "finished_reason": "error", "records": []},
                    "evaluation": {"overall_score": 0, "dimensions": {}, "violations": [str(e)], "summary": "场景评测失败"},
                })

        return EvalResponse(status="success", results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评测服务内部错误: {str(e)}")
