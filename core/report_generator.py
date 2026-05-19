"""报告生成器——将评测结果格式化为可视化报告"""

import json
import datetime
from pathlib import Path

from config import settings
from core.evaluator import EvalResult, DIMENSIONS
from core.dialogue_runner import DialogResult
from core.scenario_builder import TaskRubric


class ReportGenerator:
    """评测报告生成器"""

    def generate_markdown(
        self,
        dialog_result: DialogResult,
        eval_result: EvalResult,
        rubric: TaskRubric,
        task_instruction: str,
    ) -> str:
        """生成 Markdown 格式评测报告"""
        lines = []

        lines.append("# EvalAgent 多轮对话评测报告\n")
        lines.append(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 一、任务概述
        lines.append("## 一、任务概述\n")
        lines.append(f"**任务指令**: {task_instruction}\n")
        lines.append(f"**任务目标**: {rubric.task_goal}\n")
        if rubric.must_do:
            lines.append(f"**必须完成**: {'、'.join(rubric.must_do)}\n")
        if rubric.must_not_do:
            lines.append(f"**禁止行为**: {'、'.join(rubric.must_not_do)}\n")

        # 二、评测场景
        lines.append("## 二、评测场景\n")
        s = dialog_result.scenario
        lines.append(f"**用户画像**: {s.persona_name}\n")
        lines.append(f"**测试目标**: {s.test_goal}\n")
        lines.append(f"**结束原因**: {self._end_reason_text(dialog_result.finished_reason)}\n")
        lines.append(f"**实际轮次**: {len(dialog_result.turns)} 轮\n")

        # 三、综合评分
        lines.append("## 三、综合评分\n")
        lines.append(f"### 总分: **{eval_result.overall_score}/100**\n")
        if eval_result.overall_score >= 80:
            lines.append("> 🟢 优秀 — 模型表现良好\n")
        elif eval_result.overall_score >= 60:
            lines.append("> 🟡 一般 — 存在可改进空间\n")
        else:
            lines.append("> 🔴 较差 — 需要重点关注\n")

        # 四、各维度评分
        lines.append("## 四、各维度评分\n")
        lines.append("| 维度 | 权重 | 得分 | 评分理由 |")
        lines.append("|------|------|------|----------|")
        for dim in DIMENSIONS:
            key = dim["key"]
            if key in eval_result.dimensions:
                ds = eval_result.dimensions[key]
                score_bar = "█" * ds.score + "░" * (5 - ds.score)
                lines.append(
                    f"| {dim['name']} | {dim['weight']*100:.0f}% "
                    f"| {ds.score}/5 {score_bar} | {ds.reason} |"
                )

        # 五、违规项
        if eval_result.violations:
            lines.append("\n## 五、违规项\n")
            for v in eval_result.violations:
                lines.append(f"- ⚠️ {v}")

        # 六、亮点
        if eval_result.good_points:
            lines.append("\n## 六、表现亮点\n")
            for p in eval_result.good_points:
                lines.append(f"- ✅ {p}")

        # 七、对话记录
        lines.append("\n## 七、对话记录\n")
        for turn in dialog_result.turns:
            lines.append(f"### 第 {turn.turn_number} 轮\n")
            lines.append(f"> **用户**: {turn.user}\n")
            lines.append(f"> **客服**: {turn.assistant}\n")

        # 八、总结
        lines.append("## 八、总结\n")
        lines.append(eval_result.summary + "\n")

        return "\n".join(lines)

    def generate_json(
        self,
        dialog_result: DialogResult,
        eval_result: EvalResult,
        rubric: TaskRubric,
        task_instruction: str,
    ) -> dict:
        """生成 JSON 格式评测数据"""
        return {
            "meta": {
                "task_instruction": task_instruction,
                "generated_at": datetime.datetime.now().isoformat(),
            },
            "scenario": dialog_result.scenario.to_dict(),
            "rubric": rubric.to_dict(),
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
                    for dim in DIMENSIONS
                    if dim["key"] in eval_result.dimensions
                },
                "violations": eval_result.violations,
                "good_points": eval_result.good_points,
                "summary": eval_result.summary,
            },
        }

    def save_report(
        self,
        dialog_result: DialogResult,
        eval_result: EvalResult,
        rubric: TaskRubric,
        task_instruction: str,
        scenario_name: str,
    ) -> Path:
        """保存评测报告到文件"""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = scenario_name.replace(" ", "_")
        filename = f"report_{safe_name}_{ts}.md"
        report_path = settings.output_dir / "reports" / filename

        md = self.generate_markdown(
            dialog_result, eval_result, rubric, task_instruction
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md, encoding="utf-8")

        # 同时保存 JSON
        json_filename = f"report_{safe_name}_{ts}.json"
        json_path = settings.output_dir / "reports" / json_filename
        data = self.generate_json(
            dialog_result, eval_result, rubric, task_instruction
        )
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return report_path

    def _end_reason_text(self, reason: str) -> str:
        mapping = {
            "max_turns": "达到最大轮次限制",
            "end_signal": "用户模拟器发送结束信号",
            "natural_end": "对话自然结束",
            "error": "执行出错",
        }
        return mapping.get(reason, reason)
