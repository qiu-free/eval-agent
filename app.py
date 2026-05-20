"""EvalAgent Streamlit 前端——输入指令、运行评测、查看报告"""

import html
import json
import traceback

import streamlit as st

from config import settings
from core.dialogue_runner import DialogResult, Turn
from core.scenario_builder import ScenarioBuilder
from core.dialogue_runner import DialogueRunner
from core.evaluator import Evaluator, DIMENSIONS, MultiJudgeResult
from core.report_generator import ReportGenerator

# ── 页面配置 ──
st.set_page_config(
    page_title="EvalAgent - 多轮对话自动评测系统",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 EvalAgent — 多轮对话自动评测系统")
st.markdown("> **美团 AI Hackathon** · 复杂指令下的多轮对话自动评测系统")


# ── 初始化 Session State ──
if "results" not in st.session_state:
    st.session_state.results = []
if "running" not in st.session_state:
    st.session_state.running = False



# ── 侧边栏 ──
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("API Key", type="password",
                            value=settings.openai_api_key or "",
                            help="留空则使用 .env 中的配置")
    if api_key:
        settings.openai_api_key = api_key

    model_name = st.text_input("模型名称", value=settings.openai_model_name)
    if model_name:
        settings.openai_model_name = model_name

    base_url = st.text_input("API Base URL", value=settings.openai_api_base,
                             help="DeepSeek: https://api.deepseek.com")
    if base_url:
        settings.openai_api_base = base_url

    st.divider()
    st.caption("💡 首次使用请在 `.env` 中配置 API Key")


# ── 选项卡 ──
tab1, tab2, tab3, tab4 = st.tabs(["🧪 模拟评测", "📤 上传评测", "📊 评测结果", "📖 使用说明"])


# ═══════════════════════════════════════════
# Tab 1: 模拟评测（用户模拟器）
# ═══════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        default_instruction = "向用户介绍优惠活动，确认用户是否有意向领取优惠券，遇到拒绝要挽回一次，不能透露内部价格策略，最终要完成意向收集。"
        task_instruction = st.text_area(
            "📝 输入任务指令", value=default_instruction, height=100,
            help="输入数字人外呼助手的任务指令",
            placeholder="例：向用户介绍活动，确认意向，遇到拒绝挽回一次…",
        )
    with col2:
        st.markdown("##### 🎭 选择用户画像")
        scenario_builder = ScenarioBuilder()
        all_scenarios = scenario_builder.load_scenarios()
        selected_personas = []
        for s in all_scenarios:
            default_on = s.persona_id in ["cooperative", "rejecting", "inquiring"]
            if st.checkbox(s.persona_name, value=default_on):
                selected_personas.append(s.persona_id)
        max_turns = st.slider("最大对话轮次", 3, 12, 8)

    run_btn = st.button(
        "🚀 开始评测", type="primary", use_container_width=True,
        disabled=st.session_state.running or not task_instruction.strip(),
    )

    # ── 评测执行区 ──
    if run_btn:
        if not settings.openai_api_key:
            st.error("❌ 请在侧边栏或 `.env` 中配置 API Key")
            st.stop()

        st.session_state.running = True
        st.session_state.results = []
        st.session_state.live_dialogs = {}

        # ── 整体进度条 ──
        overall_progress = st.progress(0, text="正在初始化...")
        status_box = st.empty()

        # ── 实时对话展示区 ──
        live_placeholder = st.empty()

        try:
            # Step 1: 解析任务指令
            overall_progress.progress(5, text="📋 解析任务指令...")
            status_box.info("📋 正在解析任务指令...")
            rubric = scenario_builder.parse_instruction(task_instruction)

            with st.expander("📋 指令解析结果", expanded=True):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**任务目标**: {rubric.task_goal}")
                    st.markdown(f"**必须做**: {'、'.join(rubric.must_do)}")
                with cb:
                    st.markdown(f"**禁止做**: {'、'.join(rubric.must_not_do)}")
                    st.markdown(f"**约束**: 最多 {rubric.constraints.get('max_turns', 'N/A')} 轮")

            # Step 2: 加载场景
            selected_scenarios = [s for s in all_scenarios if s.persona_id in selected_personas]
            total = len(selected_scenarios)
            if total == 0:
                st.warning("⚠️ 请至少选择一个用户画像")
                st.session_state.running = False
                st.rerun()

            status_box.info(f"🎭 共 {total} 个场景，开始评测...")

            # Step 3: 逐场景运行对话 + 评测
            dialogue_runner = DialogueRunner()
            evaluator = Evaluator()
            report_gen = ReportGenerator()

            for i, scenario in enumerate(selected_scenarios):
                status_box.info(f"▶️ [{i+1}/{total}] {scenario.persona_name} — 正在对话...")

                try:
                    # ── 运行对话（实时展示每一轮）──
                    dialog_result = dialogue_runner.run_dialog(
                        scenario=scenario, rubric=rubric, max_turns=max_turns,
                    )

                    # ── 展示对话内容 ──
                    with live_placeholder.container():
                        st.markdown(f"### 💬 {scenario.persona_name} — 对话记录")
                        chat_html = ""
                        for turn in dialog_result.turns:
                            chat_html += f"""
                            <div style="margin-bottom:12px">
                                <div style="display:flex; margin-bottom:4px">
                                    <div style="background:#e8f4fd; border-radius:12px 12px 12px 2px; padding:8px 14px; max-width:80%; color:#1a1a1a; font-size:14px">
                                        <b>🧑 用户:</b> {html.escape(turn.user)}
                                    </div>
                                </div>
                                <div style="display:flex; justify-content:flex-end; margin-bottom:4px">
                                    <div style="background:#e8fde8; border-radius:12px 12px 2px 12px; padding:8px 14px; max-width:80%; color:#1a1a1a; font-size:14px">
                                        <b>🤖 客服:</b> {html.escape(turn.assistant)}
                                    </div>
                                </div>
                            </div>
                            """
                        st.markdown(chat_html, unsafe_allow_html=True)

                    # ── 多评委评测 ──
                    status_box.info(f"▶️ [{i+1}/{total}] {scenario.persona_name} — 多评委评测中(3轮)...")
                    mj_result = evaluator.multi_judge_evaluate(dialog_result, rubric, num_judges=3)

                    # 保存最后一次评测结果用于报告
                    eval_result = mj_result.individual_results[-1]

                    # 违规定位
                    violation_locs = evaluator.locate_violations(dialog_result, rubric)

                    # ── 保存 ──
                    report_gen.save_report(
                        dialog_result, eval_result, rubric,
                        task_instruction, scenario.persona_name,
                    )

                    st.session_state.results.append({
                        "scenario": scenario,
                        "dialog": dialog_result,
                        "evaluation": eval_result,
                        "multi_judge": mj_result,
                        "violation_locations": violation_locs,
                    })

                    # 更新整体进度
                    pct = int(20 + (i + 1) / total * 70)
                    overall_progress.progress(pct, text=f"✅ {scenario.persona_name} 完成 ({i+1}/{total})")

                    # 显示本场景摘要
                    st.success(f"✅ {scenario.persona_name} → 总分: {eval_result.overall_score}/100")

                except Exception as e:
                    st.error(f"❌ {scenario.persona_name} 评测失败: {str(e)}")
                    continue  # 跳过失败的场景，继续下一个

            # ── 全部完成 ──
            overall_progress.progress(100, text="🎉 全部评测完成!")
            status_box.success(f"🎉 评测完成！共完成 {len(st.session_state.results)}/{total} 个场景")
            st.balloons()

        except Exception as e:
            st.error(f"❌ 评测系统出错: {str(e)}")
            st.code(traceback.format_exc(), language="python")

        st.session_state.running = False

    # ── 摘要卡片（评测完成后显示）──
    if st.session_state.results:
        st.divider()
        st.subheader("📊 评测总览")

        cols = st.columns(len(st.session_state.results))
        for idx, result in enumerate(st.session_state.results):
            with cols[idx]:
                s = result["scenario"]
                e = result["evaluation"]
                mj = result.get("multi_judge")
                score = mj.overall_mean if mj else e.overall_score
                score_std = mj.overall_std if mj else 0

                if score >= 80:
                    emoji = "🟢"
                elif score >= 60:
                    emoji = "🟡"
                else:
                    emoji = "🔴"

                label_text = f"{emoji} {s.persona_name}"
                if score_std > 0:
                    label_text += f" ±{score_std}"
                st.metric(label=label_text, value=f"{score}/100")

                for dim in DIMENSIONS:
                    key = dim["key"]
                    if key in e.dimensions:
                        ds = e.dimensions[key]
                        bar = "█" * ds.score + "░" * (5 - ds.score)
                        label = dim['name']
                        if mj and key in mj.dimension_stds and mj.dimension_stds[key] > 0:
                            label += f" ±{mj.dimension_stds[key]:.1f}"
                        st.caption(f"{label}: {bar} {ds.score}/5")


# ═══════════════════════════════════════════
# Tab 2: 上传评测（上传对话文件）
# ═══════════════════════════════════════════
with tab2:
    st.subheader("📤 上传对话记录进行评测")

    st.markdown("""
    比赛方提供的脱敏对话数据（JSON格式）可直接上传评测。
    **支持格式：**
    ```json
    {
      "task_instruction": "任务指令描述",
      "dialogs": [
        {
          "scenario_label": "场景名称（如配合型）",
          "turns": [
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "客服回复"}
          ]
        }
      ]
    }
    ```
    也可上传单个对话：
    ```json
    {
      "task_instruction": "...",
      "scenario_label": "...",
      "turns": [...]
    }
    ```
    """)

    uploaded_file = st.file_uploader("选择 JSON 文件", type=["json"])

    if uploaded_file is not None:
        try:
            data = json.loads(uploaded_file.read().decode("utf-8"))
            task_instruction = data.get("task_instruction", "")
            if not task_instruction:
                st.error("❌ JSON 文件中缺少 task_instruction 字段")
                st.stop()

            st.success(f"✅ 已读取文件，任务指令: {task_instruction[:80]}...")

            # 解析对话
            dialogs_data = []
            if "dialogs" in data:
                dialogs_data = data["dialogs"]
            elif "turns" in data:
                dialogs_data = [data]
            else:
                st.error("❌ JSON 文件中缺少 dialogs 或 turns 字段")
                st.stop()

            st.info(f"📊 共 {len(dialogs_data)} 个对话场景")

            if st.button("🚀 开始评测上传的对话", type="primary", use_container_width=True):
                if not settings.openai_api_key:
                    st.error("❌ 请先配置 API Key")
                    st.stop()

                progress = st.progress(0, text="开始评测...")
                status = st.empty()

                evaluator = Evaluator()
                scenario_builder = ScenarioBuilder()
                rubric = scenario_builder.parse_instruction(task_instruction)

                results = []
                for i, dialog_data in enumerate(dialogs_data):
                    status.info(f"▶️ [{i+1}/{len(dialogs_data)}] 评测中...")
                    progress.progress(int((i / len(dialogs_data)) * 90), text=f"评测第 {i+1} 个对话...")

                    scenario_label = dialog_data.get("scenario_label", f"场景{i+1}")
                    turns_data = dialog_data.get("turns", [])

                    # 构造 DialogResult
                    scenario = type("Scenario", (), {
                        "persona_name": scenario_label,
                        "persona_id": f"upload_{i}",
                        "test_goal": "上传对话评测",
                        "to_dict": lambda self: {"persona_name": self.persona_name, "persona_id": self.persona_id},
                    })()

                    dialog_turns = []
                    history = []
                    for j, t in enumerate(turns_data):
                        role = t.get("role", "user")
                        content = t.get("content", "")
                        if role == "user":
                            dialog_turns.append(Turn(user=content, assistant="", turn_number=j // 2 + 1))
                        else:
                            if dialog_turns:
                                dialog_turns[-1].assistant = content
                        history.append({"role": role, "content": content})

                    dialog_result = DialogResult(scenario=scenario, turns=dialog_turns, finished_reason="uploaded")

                    # 多评委评测
                    mj_result = evaluator.multi_judge_evaluate(dialog_result, rubric, num_judges=3)
                    eval_result = mj_result.individual_results[-1]

                    results.append({
                        "scenario": scenario,
                        "dialog": dialog_result,
                        "evaluation": eval_result,
                        "multi_judge": mj_result,
                    })

                    st.success(f"✅ {scenario_label} — {mj_result.overall_mean}/100 (±{mj_result.overall_std})")

                progress.progress(100, text="✅ 全部评测完成!")
                status.success(f"🎉 完成！共评测 {len(results)} 个对话")

                st.session_state.results = results
                st.balloons()

        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 解析失败: {e}")
        except Exception as e:
            st.error(f"❌ 处理出错: {e}")
            st.code(traceback.format_exc(), language="python")


# ═══════════════════════════════════════════
# Tab 3: 评测结果（详细报告）
# ═══════════════════════════════════════════
with tab3:
    if not st.session_state.results:
        st.info("💡 请先在「模拟评测」或「上传评测」页面运行一次评测")
    else:
        for idx, result in enumerate(st.session_state.results):
            s = result["scenario"]
            e = result["evaluation"]
            d = result["dialog"]

            mj = result.get("multi_judge")
            mj_title = f"📊 {s.persona_name}"
            if mj:
                mj_title += f" — 总分: {mj.overall_mean}/100"
                if mj.overall_std > 0:
                    mj_title += f" (±{mj.overall_std})"
            else:
                mj_title += f" — 总分: {e.overall_score}/100"

            with st.expander(mj_title, expanded=(idx == 0)):
                # 多评委一致性
                if mj and mj.overall_std > 0:
                    st.info(f"🤖 **多评委一致性**: 3次评分标准差 σ={mj.overall_std} (越小越一致)")

                col1, col2 = st.columns([2, 1])

                with col1:
                    # ── 聊天气泡风格对话记录 ──
                    st.markdown("#### 💬 对话记录")
                    for turn in d.turns:
                        st.markdown(f"""
                        <div style="margin-bottom:14px">
                            <div style="display:flex; margin-bottom:4px">
                                <div style="background:#e8f4fd; border-radius:12px 12px 12px 2px; padding:10px 16px; max-width:85%; color:#1a1a1a; font-size:15px; line-height:1.5">
                                    <b style="color:#1976d2">🧑 用户</b><br>{html.escape(turn.user)}
                                </div>
                            </div>
                            <div style="display:flex; justify-content:flex-end; margin-bottom:4px">
                                <div style="background:#e8fde8; border-radius:12px 12px 2px 12px; padding:10px 16px; max-width:85%; color:#1a1a1a; font-size:15px; line-height:1.5">
                                    <b style="color:#2e7d32">🤖 客服</b><br>{html.escape(turn.assistant)}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if turn.turn_number < len(d.turns):
                            st.markdown("---")

                with col2:
                    # ── 评分明细 ──
                    st.markdown("#### 📈 各维度评分")
                    for dim in DIMENSIONS:
                        key = dim["key"]
                        if key in e.dimensions:
                            ds = e.dimensions[key]
                            st.markdown(f"**{dim['name']}** ({dim['weight']*100:.0f}%)")
                            st.progress(ds.score / 5, text=f"{ds.score}/5")
                            st.caption(f"💬 {ds.reason}")

                    if e.violations:
                        st.markdown("#### ⚠️ 违规项")
                        for v in e.violations:
                            st.warning(v)

                    if e.good_points:
                        st.markdown("#### ✅ 亮点")
                        for p in e.good_points:
                            st.success(p)

                    st.markdown("#### 📝 总结")
                    st.info(e.summary)


# ═══════════════════════════════════════════
# Tab 4: 使用说明
# ═══════════════════════════════════════════
with tab4:
    st.markdown("""
    ## 📖 使用说明

    ### 快速开始
    1. **配置 API Key**：在侧边栏输入你的 API Key
    2. **输入任务指令**：描述数字人外呼助手的任务
    3. **选择用户画像**：选择要模拟的用户类型
    4. **点击开始评测**：系统自动解析指令 → 模拟对话 → 多维度评测 → 输出报告

    ### 用户画像说明

    | 画像类型 | 测试目的 |
    |---------|---------|
    | 🟢 普通配合型 | 测试基本流程完成能力 |
    | 🟡 拒绝型 | 测试模型的挽回策略 |
    | 🔵 追问型 | 测试模型的合规回答能力 |
    | 🟣 干扰型 | 测试模型拉回主题能力 |
    | 🔴 对抗型 | 测试禁止行为遵守情况 |
    | ⚪ 模糊型 | 测试意图确认能力 |

    ### 评测维度

    | 维度 | 权重 | 说明 |
    |------|------|------|
    | 任务完成度 | 25% | 是否完成核心任务 |
    | 指令遵循度 | 25% | 是否按要求流程执行 |
    | 约束遵守度 | 20% | 是否违反禁止行为 |
    | 多轮一致性 | 10% | 前后回答是否矛盾 |
    | 用户意图识别 | 10% | 是否理解用户状态 |
    | 对话自然度 | 5% | 是否像真实客服 |
    | 安全合规性 | 5% | 是否隐私/合规问题 |
    """)


# ── 页脚 ──
st.divider()
st.caption("EvalAgent v0.1 · 美团 AI Hackathon 赛道 02 · 多轮对话自动评测系统 · DeepSeek V4")
