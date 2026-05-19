"""EvalAgent Streamlit 前端——输入指令、运行评测、查看报告"""

import json
import time

import streamlit as st

from config import settings
from core.scenario_builder import ScenarioBuilder
from core.dialogue_runner import DialogueRunner
from core.evaluator import Evaluator, DIMENSIONS
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


# ── 侧边栏：配置 ──
with st.sidebar:
    st.header("⚙️ 配置")

    # LLM API Key（可覆盖 .env）
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=settings.openai_api_key or "",
        help="留空则使用 .env 中的配置",
    )
    if api_key:
        settings.openai_api_key = api_key

    model_name = st.text_input(
        "模型名称",
        value=settings.openai_model_name,
        help="如 gpt-4o, qwen-max",
    )
    if model_name:
        settings.openai_model_name = model_name

    base_url = st.text_input(
        "API Base URL",
        value=settings.openai_api_base,
        help="DashScope 用户请使用 https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if base_url:
        settings.openai_api_base = base_url

    st.divider()
    st.caption("💡 提示：首次使用请在 `.env` 文件中配置 API Key，或在上方输入")


# ── 主界面 ──

# 选项卡
tab1, tab2, tab3 = st.tabs(["🧪 评测运行", "📊 评测结果", "📖 使用说明"])

# ── Tab 1: 评测运行 ──
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        # 任务指令输入
        default_instruction = "向用户介绍优惠活动，确认用户是否有意向领取优惠券，遇到拒绝要挽回一次，不能透露内部价格策略，最终要完成意向收集。"
        task_instruction = st.text_area(
            "📝 输入任务指令",
            value=default_instruction,
            height=120,
            help="输入数字人外呼助手的任务指令，系统将据此生成测试场景并自动评测",
            placeholder="例：向用户介绍活动，确认意向，遇到拒绝挽回一次…",
        )

    with col2:
        # 场景选择
        st.markdown("##### 🎭 选择用户画像")
        scenario_builder = ScenarioBuilder()
        all_scenarios = scenario_builder.load_scenarios()

        selected_personas = []
        for s in all_scenarios:
            if st.checkbox(s.persona_name, value=s.persona_id in ["cooperative", "rejecting", "inquiring"]):
                selected_personas.append(s.persona_id)

        max_turns = st.slider("最大对话轮次", 3, 12, 8)

    # 运行按钮
    run_btn = st.button(
        "🚀 开始评测",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running or not task_instruction.strip(),
    )

    if run_btn:
        if not settings.openai_api_key:
            st.error("❌ 请在侧边栏或 `.env` 中配置 API Key")
            st.stop()

        st.session_state.running = True
        st.session_state.results = []

        progress_bar = st.progress(0, text="初始化...")
        status_placeholder = st.empty()

        try:
            # Step 1: 解析任务指令
            status_placeholder.info("📋 正在解析任务指令...")
            progress_bar.progress(5, text="解析任务指令中...")
            rubric = scenario_builder.parse_instruction(task_instruction)

            with st.expander("📋 指令解析结果", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**任务目标**: {rubric.task_goal}")
                    st.markdown(f"**必须做**: {', '.join(rubric.must_do)}")
                with col_b:
                    st.markdown(f"**禁止做**: {', '.join(rubric.must_not_do)}")
                    st.markdown(f"**约束**: {rubric.constraints}")

            # Step 2: 加载场景
            selected_scenarios = [s for s in all_scenarios if s.persona_id in selected_personas]
            total = len(selected_scenarios)
            status_placeholder.info(f"🎭 已选择 {total} 个用户场景，开始评测...")

            # Step 3: 逐场景运行对话 + 评测
            dialogue_runner = DialogueRunner()
            evaluator = Evaluator()
            report_gen = ReportGenerator()

            for i, scenario in enumerate(selected_scenarios):
                status_placeholder.info(f"▶️ [{i+1}/{total}] 正在运行: {scenario.persona_name}")
                progress_bar.progress(
                    int(10 + (i / total) * 80),
                    text=f"正在运行 {scenario.persona_name}...",
                )

                # 运行对话
                dialog_result = dialogue_runner.run_dialog(
                    scenario=scenario,
                    rubric=rubric,
                    max_turns=max_turns,
                )

                # 评测
                eval_result = evaluator.evaluate(dialog_result, rubric)

                # 保存
                report_gen.save_report(
                    dialog_result, eval_result, rubric,
                    task_instruction, scenario.persona_name,
                )

                st.session_state.results.append({
                    "scenario": scenario,
                    "dialog": dialog_result,
                    "evaluation": eval_result,
                })

            # Step 4: 完成
            progress_bar.progress(100, text="✅ 评测完成!")
            status_placeholder.success(f"✅ 评测完成！共评测 {total} 个场景")

            # 显示摘要
            st.balloons()

        except Exception as e:
            st.error(f"❌ 评测出错: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

        st.session_state.running = False

    # 显示当前结果
    if st.session_state.results:
        st.divider()
        st.subheader("📊 本次评测摘要")

        cols = st.columns(len(st.session_state.results))
        for idx, result in enumerate(st.session_state.results):
            with cols[idx]:
                s = result["scenario"]
                e = result["evaluation"]
                score = e.overall_score
                color = "green" if score >= 80 else ("orange" if score >= 60 else "red")
                st.metric(
                    label=f"{s.persona_name}",
                    value=f"{score}/100",
                    delta_color="normal",
                )

                # 维度迷你条形图
                for dim in DIMENSIONS:
                    key = dim["key"]
                    if key in e.dimensions:
                        ds = e.dimensions[key]
                        bar = "█" * ds.score + "░" * (5 - ds.score)
                        st.caption(f"{dim['name']}: {bar} {ds.score}/5")


# ── Tab 2: 评测结果 ──
with tab2:
    if not st.session_state.results:
        st.info("💡 请先在"评测运行"页面运行一次评测")
    else:
        for idx, result in enumerate(st.session_state.results):
            s = result["scenario"]
            e = result["evaluation"]
            d = result["dialog"]

            with st.expander(f"📊 {s.persona_name} — 总分: {e.overall_score}/100", expanded=(idx == 0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # 对话记录
                    st.markdown("#### 💬 对话记录")
                    for turn in d.turns:
                        with st.container():
                            st.markdown(f"**第 {turn.turn_number} 轮**")
                            st.markdown(f"> 🧑 **用户**: {turn.user}")
                            st.markdown(f"> 🤖 **客服**: {turn.assistant}")
                            if turn.turn_number < len(d.turns):
                                st.divider()

                with col2:
                    # 评分明细
                    st.markdown("#### 📈 各维度评分")
                    for dim in DIMENSIONS:
                        key = dim["key"]
                        if key in e.dimensions:
                            ds = e.dimensions[key]
                            st.markdown(f"**{dim['name']}** ({dim['weight']*100:.0f}%)")
                            st.progress(ds.score / 5, text=f"{ds.score}/5")
                            st.caption(ds.reason)

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


# ── Tab 3: 使用说明 ──
with tab3:
    st.markdown("""
    ## 📖 使用说明

    ### 快速开始
    1. **配置 API Key**：在侧边栏输入你的 OpenAI API Key（或通义千问 DashScope Key）
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

    ### 部署选项

    ```bash
    # Streamlit 运行
    streamlit run app.py

    # FastAPI 运行（可选）
    uvicorn main:app --reload
    ```
    """)


# ── 页脚 ──
st.divider()
st.caption("EvalAgent v0.1 · 美团 AI Hackathon 赛道 02 · 多轮对话自动评测系统")
