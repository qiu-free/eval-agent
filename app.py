"""EvalAgent Streamlit 前端——输入指令、运行评测、查看报告"""

import html
import json
import traceback
import re
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
    initial_sidebar_state="expanded",
)

# ── CSS 样式 ──
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .main-header p { color: #666; font-size: 1rem; margin-top: 0.2rem; }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 20px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .score-card .score-value { font-size: 3rem; font-weight: 800; line-height: 1; }
    .score-card .score-label { font-size: 0.9rem; opacity: 0.9; }
    .score-card .score-std { font-size: 1rem; opacity: 0.7; }
    .dim-card {
        background: white; border-radius: 12px; padding: 12px 16px;
        border: 1px solid #eee; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .dim-card .dim-name { font-weight: 600; font-size: 0.85rem; color: #333; }
    .dim-card .dim-score { font-size: 1.2rem; font-weight: 700; }
    .dim-card .dim-reason { font-size: 0.75rem; color: #888; margin-top: 4px; }
    .chat-bubble-user {
        background: #e8f4fd; border-radius: 16px 16px 16px 4px;
        padding: 10px 16px; margin-bottom: 8px; max-width: 85%;
        border-left: 3px solid #1976d2;
    }
    .chat-bubble-assistant {
        background: #e8fde8; border-radius: 16px 16px 4px 16px;
        padding: 10px 16px; margin-bottom: 8px; max-width: 85%; margin-left: auto;
        border-right: 3px solid #2e7d32;
    }
    .chat-label { font-size: 0.75rem; font-weight: 600; margin-bottom: 2px; }
    .chat-text { font-size: 0.95rem; line-height: 1.5; color: #1a1a1a; }
    .stProgress > div > div > div > div { background-image: linear-gradient(90deg, #667eea, #764ba2); }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 4px;
    }
    .badge-green { background: #e8fde8; color: #2e7d32; }
    .badge-red { background: #fde8e8; color: #c62828; }
    .badge-yellow { background: #fff8e1; color: #f57f17; }
    .badge-blue { background: #e3f2fd; color: #1565c0; }
    .suggestion-card {
        background: #fff8e1; border-radius: 12px; padding: 16px;
        border-left: 4px solid #ffa000; margin: 8px 0;
    }
    .innovation-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ── 页面标题 ──
st.markdown("""
<div class="main-header">
    <h1>🎯 EvalAgent</h1>
    <p>多轮对话自动评测系统 · 美团 AI Hackathon 赛道 02</p>
</div>
""", unsafe_allow_html=True)

# ── 初始化 Session State ──
if "results" not in st.session_state:
    st.session_state.results = []
if "running" not in st.session_state:
    st.session_state.running = False
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "generated_scenarios" not in st.session_state:
    st.session_state.generated_scenarios = None

# ── 工具函数 ──
def render_radar_chart(eval_result, title="评分维度"):
    """绘制7维雷达图"""
    dims = []
    scores = []
    for dim in DIMENSIONS:
        key = dim["key"]
        if key in eval_result.dimensions:
            dims.append(dim["name"][:4])  # 短名称
            scores.append(eval_result.dimensions[key].score)

    fig = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],
        theta=dims + [dims[0]],
        fill='toself',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea'),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont_size=10),
            angularaxis=dict(tickfont_size=12),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
    )
    return fig

def render_score_card(score, label="总分", std=None, size="normal"):
    """渲染评分卡片"""
    if score >= 80:
        emoji, color = "🟢", "linear-gradient(135deg, #43a047, #66bb6a)"
    elif score >= 60:
        emoji, color = "🟡", "linear-gradient(135deg, #ef6c00, #ffa726)"
    else:
        emoji, color = "🔴", "linear-gradient(135deg, #c62828, #ef5350)"

    font_size = "2.5rem" if size == "normal" else "1.8rem"
    return f"""
    <div style="background:{color}; border-radius:14px; padding:16px; color:white; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
        <div style="font-size:{font_size}; font-weight:800;">{emoji} {score}</div>
        <div style="font-size:0.85rem; opacity:0.9;">{label}</div>
        {f'<div style="font-size:0.75rem; opacity:0.7;">一致性 σ={std}</div>' if std else ''}
    </div>
    """

def render_suggestions(eval_result, scenario_name):
    """生成改进建议"""
    suggestions = []
    for dim in DIMENSIONS:
        key = dim["key"]
        if key in eval_result.dimensions:
            ds = eval_result.dimensions[key]
            if ds.score <= 2:
                suggestions.append(f"🔴 **{dim['name']}**({ds.score}/5): {ds.reason}")
            elif ds.score <= 3:
                suggestions.append(f"🟡 **{dim['name']}**({ds.score}/5): {ds.reason}")
    return suggestions

def generate_test_scenarios(task_instruction):
    """AI自动生成测试场景"""
    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )
    prompt = f"""基于以下任务指令，生成 4 个测试场景的用户画像和行为模式。

任务指令：{task_instruction}

请生成 JSON 格式（不要其他内容）：
{{
  "scenarios": [
    {{
      "name": "场景名称",
      "description": "用户行为描述",
      "test_goal": "测试目的",
      "behavior": ["行为1", "行为2", "行为3"]
    }}
  ]
}}

要求场景多样化、有针对性，覆盖成功路径和边缘情况。"""
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception as e:
        return {"scenarios": []}


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("### ⚙️ 配置")
    api_key = st.text_input("API Key", type="password",
                            value=settings.openai_api_key or "",
                            help="留空使用 .env 配置")
    if api_key:
        settings.openai_api_key = api_key

    settings.openai_model_name = st.text_input("模型", value=settings.openai_model_name,
                                                help="deepseek-v4-flash / gpt-4o")

    settings.openai_api_base = st.text_input("API Base URL", value=settings.openai_api_base,
                                              help="https://api.deepseek.com")

    st.divider()
    st.markdown("### 📊 评测历史")
    if st.session_state.eval_history:
        for h in st.session_state.eval_history[-5:]:
            st.caption(f"🕐 {h['time']} — {h['count']}个场景")
        if st.button("🗑️ 清空历史"):
            st.session_state.eval_history = []
            st.session_state.results = []
            st.rerun()
    else:
        st.caption("暂无历史记录")

    st.divider()
    st.markdown(f"""<div class="innovation-badge">🏆 创新功能</div>""", unsafe_allow_html=True)
    st.caption("• 多评委一致性评分 σ")
    st.caption("• 违规定位到具体轮次")
    st.caption("• AI 自动生成测试场景")
    st.caption("• 改进建议引擎")
    st.caption("• 7维雷达图可视化")


# ═══════════════════════════════════════════
# Tab 1: 模拟评测
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 模拟评测",
    "📤 上传评测",
    "📊 评测结果",
    "📖 使用说明",
])

with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        default_instruction = "向用户介绍优惠活动，确认用户是否有意向领取优惠券，遇到拒绝要挽回一次，不能透露内部价格策略，最终要完成意向收集。"
        task_instruction = st.text_area(
            "📝 任务指令", value=default_instruction, height=100,
        )

    with col_right:
        st.markdown("##### 🎭 用户画像")
        scenario_builder = ScenarioBuilder()
        all_scenarios = scenario_builder.load_scenarios()

        selected_personas = []
        for s in all_scenarios:
            default_on = s.persona_id in ["cooperative", "rejecting", "inquiring"]
            if st.checkbox(s.persona_name, value=default_on, key=f"cb_{s.persona_id}"):
                selected_personas.append(s.persona_id)

        max_turns = st.slider("最大轮次", 3, 12, 8)

        # ＃ 自动生成场景按钮
        if st.button("🤖 自动生成测试场景", use_container_width=True, type="secondary"):
            if not settings.openai_api_key:
                st.error("❌ 请先配置 API Key")
            elif not task_instruction.strip():
                st.error("❌ 请输入任务指令")
            else:
                with st.spinner("AI 正在生成测试场景..."):
                    result = generate_test_scenarios(task_instruction)
                    if result.get("scenarios"):
                        st.session_state.generated_scenarios = result["scenarios"]
                        st.success(f"✅ 已生成 {len(result['scenarios'])} 个场景")
                        st.rerun()
                    else:
                        st.error("❌ 生成失败，请重试")

    # 显示自动生成的场景
    if st.session_state.generated_scenarios:
        st.markdown("#### 🤖 AI 生成的测试场景")
        cols = st.columns(len(st.session_state.generated_scenarios))
        for i, sc in enumerate(st.session_state.generated_scenarios):
            with cols[i]:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#667eea15,#764ba215); border-radius:12px; padding:12px; border:1px solid #667eea30;">
                    <div style="font-weight:700; color:#667eea;">{sc.get('name','')}</div>
                    <div style="font-size:0.8rem; color:#666; margin:6px 0;">{sc.get('description','')}</div>
                    <div style="font-size:0.75rem;"><b>测试目标:</b> {sc.get('test_goal','')}</div>
                </div>
                """, unsafe_allow_html=True)
        if st.button("🗑️ 清除生成场景", use_container_width=True):
            st.session_state.generated_scenarios = None
            st.rerun()

    # 运行按钮
    run_btn = st.button(
        "🚀 开始评测", type="primary", use_container_width=True,
        disabled=st.session_state.running or not task_instruction.strip(),
    )

    if run_btn:
        if not settings.openai_api_key:
            st.error("❌ 请配置 API Key")
            st.stop()

        st.session_state.running = True
        st.session_state.results = []

        overall_progress = st.progress(0, text="初始化...")
        status_box = st.empty()
        live_area = st.empty()

        try:
            overall_progress.progress(5, text="解析任务指令...")
            rubric = scenario_builder.parse_instruction(task_instruction)

            with st.expander("📋 指令解析", expanded=True):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**目标**: {rubric.task_goal}")
                    st.markdown(f"**必须做**: {'、'.join(rubric.must_do)}")
                with cb:
                    st.markdown(f"**禁止做**: {'、'.join(rubric.must_not_do)}")
                    st.markdown(f"**约束**: 最多 {rubric.constraints.get('max_turns', 'N/A')} 轮")

            selected_scenarios = [s for s in all_scenarios if s.persona_id in selected_personas]
            total = len(selected_scenarios)

            if total == 0:
                st.warning("⚠️ 请至少选一个画像")
                st.session_state.running = False
                st.rerun()

            status_box.info(f"🎭 {total} 个场景开始评测...")
            dialogue_runner = DialogueRunner()
            evaluator = Evaluator()
            report_gen = ReportGenerator()

            for i, scenario in enumerate(selected_scenarios):
                status_box.info(f"▶️ [{i+1}/{total}] {scenario.persona_name} — 对话中...")

                try:
                    dialog_result = dialogue_runner.run_dialog(
                        scenario=scenario, rubric=rubric, max_turns=max_turns,
                    )

                    with live_area.container():
                        st.markdown(f"### 💬 {scenario.persona_name}")
                        for turn in dialog_result.turns:
                            st.markdown(f"""
                            <div class="chat-bubble-user">
                                <div class="chat-label" style="color:#1976d2;">🧑 用户</div>
                                <div class="chat-text">{html.escape(turn.user)}</div>
                            </div>
                            <div class="chat-bubble-assistant">
                                <div class="chat-label" style="color:#2e7d32; text-align:right;">🤖 客服</div>
                                <div class="chat-text">{html.escape(turn.assistant)}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    status_box.info(f"▶️ [{i+1}/{total}] {scenario.persona_name} — 多评委评测...")
                    mj_result = evaluator.multi_judge_evaluate(dialog_result, rubric, num_judges=3)
                    eval_result = mj_result.individual_results[-1]
                    violation_locs = evaluator.locate_violations(dialog_result, rubric)

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

                    pct = int(20 + (i + 1) / total * 70)
                    overall_progress.progress(pct, text=f"✅ {scenario.persona_name} ({i+1}/{total})")

                    st.markdown(render_score_card(
                        mj_result.overall_mean, scenario.persona_name, mj_result.overall_std
                    ), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ {scenario.persona_name} 失败: {e}")
                    continue

            overall_progress.progress(100, text="🎉 全部完成!")
            status_box.success(f"🎉 完成 {len(st.session_state.results)}/{total} 个场景")

            # 保存到历史
            st.session_state.eval_history.append({
                "time": datetime.now().strftime("%H:%M"),
                "count": len(st.session_state.results),
                "results": st.session_state.results.copy(),
            })

            st.balloons()

        except Exception as e:
            st.error(f"❌ 出错: {e}")
            st.code(traceback.format_exc(), language="python")

        st.session_state.running = False

    # 结果摘要
    if st.session_state.results:
        st.divider()
        st.markdown("### 📊 评测总览")
        cols = st.columns(len(st.session_state.results))
        for idx, result in enumerate(st.session_state.results):
            with cols[idx]:
                s = result["scenario"]
                mj = result.get("multi_judge")
                score = mj.overall_mean if mj else result["evaluation"].overall_score
                score_std = mj.overall_std if mj else 0
                st.markdown(render_score_card(score, s.persona_name, score_std, "small"), unsafe_allow_html=True)

        # 场景对比雷达图
        if len(st.session_state.results) > 1:
            st.markdown("#### 📈 场景对比")
            fig = go.Figure()
            colors = ['#667eea', '#43a047', '#ef6c00', '#c62828', '#8e24aa', '#00acc1']
            for idx, result in enumerate(st.session_state.results):
                s = result["scenario"]
                e = result["evaluation"]
                dims = [dim["name"][:4] for dim in DIMENSIONS if dim["key"] in e.dimensions]
                scores = [e.dimensions[dim["key"]].score for dim in DIMENSIONS if dim["key"] in e.dimensions]
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],
                    theta=dims + [dims[0]],
                    name=s.persona_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                legend=dict(orientation="h", y=-0.1),
                height=350, margin=dict(l=40, r=40, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# Tab 2: 上传评测
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 📤 上传对话文件进行评测")
    st.caption("支持 CSV / JSON / JSONL 格式，自动适配多种字段命名")

    uploaded_file = st.file_uploader("选择文件", type=["json", "csv", "jsonl"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "json"

        try:
            raw_text = uploaded_file.read().decode("utf-8-sig")

            # ── 字段名归一化 ──
            def normalize_turn(t: dict) -> tuple:
                role_keys = ["role", "speaker", "from", "sender", "说话人"]
                content_keys = ["content", "text", "message", "msg", "utterance", "话", "text_content"]
                role, content = "", ""
                for k in role_keys:
                    if k in t and t[k]:
                        v = str(t[k]).strip().lower()
                        if v in ("user", "用户", "u", "human", "顾客", "客户"): role = "user"
                        elif v in ("assistant", "客服", "a", "bot", "agent", "model"): role = "assistant"
                        break
                for k in content_keys:
                    if k in t and t[k]:
                        content = str(t[k]).strip()
                        break
                return role, content

            def parse_turns(item) -> list[dict]:
                if isinstance(item, list):
                    if all(isinstance(x, str) for x in item):
                        return [{"role": "user" if i % 2 == 0 else "assistant", "content": x} for i, x in enumerate(item)]
                    return [{"role": r, "content": c} for r, c in [normalize_turn(t) for t in item if isinstance(t, dict)] if r and c]
                if isinstance(item, dict):
                    for key in ["turns", "messages", "dialog", "conversation", "history", "对话", "对话记录"]:
                        if key in item: return parse_turns(item[key])
                    r, c = normalize_turn(item)
                    return [{"role": r, "content": c}] if r and c else []
                return []

            def extract_task_instruction(data) -> str:
                for key in ["task_instruction", "instruction", "task", "prompt", "system_prompt", "任务", "任务指令", "指令"]:
                    if key in data and isinstance(data[key], str) and len(data[key]) > 10:
                        return data[key]
                return ""

            def extract_label(item, idx):
                if isinstance(item, dict):
                    for key in ["scenario_label", "scenario", "label", "name", "场景", "type"]:
                        if key in item and item[key]: return str(item[key])
                return f"场景 {idx+1}"

            # 解析
            dialogs_data = []
            task_instruction = ""

            if file_ext == "csv":
                import csv, io
                rows = list(csv.DictReader(io.StringIO(raw_text)))
                if rows:
                    dk = next((k for k in ["dialog_id","dialog","session","会话","id","conversation_id"] if k in rows[0]), None)
                    tk = next((k for k in ["task_instruction","instruction","task","任务","任务指令"] if k in rows[0]), None)
                    if tk: task_instruction = rows[0].get(tk, "")
                    if dk:
                        from itertools import groupby
                        for gid, group in groupby(rows, lambda r: r.get(dk, "")):
                            turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in list(group)] if r and c]
                            if turns: dialogs_data.append({"scenario_label": gid or "对话", "turns": turns})
                    else:
                        turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in rows] if r and c]
                        if turns: dialogs_data.append({"scenario_label": "对话", "turns": turns})

            elif file_ext == "jsonl":
                for idx, line in enumerate([l for l in raw_text.split("\n") if l.strip()]):
                    item = json.loads(line)
                    turns = parse_turns(item)
                    if turns:
                        ti = extract_task_instruction(item)
                        if ti and not task_instruction: task_instruction = ti
                        dialogs_data.append({"scenario_label": extract_label(item, idx), "turns": turns})

            else:
                data = json.loads(raw_text)
                task_instruction = extract_task_instruction(data)
                if "dialogs" in data:
                    for item in data["dialogs"]:
                        turns = parse_turns(item)
                        if turns: dialogs_data.append({"scenario_label": extract_label(item, len(dialogs_data)), "turns": turns})
                elif isinstance(data, list):
                    for item in data:
                        turns = parse_turns(item)
                        if turns: dialogs_data.append({"scenario_label": extract_label(item, len(dialogs_data)), "turns": turns})
                else:
                    turns = parse_turns(data)
                    if turns: dialogs_data.append({"scenario_label": extract_label(data, 0), "turns": turns})

            if not dialogs_data:
                st.error("❌ 未能解析出有效对话")
                st.stop()

            if not task_instruction:
                task_instruction = st.text_area("📝 输入任务指令", value="向用户介绍优惠活动...", height=60)
                if not task_instruction.strip(): st.stop()

            st.success(f"✅ 解析出 {len(dialogs_data)} 个对话")
            st.info(f"📋 任务: {task_instruction[:100]}...")

            with st.expander("🔍 预览", expanded=False):
                for i, dd in enumerate(dialogs_data):
                    st.markdown(f"**{dd['scenario_label']}** ({len(dd['turns'])}轮)")
                    for t in dd['turns'][:4]:
                        st.text(f"  {'🧑' if t['role']=='user' else '🤖'} {t['content'][:80]}")
                    if len(dd['turns']) > 4: st.text(f"  ... +{len(dd['turns'])-4}轮")

            if st.button("🚀 评测上传的对话", type="primary", use_container_width=True):
                if not settings.openai_api_key:
                    st.error("❌ 请先配置 API Key")
                    st.stop()

                progress = st.progress(0, text="评测中...")
                status = st.empty()
                evaluator = Evaluator()
                sb = ScenarioBuilder()
                rubric = sb.parse_instruction(task_instruction)
                results = []

                for i, dd in enumerate(dialogs_data):
                    status.info(f"[{i+1}/{len(dialogs_data)}] {dd['scenario_label']}")
                    progress.progress(int(i/len(dialogs_data)*90), text=f"评测 {i+1}...")

                    scenario = type("S", (), {
                        "persona_name": dd['scenario_label'], "persona_id": f"u{i}",
                        "test_goal": "上传评测", "to_dict": lambda s: {"persona_name": s.persona_name},
                    })()
                    turns = []
                    for j, t in enumerate(dd['turns']):
                        if t['role'] == 'user':
                            turns.append(Turn(user=t['content'], assistant="", turn_number=j//2+1))
                        elif turns:
                            turns[-1].assistant = t['content']

                    dr = DialogResult(scenario=scenario, turns=turns, finished_reason="uploaded")
                    mj = evaluator.multi_judge_evaluate(dr, rubric, num_judges=3)
                    results.append({"scenario": scenario, "dialog": dr, "evaluation": mj.individual_results[-1], "multi_judge": mj})
                    st.success(f"✅ {dd['scenario_label']}: {mj.overall_mean}/100")

                progress.progress(100, text="完成!")
                st.session_state.results = results
                st.session_state.eval_history.append({
                    "time": datetime.now().strftime("%H:%M"), "count": len(results), "results": results.copy(),
                })
                st.balloons()

        except Exception as e:
            st.error(f"❌ 处理出错: {e}")
            st.code(traceback.format_exc(), language="python")


# ═══════════════════════════════════════════
# Tab 3: 评测结果
# ═══════════════════════════════════════════
with tab3:
    if not st.session_state.results:
        st.info("💡 先在「模拟评测」或「上传评测」运行一次")
    else:
        # 总体仪表盘
        st.markdown("### 📊 综合仪表盘")
        cols = st.columns(len(st.session_state.results))
        for idx, result in enumerate(st.session_state.results):
            with cols[idx]:
                s = result["scenario"]
                mj = result.get("multi_judge")
                e = result["evaluation"]
                score = mj.overall_mean if mj else e.overall_score
                std = mj.overall_std if mj else 0
                st.markdown(render_score_card(score, s.persona_name, std), unsafe_allow_html=True)

                # 改进建议
                suggestions = render_suggestions(e, s.persona_name)
                if suggestions:
                    with st.expander("💡 改进建议", expanded=False):
                        for sug in suggestions:
                            st.markdown(f"<div class='suggestion-card'>{sug}</div>", unsafe_allow_html=True)

        # 雷达图对比
        if len(st.session_state.results) >= 1:
            st.markdown("#### 📈 评分维度雷达图")
            fig = go.Figure()
            colors = ['#667eea', '#43a047', '#ef6c00', '#c62828', '#8e24aa', '#00acc1']
            for idx, result in enumerate(st.session_state.results):
                s = result["scenario"]
                e = result["evaluation"]
                dims = [dim["name"][:4] for dim in DIMENSIONS if dim["key"] in e.dimensions]
                scores = [e.dimensions[dim["key"]].score for dim in DIMENSIONS if dim["key"] in e.dimensions]
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]], theta=dims + [dims[0]],
                    name=s.persona_name, fill='toself',
                    line=dict(color=colors[idx % len(colors)], width=2),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                legend=dict(orientation="h", y=-0.15),
                height=400, margin=dict(l=60, r=60, t=10, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

        # 详细结果
        for idx, result in enumerate(st.session_state.results):
            s = result["scenario"]
            e = result["evaluation"]
            d = result["dialog"]
            mj = result.get("multi_judge")
            vl = result.get("violation_locations", {})

            title = f"📊 {s.persona_name}"
            if mj:
                title += f" — {mj.overall_mean}/100" + (f" (±{mj.overall_std})" if mj.overall_std > 0 else "")
            else:
                title += f" — {e.overall_score}/100"

            with st.expander(title, expanded=(idx == 0)):
                if mj and mj.overall_std > 0:
                    st.info(f"🤖 **一致性** σ={mj.overall_std} (越小越一致，3次评分)")

                # 改进建议
                suggestions = render_suggestions(e, s.persona_name)
                if suggestions:
                    st.markdown("#### 💡 改进建议")
                    for sug in suggestions:
                        st.markdown(f"<div class='suggestion-card'>{sug}</div>", unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("#### 💬 对话记录")
                    for turn in d.turns:
                        st.markdown(f"""
                        <div class="chat-bubble-user">
                            <div class="chat-label" style="color:#1976d2;">🧑 用户</div>
                            <div class="chat-text">{html.escape(turn.user)}</div>
                        </div>
                        <div class="chat-bubble-assistant">
                            <div class="chat-label" style="color:#2e7d32; text-align:right;">🤖 客服</div>
                            <div class="chat-text">{html.escape(turn.assistant)}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 📈 维度评分")
                    fig = render_radar_chart(e)
                    st.plotly_chart(fig, use_container_width=True)

                    for dim in DIMENSIONS:
                        key = dim["key"]
                        if key in e.dimensions:
                            ds = e.dimensions[key]
                            pct = ds.score / 5
                            if pct >= 0.8: color = "#43a047"
                            elif pct >= 0.6: color = "#ef6c00"
                            else: color = "#c62828"
                            st.markdown(f"""
                            <div class="dim-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span class="dim-name">{dim['name']}</span>
                                    <span class="dim-score" style="color:{color}">{ds.score}/5</span>
                                </div>
                                <div style="background:#eee; border-radius:8px; height:6px; margin:6px 0;">
                                    <div style="background:{color}; width:{pct*100}%; height:6px; border-radius:8px;"></div>
                                </div>
                                <div class="dim-reason">{ds.reason}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    if vl:
                        st.markdown("#### ⚠️ 违规定位")
                        for violation, locs in vl.items():
                            with st.expander(f"⚠️ {violation}", expanded=False):
                                for turn_num, role, text in locs:
                                    st.caption(f"第 {turn_num} 轮 ({role}): {text}")

                    if e.violations:
                        for v in e.violations:
                            st.warning(v)

                    if e.good_points:
                        for p in e.good_points:
                            st.success(f"✅ {p}")

                    if e.summary:
                        st.info(f"📝 {e.summary}")


# ═══════════════════════════════════════════
# Tab 4: 使用说明
# ═══════════════════════════════════════════
with tab4:
    st.markdown("""
    ## 📖 使用说明

    ### 两种评测模式
    | 模式 | 说明 |
    |------|------|
    | 🧪 **模拟评测** | 输入指令 → 自动生成对话 → 评测 |
    | 📤 **上传评测** | 上传已有对话文件 → 评测 |

    ### 🏆 创新功能
    | 功能 | 说明 |
    |------|------|
    | 🤖 **自动生成测试场景** | 输入指令，AI 自动生成多样化测试用例 |
    | 👥 **多评委一致性评分** | 3次独立评分 + 标准差 σ，可信度一目了然 |
    | 🎯 **违规定位** | 违规行为精准定位到第几轮、哪句话 |
    | 💡 **改进建议引擎** | 根据评测结果自动生成可操作建议 |
    | 📊 **雷达图可视化** | 7维评分直观对比 |

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

    ### 支持的文件格式
    - **JSON**: 标准格式、纯数组、单个对话
    - **JSONL**: 每行一个对话
    - **CSV**: 按 dialog_id 分组
    """)

st.divider()
st.caption("EvalAgent v2.0 · 美团 AI Hackathon 赛道 02 · DeepSeek V4 · 🏆 多评委 · 📊 雷达图 · 💡 建议引擎")
