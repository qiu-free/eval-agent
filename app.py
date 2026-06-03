"""EvalAgent Streamlit 前端——输入指令、运行评测、查看报告"""

import sys
sys.dont_write_bytecode = True

import html
import json
import traceback
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from config import settings
from core.dialogue_runner import DialogResult, Turn

# ── 从 Streamlit Cloud Secrets 加载 API Key（部署用）──
import streamlit as _st_secrets
try:
    if not settings.openai_api_key:
        _s = _st_secrets.secrets
        if _s.get("openai_api_key"):
            settings.openai_api_key = _s["openai_api_key"]
        if _s.get("openai_api_base"):
            settings.openai_api_base = _s["openai_api_base"]
except Exception:
    pass
del _st_secrets
from core.scenario_builder import ScenarioBuilder, Scenario
from core.dialogue_runner import DialogueRunner
from core.evaluator import Evaluator, DIMENSIONS, MultiJudgeResult
from core.report_generator import ReportGenerator
from core.llm_utils import safe_llm_call, get_llm_client
from core.file_parser import parse_file, ParseResult
from core.i18n_utils import _, _f, set_language, get_language

# ── 页面配置 ──
st.set_page_config(
    page_title="EvalAgent - 多轮对话自动评测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 样式（从外部文件加载，可被浏览器缓存）──
_CSS_PATH = Path(__file__).parent / "static" / "styles.css"
st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
    <h1>🎯 EvalAgent</h1>
    <p class="subtitle">多轮对话自动评测系统 · 让模型评测像呼吸一样简单</p>  <!-- 保留中文固定 -->
    <div class="hero-stats">
        <div class="hero-stat"><div class="stat-num">10维</div><div class="stat-label">评测体系</div></div>
        <div class="hero-stat"><div class="stat-num">6种</div><div class="stat-label">用户画像</div></div>
        <div class="hero-stat"><div class="stat-num">3评委</div><div class="stat-label">一致性评分</div></div>
        <div class="hero-stat"><div class="stat-num">4格式</div><div class="stat-label">上传支持</div></div>
        <div class="hero-stat"><div class="stat-num">σ&lt;1</div><div class="stat-label">评测可信度</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 历史持久化工具 ──
HISTORY_FILE = Path(__file__).parent / "outputs" / "eval_history.json"

def _load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_history(history: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 只保留最近 20 条，每条只存摘要（不含完整对话）
    slim = []
    for h in history[-20:]:
        scores = []
        for r in h.get("results", []):
            sc = r.get("scenario") if isinstance(r, dict) else r
            pn = sc.persona_name if hasattr(sc, "persona_name") else (sc.get("persona_name","?") if isinstance(sc,dict) else "?")
            ev = r.get("evaluation", {}) if isinstance(r, dict) else {}
            ov = ev.overall_score if hasattr(ev, "overall_score") else (ev.get("overall_score",0) if isinstance(ev,dict) else 0)
            scores.append({"persona": pn, "score": ov})
        slim.append({
            "time": h.get("time", ""),
            "count": h.get("count", 0),
            "instruction": h.get("instruction", "")[:120],
            "scores": scores,
        })
    HISTORY_FILE.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

# ── 初始化 Session State ──
if "results" not in st.session_state:
    st.session_state.results = []
if "running" not in st.session_state:
    st.session_state.running = False
if "eval_history" not in st.session_state:
    st.session_state.eval_history = _load_history()
if "generated_scenarios" not in st.session_state:
    st.session_state.generated_scenarios = None
if "upload_task" not in st.session_state:
    st.session_state.upload_task = None
if "loaded_history_label" not in st.session_state:
    st.session_state.loaded_history_label = ""

# ── 工具函数 ──
# 维度简称映射（避免截断导致歧义）
SHORT_NAMES = {
    "任务完成度": "任务完成",
    "指令遵循度": "指令遵循",
    "流程完成度": "流程完成",
    "约束遵守度": "约束遵守",
    "多轮一致性": "多轮一致",
    "用户意图识别": "意图识别",
    "知识准确性": "知识准确",
    "开场合规度": "开场合规",
    "对话自然度": "对话自然",
    "安全合规性": "安全合规",
}

# 雷达图配色方案
RADAR_COLORS = ['#667eea', '#43a047', '#ef6c00', '#c62828', '#8e24aa', '#00acc1']


def render_radar_chart(eval_result, title="评分维度"):
    """绘制单场景多维雷达图"""
    dims = []
    scores = []
    for dim in DIMENSIONS:
        key = dim["key"]
        if key in eval_result.dimensions:
            dims.append(SHORT_NAMES.get(dim["name"], dim["name"][:4]))
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


def render_comparison_radar_chart(results: list, max_scenarios: int = None, height: int = 400) -> go.Figure:
    """绘制多场景对比雷达图

    Args:
        results: 结果列表，每项包含 "scenario" 和 "evaluation" 键
        max_scenarios: 最大场景数（None 表示全部）
        height: 图表高度

    Returns:
        Plotly Figure 对象
    """
    items = results[:max_scenarios] if max_scenarios else results
    fig = go.Figure()

    for idx, result in enumerate(items):
        e = result["evaluation"]
        s = result["scenario"]
        dims = []
        scores = []
        for dim in DIMENSIONS:
            key = dim["key"]
            if key in e.dimensions:
                dims.append(SHORT_NAMES.get(dim["name"], dim["name"][:4]))
                scores.append(e.dimensions[key].score)
        if not scores:
            continue
        name = s.persona_name if hasattr(s, 'persona_name') else str(s)
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dims + [dims[0]],
            name=name,
            fill='toself',
            line=dict(color=RADAR_COLORS[idx % len(RADAR_COLORS)], width=2),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        legend=dict(orientation="h", y=-0.15),
        height=height,
        margin=dict(l=60, r=60, t=10, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
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

def _fmt_time(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{int(s%60)}s"

@st.cache_data(ttl=3600, show_spinner=False)
def _load_scenarios_cached():
    """缓存场景加载，避免每次 Streamlit 重执行都读取磁盘"""
    sb = ScenarioBuilder()
    return [s.to_dict() for s in sb.load_scenarios()]

def _dicts_to_scenarios(dicts: list) -> list:
    """将缓存的字典列表转换为 Scenario 对象列表"""
    return [Scenario(
        persona_id=d["persona_id"],
        persona_name=d["persona_name"],
        persona_description=d["description"],
        behavior=d["behavior"],
        test_goal=d["test_goal"],
    ) for d in dicts]

def generate_test_scenarios(task_instruction):
    """AI自动生成测试场景（使用共享客户端 + 自动重试）"""
    client = get_llm_client()
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

    def _call():
        resp = client.chat.completions.create(
            model=settings.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()

    try:
        text = safe_llm_call(_call, label="场景生成", fallback="")
        if not text:
            st.warning("AI 生成测试场景失败: 空响应")
            return {"scenarios": []}
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception as e:
        st.warning(f"AI 生成测试场景失败: {e}")
        return {"scenarios": []}


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 16px 0;">
        <div style="font-size:2rem;">🎯</div>
        <div style="font-weight:800; font-size:1.1rem; color:#1a1a2e;">EvalAgent</div>
        <div style="font-size:0.7rem; color:#999;">v2.0 · 美团 Hackathon</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 语言切换 ──
    lang = st.selectbox(_("🌐 语言/Language"), ["中文", "English"],
                        index=0 if get_language() == "zh" else 1)
    if lang == "English":
        if get_language() != "en": set_language("en"); st.rerun()
    elif lang == "中文":
        if get_language() != "zh": set_language("zh"); st.rerun()
    st.divider()

    with st.expander(_("⚙️ API 配置"), expanded=True):
        api_key = st.text_input("API Key", type="password",
                                value=settings.openai_api_key or "",
                                placeholder="sk-...", label_visibility="collapsed")
        if api_key:
            settings.openai_api_key = api_key

        settings.openai_model_name = st.selectbox("模型",
            ["deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o", "gpt-4o-mini", "qwen-plus"],
            index=0, label_visibility="collapsed")

        settings.openai_api_base = st.text_input("API 地址", value=settings.openai_api_base,
                                                  placeholder="https://api.deepseek.com",
                                                  label_visibility="collapsed")

    st.divider()
    st.markdown("### 📊 评测历史")
    if st.session_state.eval_history:
        for h in reversed(st.session_state.eval_history[-10:]):
            scores = h.get("scores", [])
            if not scores:
                for r in h.get("results", []):
                    sc = r.get("scenario") if isinstance(r, dict) else r
                    pn = sc.persona_name if hasattr(sc, "persona_name") else (sc.get("persona_name","?") if isinstance(sc,dict) else "?")
                    ev = r.get("evaluation", {}) if isinstance(r, dict) else {}
                    ov = ev.overall_score if hasattr(ev, "overall_score") else (ev.get("overall_score",0) if isinstance(ev,dict) else 0)
                    scores.append({"persona": pn, "score": ov})
            avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0
            color = "#43a047" if avg_score >= 80 else ("#ef6c00" if avg_score >= 60 else "#c62828")
            hid = h.get("time", "")
            has_results = bool(h.get("results"))
            disabled = not has_results
            if st.button(f"📂  {hid}  {h.get('count',0)}场  {avg_score:.0f}分" + (" 📄" if has_results else " ⚠️"),
                         key=f"hist_{hid}", use_container_width=True, disabled=disabled):
                st.session_state.results = h.get("results", [])
                st.session_state.loaded_history_label = hid
                st.rerun()
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.eval_history = []
            st.session_state.results = []
            _save_history([])
            st.rerun()
    else:
        st.caption("暂无历史记录")

    st.divider()
    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea15,#764ba215); border-radius:12px; padding:14px; margin-top:8px;">
        <div style="font-weight:700; color:#667eea; margin-bottom:8px;">🏆 创新亮点</div>
        <div style="font-size:0.8rem; line-height:1.8; color:#444;">
            <div>👥 多评委一致性 σ</div>
            <div>🎯 违规定位引擎</div>
            <div>🤖 AI 生成场景</div>
            <div>💡 改进建议引擎</div>
            <div>📊 10维雷达图</div>
            <div>📋 结构化指令 <span class="innovation-badge">NEW</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# Tab 1: 模拟评测
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 模拟评测",
    "📤 上传评测",
    "📊 评测结果",
    "⚙️ 系统设置",
    "📖 使用说明",
])

with tab1:
    # ── 输入模式切换 ──
    input_mode = st.radio(
        "📝 输入模式",
        ["自由文本", "结构化指令 (JSON)"],
        horizontal=True,
        help="自由文本：简单输入一段任务描述。结构化指令：支持 Role/Task/OpeningLine/CallFlow/FAQ/Constraints 完整字段"
    )

    if input_mode == "结构化指令 (JSON)":
        default_structured = json.dumps({
            "role": "美团外卖骑手的站长",
            "task": "致电「飞毛腿」骑手，通知合同签署并提醒配送任务",
            "opening_line": "你好，请问是${rider_name}吗？我是站长。你已报名飞毛腿，今天合同生效了。",
            "call_flow": [
                {
                    "step_id": "1",
                    "title": "告知合同生效并询问配送",
                    "reference_script": "今天飞毛腿合同已生效，请问你可以开始配送吗？"
                },
                {
                    "step_id": "2",
                    "title": "说明连续配送要求",
                    "reference_script": "单日合同需要连续完成配送，否则合同会受影响。"
                },
                {
                    "step_id": "3",
                    "title": "挽留与鼓励",
                    "reference_script": "尽量完成配送，注意安全。飞毛腿名额按排名，减少拒单取消超时有助于保住资格。"
                }
            ],
            "knowledge_points": {
                "飞毛腿名额": "飞毛腿报名按排名进行，非站长干预。减少拒单、取消和超时，恶劣天气多跑单有助于保住资格。",
                "如何退出": "需在前一天指定时间前在 App「飞毛腿报名」中取消，次日生效。",
                "合同要求": "单日合同当天需完成指定单数；多日合同每天需完成指定单数。"
            },
            "constraints": {
                "max_words_per_turn": 30,
                "tone": "口语化、像打电话一样自然",
                "forbidden_phrases": ["好的", "哈哈", "嘿嘿", "嘻嘻"]
            }
        }, ensure_ascii=False, indent=2)

        structured_json = st.text_area(
            "📋 结构化指令 (JSON)", value=default_structured, height=300,
            help="支持字段: role, task, opening_line, call_flow, knowledge_points, constraints"
        )

        # 解析结构化指令
        try:
            structured_data = json.loads(structured_json)
            task_instruction = structured_data.get("task", "")
            st.success(f"✅ 已解析结构化指令 — Role: {structured_data.get('role','N/A')} | {len(structured_data.get('call_flow',[]))} 个流程步骤 | {len(structured_data.get('knowledge_points',{}))} 条FAQ")
        except json.JSONDecodeError:
            st.error("❌ JSON 格式无效，请检查")
            task_instruction = ""
    else:
        default_instruction = "向用户介绍优惠活动，确认用户是否有意向领取优惠券，遇到拒绝要挽回一次，不能透露内部价格策略，最终要完成意向收集。"
        task_instruction = st.text_area(
            "📝 任务指令", value=default_instruction, height=120,
        )

    # ── 场景选择 + 参数区 ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎭 测试场景配置</div>', unsafe_allow_html=True)

    scenario_builder = ScenarioBuilder()
    all_scenarios = _dicts_to_scenarios(_load_scenarios_cached())

    # 构建带图标的选项标签
    persona_icons = {
        "cooperative": "😊", "rejecting": "😤", "inquiring": "🤔",
        "distracting": "😵", "adversarial": "😈", "ambiguous": "😶"
    }
    persona_options = [
        f"{persona_icons.get(s.persona_id, '👤')} {s.persona_name}"
        for s in all_scenarios
    ]
    persona_ids = [s.persona_id for s in all_scenarios]

    # 默认选中前三种
    default_ids = ["cooperative", "rejecting", "inquiring"]
    default_indices = [i for i, pid in enumerate(persona_ids) if pid in default_ids]

    selected_labels = st.pills(
        "选择画像（可多选）",
        options=persona_options,
        default=[persona_options[i] for i in default_indices],
        selection_mode="multi",
        label_visibility="collapsed",
    )

    # 将标签映射回 ID
    selected_personas = []
    for label in selected_labels:
        idx = persona_options.index(label)
        selected_personas.append(persona_ids[idx])

    # 显示已选画像描述
    if selected_personas:
        descriptions = []
        persona_desc = {
            "cooperative": "配合基本流程",
            "rejecting": "先拒绝，测试挽回",
            "inquiring": "追问细节与合规",
            "distracting": "打断转移话题",
            "adversarial": "诱导违规边界",
            "ambiguous": "回答模糊，测意图确认"
        }
        for pid in selected_personas:
            desc = persona_desc.get(pid, "")
            icon = persona_icons.get(pid, "")
            if desc:
                descriptions.append(f"{icon} {desc}")
        st.caption("已选: " + " · ".join(descriptions))
    else:
        st.warning("⚠️ 请至少选择一个用户画像")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        max_turns = st.slider("🔁 最大轮次", 3, 12, 8)
        compare_mode = st.checkbox("🔀 双模型对比", value=False, help="同时用两个模型跑对话并对比评分")
        if compare_mode:
            compare_model = st.text_input("对比模型", value=settings.compare_model_name, key="compare_model_input")
    with c2:
        # 自动生成场景按钮
        gen_clicked = st.button("🤖 AI 生成场景", use_container_width=True)
        if gen_clicked:
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
    with c3:
        st.caption("")  # spacing

    st.markdown('</div>', unsafe_allow_html=True)

    # 显示自动生成的场景
    if st.session_state.generated_scenarios:
        with st.expander("🤖 AI 生成的测试场景", expanded=True):
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

        live_area = st.empty()

        try:
            # 使用 st.status 实现流式进度展示
            with st.status("🔍 解析任务指令...", expanded=True) as eval_status:
                if input_mode == "结构化指令 (JSON)" and structured_data:
                    rubric = scenario_builder.parse_structured_instruction(structured_data)
                else:
                    rubric = scenario_builder.parse_instruction(task_instruction)
                eval_status.update(label="✅ 指令解析完成", state="running")

            with st.expander("📋 指令解析", expanded=True):
                if rubric.role:
                    st.markdown(f"**🎭 角色**: {rubric.role}")
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**目标**: {rubric.task_goal}")
                    if rubric.must_do:
                        st.markdown(f"**必须做**: {'、'.join(rubric.must_do)}")
                    if rubric.opening_line:
                        st.markdown(f"**开场白**: {rubric.opening_line[:60]}...")
                with cb:
                    if rubric.must_not_do:
                        st.markdown(f"**禁止做**: {'、'.join(rubric.must_not_do)}")
                    st.markdown(f"**约束**: 最多 {rubric.constraints.get('max_turns', 'N/A')} 轮")
                    if rubric.call_flow:
                        st.markdown(f"**流程步骤**: {len(rubric.call_flow)} 步")
                if rubric.call_flow:
                    st.markdown("**通话流程**:")
                    for step in rubric.call_flow:
                        st.caption(f"步骤{step.step_id}: {step.title}")
                if rubric.knowledge_points:
                    st.markdown(f"**知识库**: {len(rubric.knowledge_points)} 条 FAQ")

            selected_scenarios = [s for s in all_scenarios if s.persona_id in selected_personas]
            total = len(selected_scenarios)

            if total == 0:
                st.warning("⚠️ 请至少选一个画像")
                st.session_state.running = False
                st.rerun()

            dialogue_runner = DialogueRunner()
            evaluator = Evaluator()
            report_gen = ReportGenerator()

            # 总进度 status
            _t0 = datetime.now()
            with st.status(f"🎭 评测中... 0/{total} 场景", expanded=True) as status:
                for i, scenario in enumerate(selected_scenarios):
                    status.update(label=f"💬 [{i+1}/{total}] {scenario.persona_name} — 对话中...", state="running")

                    try:
                        _t0 = datetime.now()
                        dialog_result = dialogue_runner.run_dialog(
                            scenario=scenario, rubric=rubric, max_turns=max_turns,
                        )
                        _t1 = (datetime.now() - _t0).total_seconds()

                        # 流式展示对话
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
                        status.write(f"💬 {scenario.persona_name} — 共 {len(dialog_result.turns)} 轮对话")

                        # 双模型对比
                        compare_result = None
                        compare_model_name = None
                        if compare_mode and compare_model:
                            compare_model_name = compare_model
                            status.update(label=f"🔀 [{i+1}/{total}] {scenario.persona_name} — {compare_model_name} 对话中...", state="running")
                            try:
                                # A/B 同轮对比：相同的用户对话历史，仅替换模型回复
                                orig_model = settings.target_model_name
                                settings.target_model_name = compare_model_name
                                try:
                                    compare_runner = DialogueRunner()
                                    compare_dialog = DialogResult(scenario=scenario)
                                    compare_history: list[dict] = []
                                    for turn in dialog_result.turns:
                                        user_msg = turn.user
                                        assistant_msg = compare_runner._call_target_model(
                                            rubric, compare_history, user_msg,
                                        )
                                        ct = _Turn(
                                            user=user_msg, assistant=assistant_msg,
                                            turn_number=turn.turn_number,
                                        )
                                        compare_dialog.turns.append(ct)
                                        compare_history.append({"role": "user", "content": user_msg})
                                        compare_history.append({"role": "assistant", "content": assistant_msg})
                                finally:
                                    settings.target_model_name = orig_model
                                compare_eval = evaluator.evaluate(compare_dialog, rubric)
                                compare_result = {"dialog": compare_dialog, "evaluation": compare_eval, "model": compare_model_name}
                            except Exception as e:
                                st.warning(f"⚠️ 对比模型 {compare_model_name} 失败: {e}")

                        status.update(label=f"📊 [{i+1}/{total}] {scenario.persona_name} — 多评委评测中...", state="running")
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
                            "compare": compare_result,
                        })

                        status.write(f"✅ {scenario.persona_name} — 完成")

                        st.markdown(render_score_card(
                            mj_result.overall_mean, scenario.persona_name, mj_result.overall_std
                        ), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ {scenario.persona_name} 失败: {e}")
                        continue

            elapsed_all = (datetime.now() - _t0).total_seconds()
            status.update(label=f"🎉 全部完成！{len(st.session_state.results)}/{total} 个场景  （耗时 {_fmt_time(elapsed_all)}）", state="complete")

            # 保存到历史
            st.session_state.eval_history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "count": len(st.session_state.results),
                "instruction": task_instruction,
                "results": st.session_state.results.copy(),
            })
            _save_history(st.session_state.eval_history)

            st.balloons()

            # 导出按钮
            with st.expander("📥 导出报告", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    for idx, result in enumerate(st.session_state.results):
                        s = result["scenario"]
                        e = result["evaluation"]
                        d = result["dialog"]
                        try:
                            pdf_buf = report_gen.generate_pdf(d, e, rubric, task_instruction)
                            st.download_button(
                                label=f"📄 {s.persona_name} PDF",
                                data=pdf_buf,
                                file_name=f"eval_report_{s.persona_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                key=f"pdf_{idx}", use_container_width=True,
                            )
                        except Exception:
                            st.caption(f"⚠️ {s.persona_name} PDF失败")
                with c2:
                    try:
                        xl_buf = report_gen.generate_excel(st.session_state.results)
                        st.download_button(
                            label="📊 全部场景 Excel",
                            data=xl_buf,
                            file_name=f"eval_report_all_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_export", use_container_width=True,
                        )
                    except Exception:
                        st.caption("⚠️ Excel导出失败")

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

        # 双模型对比摘要
        if any(r.get("compare") for r in st.session_state.results):
            st.markdown("### 🔀 模型对比")
            for idx, result in enumerate(st.session_state.results):
                cmp = result.get("compare")
                if not cmp:
                    continue
                e1 = result["evaluation"]
                e2 = cmp["evaluation"]
                m1 = settings.target_model_name
                m2 = cmp["model"]
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    delta = e1.overall_score - e2.overall_score
                    st.metric(f"{m1} ({result['scenario'].persona_name})", f"{e1.overall_score}分")
                with c2:
                    st.metric(f"{m2}", f"{e2.overall_score}分", delta=f"{delta:+.0f}" if delta != 0 else None)
                with c3:
                    winner = m1 if delta >= 0 else m2
                    st.caption(f"🏆 {winner}")

        # 场景对比雷达图
        if len(st.session_state.results) > 1:
            st.markdown("#### 📈 场景对比")
            fig = render_comparison_radar_chart(st.session_state.results, height=350)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# Tab 2: 上传评测
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 📤 上传对话文件进行评测")
    st.caption("支持 CSV / JSON / JSONL 格式，自动适配多种字段命名")

    uploaded_file = st.file_uploader("选择文件", type=["json", "csv", "jsonl", "xlsx"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "json"

        try:
            file_bytes = uploaded_file.read()
            parse_result = parse_file(file_bytes, file_ext)
            task_instruction = parse_result.task_instruction

            # ── XLSX 指令集模式 ──
            if parse_result.instruction_set:
                instructions = parse_result.instruction_set
                st.success(f"✅ 已解析 {len(instructions)} 条任务指令")
                for idx, instr in enumerate(instructions):
                    label = instr.split("\n")[0][:60].strip().strip("#").strip() or f"指令 {idx+1}"
                    if st.button(f"🚀 评测指令{idx+1}: {label}", key=f"eval_instr_{idx}", use_container_width=True):
                        st.session_state.upload_task = instr
                        st.session_state.running = True
                        st.session_state.results = []
                        st.rerun()
                st.info("👆 选择一条指令，系统将自动评测（请在下方查看进度和结果）")

            # ── 普通对话模式 ──
            dialogs_data = [{"scenario_label": d.scenario_label, "turns": d.turns} for d in parse_result.dialogs]

            if not parse_result.instruction_set and not dialogs_data:
                st.error("❌ 未能解析出有效对话")
                st.stop()

            if not parse_result.instruction_set:
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

            if not parse_result.instruction_set and st.button("🚀 评测上传的对话", type="primary", use_container_width=True):
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

                    scenario = Scenario(
                        persona_id=f"u{i}",
                        persona_name=dd['scenario_label'],
                        persona_description="上传评测",
                        behavior=[],
                        test_goal="上传评测",
                    )
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
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "count": len(results),
                    "instruction": task_instruction, "results": results.copy(),
                })
                _save_history(st.session_state.eval_history)
                st.balloons()

        except Exception as e:
            st.error(f"❌ 处理出错: {e}")
            st.code(traceback.format_exc(), language="python")

    # ── 指令集批量评测（独立于文件上传，经st.rerun后仍执行）──
    if st.session_state.get("upload_task") and st.session_state.get("running"):
        instr = st.session_state.upload_task
        sb = ScenarioBuilder()
        runner = DialogueRunner()
        ev2 = Evaluator()
        rubric = sb.parse_instruction(instr)
        key_personas = ["cooperative", "rejecting", "inquiring"]
        all_sc = [s for s in sb.load_scenarios() if s.persona_id in key_personas]
        total = len(all_sc)
        pb = st.progress(0, text="初始化...")
        sbx = st.empty()
        for i2, sc in enumerate(all_sc):
            try:
                pct = int((i2+1)/total*100)
                pb.progress(pct, text=f"[{i2+1}/{total}] {sc.persona_name}")
                sbx.info(f"▶ Step 1/2: {sc.persona_name} — 模拟对话...")
                dr = runner.run_dialog(scenario=sc, rubric=rubric, max_turns=4)
                sbx.info(f"▶ Step 2/2: {sc.persona_name} — 自动评测...")
                mj = ev2.multi_judge_evaluate(dr, rubric, num_judges=1)
                st.session_state.results.append({"scenario":sc,"dialog":dr,"evaluation":mj.individual_results[-1],"multi_judge":mj})
                sbx.success(f"✅ {sc.persona_name}: {mj.overall_mean}/100")
            except Exception as e2:
                st.error(f"❌ {sc.persona_name}: {e2}")
                with st.expander("🔍 错误堆栈"):
                    st.code(traceback.format_exc(), language="python")
        st.session_state.running = False
        sbx.success(f"🎉 完成! {len(st.session_state.results)}/{total} 个场景")

    # ── 指令集评测结果展示（评测完成后持久显示）──
    if st.session_state.get("upload_task") and not st.session_state.get("running") and len(st.session_state.results) > 0:
        st.divider()
        st.markdown("### 📊 批量评测结果")
        res_cols = st.columns(min(len(st.session_state.results), 3))
        for idx, r in enumerate(st.session_state.results):
            with res_cols[idx]:
                sc = r["scenario"]; mj = r["multi_judge"]
                n = sc.persona_name if hasattr(sc,'persona_name') else "场景"
                st.metric(n, f"{mj.overall_mean}/100", f"σ={mj.overall_std}")
        if len(st.session_state.results) > 0:
            fig = render_comparison_radar_chart(st.session_state.results, max_scenarios=3, height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 提示：点击上方「📊 评测结果」Tab可查看完整报告和对话详情")


# ═══════════════════════════════════════════
# Tab 3: 评测结果
# ═══════════════════════════════════════════
with tab3:
    if not st.session_state.results:
        st.info("💡 先在「模拟评测」或「上传评测」运行一次")
    else:
        if st.session_state.get("loaded_history_label"):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption(f"📂 正在查看历史记录: {st.session_state.loaded_history_label}")
            with c2:
                if st.button("← 返回当前结果", use_container_width=True):
                    st.session_state.results = []
                    st.session_state.loaded_history_label = ""
                    st.rerun()
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
            fig = render_comparison_radar_chart(st.session_state.results)
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
# Tab 4: 系统设置 + 仪表盘
# ═══════════════════════════════════════════
with tab4:
    # ── 评测仪表盘 ──
    hist = st.session_state.eval_history
    if hist:
        st.markdown("### 📈 评测仪表盘")
        dcols = st.columns(4)
        total_evals = len(hist)
        total_scenarios = sum(h.get("count", 0) for h in hist)
        all_scores = []
        for h in hist:
            for r in h.get("results", []):
                if isinstance(r, dict):
                    ev = r.get("evaluation") or {}
                    ov = ev.overall_score if hasattr(ev, "overall_score") else (ev.get("overall_score",0) if isinstance(ev,dict) else 0)
                    all_scores.append(ov)
        avg = sum(all_scores) / len(all_scores) if all_scores else 0
        recent_scores = all_scores[-10:] if len(all_scores) >= 10 else all_scores
        trend = "↑" if len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0] else \
                "↓" if len(recent_scores) >= 2 and recent_scores[-1] < recent_scores[0] else "→"
        dcols[0].metric("总评测次数", total_evals)
        dcols[1].metric("总场景数", total_scenarios)
        dcols[2].metric("平均分", f"{avg:.1f}", delta=trend if trend != "→" else None)
        dcols[3].metric("最近趋势", trend, delta_color="off")

        # 维度分排行
        dim_avgs = {}
        for d in DIMENSIONS:
            dim_avgs[d["name"]] = {"sum": 0, "count": 0, "weight": d["weight"]}
        for h in hist:
            for r in h.get("results", []):
                if not isinstance(r, dict): continue
                ev = r.get("evaluation") or {}
                dims = ev.dimensions if hasattr(ev, "dimensions") else (ev.get("dimensions", {}) if isinstance(ev, dict) else {})
                for dn, dv in dim_avgs.items():
                    dk = next((x["key"] for x in DIMENSIONS if x["name"] == dn), None)
                    if dk and dk in dims:
                        dv["sum"] += dims[dk].score if hasattr(dims[dk], "score") else (dims[dk].get("score",0) if isinstance(dims[dk], dict) else 0)
                        dv["count"] += 1
        st.markdown("#### 各维度平均分")
        vibar = []
        for dn, dv in sorted(dim_avgs.items(), key=lambda x: x[1]["sum"]/(x[1]["count"] or 1), reverse=True):
            av = dv["sum"] / dv["count"] if dv["count"] else 0
            vibar.append({"维度": dn, "平均分": round(av, 1), "权重": f"{dv['weight']*100:.0f}%"})
        if vibar:
            fig = go.Figure(data=[go.Bar(x=[v["维度"] for v in vibar], y=[v["平均分"] for v in vibar],
                                          marker_color=['#43a047' if v["平均分"]>=3.5 else '#ef6c00' if v["平均分"]>=2.5 else '#c62828' for v in vibar])])
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(range=[0,5.5]))
            st.plotly_chart(fig, use_container_width=True)
        st.divider()

    st.markdown("### ⚙️ 系统设置")

    col_a, col_b = st.columns(2)
    with col_a:
        api_key = st.text_input(
            "🔑 API Key", value=settings.openai_api_key, type="password",
            placeholder="sk-..."
        )
        if api_key and api_key != settings.openai_api_key:
            settings.openai_api_key = api_key
            st.success("✅ API Key 已更新")

        model = st.selectbox(
            "🤖 模型选择",
            ["deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o", "gpt-4o-mini", "qwen-plus"],
            index=0,
        )
        if model != settings.openai_model_name:
            settings.openai_model_name = model
            st.info(f"✅ 已切换为 {model}")

        provider = st.selectbox("🌐 API 地址", ["DeepSeek", "OpenAI", "DashScope"], index=0)
        if provider == "DashScope" and settings.llm_provider != "dashscope":
            settings.llm_provider = "dashscope"
        elif provider == "OpenAI" and settings.llm_provider != "openai":
            settings.llm_provider = "openai"

    with col_b:
        st.markdown(f"""
        #### 📋 当前配置
        - **API Key**: {'✅ 已设置' if settings.openai_api_key else '❌ 未设置'}
        - **模型**: {settings.openai_model_name}
        - **API**: {settings.llm_provider}
        - **数据目录**: `{settings.data_dir}`
        - **输出目录**: `{settings.output_dir}`
        """)

    st.divider()
    st.markdown("#### 💰 成本计算器")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        daily = st.number_input("日均评测量", 1, value=1000)
    with cc2:
        rate = st.number_input("人力时薪(¥)", 1, value=50)
    with cc3:
        mins = st.number_input("单场景耗时(分钟)", 1, value=30)

    if st.button("计算节省成本", use_container_width=True):
        human = daily * (mins/60) * rate * 365
        agent = daily * 2000 * 0.000002 * 365
        saving = human - agent
        repl = (1 - agent/human) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("人力成本/年", f"¥{human:,.0f}")
        c2.metric("EvalAgent/年", f"¥{agent:,.0f}")
        c3.metric("年节省", f"¥{saving:,.0f}", f"{repl:.1f}%")

    st.divider()
    st.markdown("#### 📊 系统检查")
    hc = st.columns(4)
    api_ok = bool(settings.openai_api_key)
    hc[0].metric("API Key", "✅" if api_ok else "❌")
    hc[1].metric("用户画像", "6种")
    hc[2].metric("文件格式", "5种")
    hc[3].metric("评测维度", "10维")

# ═══════════════════════════════════════════
# Tab 5: 使用说明
# ═══════════════════════════════════════════
with tab5:
    st.markdown("## 📖 EvalAgent 使用指南")

    # ── Quick Start cards ──
    st.markdown("### 🚀 快速开始")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#667eea15,#764ba215); border-radius:14px; padding:18px; border:1px solid #667eea30; height:100%;">
            <div style="font-size:2rem; text-align:center;">🧪</div>
            <div style="font-weight:700; text-align:center; margin:4px 0;">模拟评测</div>
            <div style="font-size:0.8rem; color:#666; text-align:center;">输入任务指令 → AI自动模拟对话 → 10维评测报告</div>
        </div>
        """, unsafe_allow_html=True)
    with qc2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#43a04715,#66bb6a15); border-radius:14px; padding:18px; border:1px solid #43a04730; height:100%;">
            <div style="font-size:2rem; text-align:center;">📤</div>
            <div style="font-weight:700; text-align:center; margin:4px 0;">上传评测</div>
            <div style="font-size:0.8rem; color:#666; text-align:center;">上传JSON/CSV/JSONL/XLSX对话文件 → 自动评测</div>
        </div>
        """, unsafe_allow_html=True)
    with qc3:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#ef6c0015,#ffa72615); border-radius:14px; padding:18px; border:1px solid #ef6c0030; height:100%;">
            <div style="font-size:2rem; text-align:center;">🔌</div>
            <div style="font-weight:700; text-align:center; margin:4px 0;">API评测</div>
            <div style="font-size:0.8rem; color:#666; text-align:center;">POST /evaluate → 返回JSON结果，可集成CI/CD</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Two input modes ──
    st.markdown("### 📝 两种输入模式")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:20px; border:1px solid #e0e0e0;">
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px;">🔄 自由文本模式</div>
            <div style="font-size:0.85rem; color:#555;">
                简单输入一段任务描述，系统自动解析为结构化评测要素。<br><br>
                <b>适用场景</b>：快速测试、无固定格式要求的任务<br>
                <b>示例</b>：<i>"向用户介绍优惠活动，确认意向，拒绝后挽回一次"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:20px; border:1px solid #e0e0e0;">
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px;">📋 结构化指令模式 <span class="innovation-badge">NEW</span></div>
            <div style="font-size:0.85rem; color:#555;">
                以JSON格式输入完整的结构化指令，含角色、开场白、通话流程、FAQ知识库、约束条件。<br><br>
                <b>适用场景</b>：正式评测、外呼任务、有明确SOP的场景<br>
                <b>支持字段</b>：role · task · opening_line · call_flow · knowledge_points · constraints
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Innovation features with better layout ──
    st.markdown("### 🏆 核心创新功能")
    features = [
        ("👥", "多评委一致性评分", "3次独立评分取均值±标准差σ，σ越小越可信，σ>10自动启动仲裁"),
        ("🎯", "违规定位引擎", "违规行为精准定位到第几轮、哪句话，不是笼统的「有问题」"),
        ("🤖", "AI自动生成测试场景", "输入指令自动生成4个多样化测试场景，覆盖成功/边缘路径"),
        ("💡", "改进建议引擎", "低分维度自动分析原因并给出可操作的改进方向"),
        ("📊", "雷达图可视化", "10维评分+多场景对比，一张图看懂模型能力全景"),
        ("🔀", "模型分离机制", "被测模型≠模拟用户≠评测引擎，杜绝「自己评自己」的循环偏差"),
    ]
    for emoji, title, desc in features:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; background:white; border-radius:12px; padding:14px; margin-bottom:8px; border:1px solid #eee; gap:12px;">
            <div style="font-size:1.5rem; min-width:40px; text-align:center;">{emoji}</div>
            <div>
                <div style="font-weight:700; font-size:0.95rem;">{title}</div>
                <div style="font-size:0.8rem; color:#666;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Dimensions table ──
    st.markdown("### 📊 10维评测体系")
    dim_data = []
    for d in DIMENSIONS:
        w = d['weight']
        if w >= 0.15:
            badge = '<span class="badge badge-red">核心</span>'
        elif w >= 0.08:
            badge = '<span class="badge badge-yellow">重要</span>'
        else:
            badge = '<span class="badge badge-green">辅助</span>'
        dim_data.append({
            "维度": f"{d['name']} {badge}",
            "权重": f"{w*100:.0f}%",
            "说明": d['description'],
            "评测方式": "LLM语义" if "completion" not in d['key'] and "accuracy" not in d['key'] else ("规则+LLM" if "adherence" in d['key'] or "compliance" in d['key'] else "LLM+Faq库"),
        })
    st.dataframe(dim_data, use_container_width=True, hide_index=True,
                  column_config={
                      "维度": st.column_config.Column(width="medium"),
                      "权重": st.column_config.Column(width="small"),
                      "说明": st.column_config.Column(width="large"),
                      "评测方式": st.column_config.Column(width="small"),
                  })

    st.divider()

    # ── Supported formats ──
    st.markdown("### 📁 支持的文件格式")
    fmt1, fmt2, fmt3, fmt4 = st.columns(4)
    with fmt1:
        st.markdown("""
        <div style="text-align:center; background:white; border-radius:12px; padding:16px; border:1px solid #eee;">
            <div style="font-size:1.8rem;">{ }</div>
            <div style="font-weight:700;">JSON</div>
            <div style="font-size:0.75rem; color:#888;">标准格式 / 纯数组 / 单对话</div>
        </div>
        """, unsafe_allow_html=True)
    with fmt2:
        st.markdown("""
        <div style="text-align:center; background:white; border-radius:12px; padding:16px; border:1px solid #eee;">
            <div style="font-size:1.8rem;">≡</div>
            <div style="font-weight:700;">JSONL</div>
            <div style="font-size:0.75rem; color:#888;">每行一个独立对话</div>
        </div>
        """, unsafe_allow_html=True)
    with fmt3:
        st.markdown("""
        <div style="text-align:center; background:white; border-radius:12px; padding:16px; border:1px solid #eee;">
            <div style="font-size:1.8rem;">📊</div>
            <div style="font-weight:700;">CSV</div>
            <div style="font-size:0.75rem; color:#888;">按 dialog_id 自动分组</div>
        </div>
        """, unsafe_allow_html=True)
    with fmt4:
        st.markdown("""
        <div style="text-align:center; background:white; border-radius:12px; padding:16px; border:1px solid #eee;">
            <div style="font-size:1.8rem;">📗</div>
            <div style="font-weight:700;">XLSX</div>
            <div style="font-size:0.75rem; color:#888;">Excel 对话/指令集</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Tips ──
    st.markdown("### 💡 使用技巧")
    tips = [
        "🔑 先在左侧边栏配置 API Key，支持 DeepSeek / OpenAI / DashScope",
        "🎯 结构化指令模式是评测外呼任务的最佳选择——完整的 Call Flow + FAQ + 约束确保评测全面",
        "📊 多场景对比雷达图能直观看出不同用户画像下模型表现的差异",
        "📤 上传评测时，系统自动适配多种字段命名（role/speaker/from/说话人 等），无需手动调整",
        "📈 σ<1.0 表示多评委高度一致，评测结果非常可信",
    ]
    for tip in tips:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea08,#764ba208); border-radius:10px; padding:12px 16px; margin-bottom:6px; border-left:3px solid #667eea;">
            <span style="font-size:0.9rem;">{tip}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("EvalAgent v2.0 · 美团 AI Hackathon 赛道 02 · 🏆 多评委 · 📊 10维雷达图 · 💡 建议引擎 · 📋 结构化指令")
