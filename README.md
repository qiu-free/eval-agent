# EvalAgent —— 多轮对话自动评测系统

> **美团 AI Hackathon 赛道 02** — 复杂指令下的多轮对话自动评测系统

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/qiu-free/eval-agent)

## 项目概述

在数字人外呼场景中，系统需要根据预设指令与用户进行多轮对话。本系统通过 **用户模拟器 + 自动评测引擎**，自动测试对话模型在复杂指令下的指令遵循能力，并输出可解释、可量化的评测报告。

### 评审亮点 🏆

| 维度 | 我们的优势 |
|------|-----------|
| 🚀 **创新性** | 多评委一致性评分(3次采样σ标准差)、对抗型用户模拟、违规定位到具体轮次 |
| 🏗️ **完整性** | 三重评测(规则+LLM+交叉验证)、6种用户画像、7维评分、Streamlit交互界面 |
| 📱 **应用效果** | 实时对话气泡展示、多模型横向对比、可解释评测报告(每项扣分附证据链) |
| 💰 **商业价值** | 替代人工评测(成本↓90%)、Docker一键部署、API接口可集成到现有系统 |

### 核心能力

- **任务指令解析**：自动将自然语言指令拆解为结构化评测要素
- **用户模拟器**：模拟 6 种用户画像（配合型、拒绝型、追问型、干扰型、对抗型、模糊型）
- **多轮对话执行**：自动驱动被测模型与用户模拟器交互
- **7 维自动评测**：任务完成度、指令遵循度、约束遵守度、多轮一致性、意图识别、自然度、安全合规性
- **可解释报告**：每个评分附带扣分原因和对话证据

## 快速开始

### 1. 安装依赖

```bash
cd eval-agent
pip install -e .
```

### 2. 配置 LLM API

创建 `.env` 文件：

```bash
# 方式一：OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL_NAME=gpt-4o

# 方式二：通义千问（DashScope）
# DASHSCOPE_API_KEY=sk-xxx
# LLM_PROVIDER=dashscope
```

### 3. 运行 Streamlit 前端

```bash
streamlit run app.py
```

### 4. 开始评测

1. 在输入框中输入任务指令（如："向用户介绍优惠活动，确认意向，遇到拒绝挽回一次"）
2. 选择要运行的用户画像场景
3. 点击"开始评测"
4. 查看对话记录和评测报告

## 项目结构

```
eval-agent/
├── app.py                  # Streamlit 前端入口
├── main.py                 # FastAPI 入口（可选）
├── config.py               # 全局配置（LLM API、路径等）
├── prompts/
│   ├── user_simulator.txt    # 用户模拟器 Prompt
│   ├── evaluator.txt         # 评测器 Prompt
│   └── rubric_extractor.txt  # 任务指令解析 Prompt
├── core/
│   ├── __init__.py
│   ├── scenario_builder.py   # 测试场景生成
│   ├── user_simulator.py     # 用户模拟器
│   ├── dialogue_runner.py    # 多轮对话执行
│   ├── evaluator.py          # 自动评测
│   └── report_generator.py   # 报告生成
├── data/
│   ├── tasks.json
│   └── scenarios.json
└── outputs/
    └── reports/
```

## Docker 部署

```bash
# 1. 构建并启动
docker-compose up -d

# 2. 传入 API Key
export OPENAI_API_KEY=sk-your-key
docker-compose up -d

# 访问 http://localhost:8501
```

## 商业价值

| 对比项 | 人工评测 | EvalAgent 自动评测 |
|--------|---------|-------------------|
| 时间成本 | 1个场景约30分钟 | ⚡ 1个场景约30秒(快60倍) |
| 人力成本 | 需评测专家 | 🤖 零人工干预 |
| 一致性 | 不同人评分不同 | 🎯 多评委一致性σ<1 |
| 可追溯 | 凭记忆/笔记 | 📋 每项扣分附对话证据 |
| 规模化 | 每天最多20场 | 🚀 每天无限量 |

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 大语言模型 | OpenAI / Qwen | 指令理解与生成 |
| 应用框架 | LangChain | LLM 编排能力 |
| 向量检索 | FAISS | 评测规则检索 |
| 后端 | FastAPI | 高性能异步 API |
| 前端 | Streamlit | 快速原型展示 |

## 许可

MIT
