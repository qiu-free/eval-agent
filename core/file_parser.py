"""文件解析模块——解析上传的对话文件（CSV/JSON/JSONL/XLSX）"""

import csv
import io
import json
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any


@dataclass
class ParsedDialog:
    """解析后的单个对话"""
    scenario_label: str
    turns: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


@dataclass
class ParseResult:
    """文件解析结果"""
    task_instruction: str = ""
    dialogs: list[ParsedDialog] = field(default_factory=list)
    instruction_set: list[str] = field(default_factory=list)  # XLSX 指令集


# ── 字段名归一化 ──

ROLE_KEYS = ["role", "speaker", "from", "sender", "说话人"]
CONTENT_KEYS = ["content", "text", "message", "msg", "utterance", "话", "text_content"]
DIALOG_KEYS = ["turns", "messages", "dialog", "conversation", "history", "对话", "对话记录"]
INSTRUCTION_KEYS = ["task_instruction", "instruction", "task", "prompt", "system_prompt", "任务", "任务指令", "指令"]
LABEL_KEYS = ["scenario_label", "scenario", "label", "name", "场景", "type"]
DIALOG_ID_KEYS = ["dialog_id", "dialog", "session", "会话", "id", "conversation_id"]
TASK_KEYS = ["task_instruction", "instruction", "task", "任务", "任务指令"]


def normalize_turn(t: dict) -> tuple[str, str]:
    """归一化对话轮次字段名，返回 (role, content)"""
    role, content = "", ""
    for k in ROLE_KEYS:
        if k in t and t[k]:
            v = str(t[k]).strip().lower()
            if v in ("user", "用户", "u", "human", "顾客", "客户"):
                role = "user"
            elif v in ("assistant", "客服", "a", "bot", "agent", "model"):
                role = "assistant"
            break
    for k in CONTENT_KEYS:
        if k in t and t[k]:
            content = str(t[k]).strip()
            break
    return role, content


def parse_turns(item: Any) -> list[dict]:
    """递归解析对话轮次，支持多种嵌套结构"""
    if isinstance(item, list):
        if all(isinstance(x, str) for x in item):
            return [{"role": "user" if i % 2 == 0 else "assistant", "content": x} for i, x in enumerate(item)]
        return [{"role": r, "content": c} for r, c in [normalize_turn(t) for t in item if isinstance(t, dict)] if r and c]
    if isinstance(item, dict):
        for key in DIALOG_KEYS:
            if key in item:
                return parse_turns(item[key])
        r, c = normalize_turn(item)
        return [{"role": r, "content": c}] if r and c else []
    return []


def extract_task_instruction(data: dict) -> str:
    """从数据中提取任务指令"""
    for key in INSTRUCTION_KEYS:
        if key in data and isinstance(data[key], str) and len(data[key]) > 10:
            return data[key]
    return ""


def extract_label(item: Any, idx: int) -> str:
    """从数据项中提取场景标签"""
    if isinstance(item, dict):
        for key in LABEL_KEYS:
            if key in item and item[key]:
                return str(item[key])
    return f"场景 {idx + 1}"


def _parse_csv(raw_text: str) -> ParseResult:
    """解析 CSV 格式"""
    result = ParseResult()
    rows = list(csv.DictReader(io.StringIO(raw_text)))
    if not rows:
        return result

    dk = next((k for k in DIALOG_ID_KEYS if k in rows[0]), None)
    tk = next((k for k in TASK_KEYS if k in rows[0]), None)
    if tk:
        result.task_instruction = rows[0].get(tk, "")

    if dk:
        for gid, group in groupby(rows, lambda r: r.get(dk, "")):
            turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in list(group)] if r and c]
            if turns:
                result.dialogs.append(ParsedDialog(scenario_label=gid or "对话", turns=turns))
    else:
        turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in rows] if r and c]
        if turns:
            result.dialogs.append(ParsedDialog(scenario_label="对话", turns=turns))

    return result


def _parse_jsonl(raw_text: str) -> ParseResult:
    """解析 JSONL 格式"""
    result = ParseResult()
    for idx, line in enumerate(raw_text.split("\n")):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        turns = parse_turns(item)
        if turns:
            ti = extract_task_instruction(item)
            if ti and not result.task_instruction:
                result.task_instruction = ti
            result.dialogs.append(ParsedDialog(
                scenario_label=extract_label(item, idx),
                turns=turns,
            ))
    return result


def _parse_xlsx(file_bytes: bytes) -> ParseResult:
    """解析 XLSX 格式"""
    import openpyxl

    result = ParseResult()
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
    ws = wb.active
    rlist = list(ws.iter_rows(values_only=True))
    if not rlist:
        return result

    hdr = [str(h).lower() if h else "" for h in rlist[0]]
    drows = [dict(zip(hdr, r)) for r in rlist[1:] if any(c is not None for c in r)]
    if not drows:
        return result

    # 检测是否为指令集模式
    is_instr = any(k in hdr for k in ["任务指令示例", "instruction_example", "task_instruction", "任务指令"])
    if is_instr:
        ik = next((k for k in ["任务指令示例", "任务指令", "instruction", "task_instruction"] if k in hdr), None)
        instructions = []
        for r in drows:
            if ik and r.get(ik) and str(r[ik]).strip():
                instructions.append(str(r[ik]))
        if instructions:
            result.instruction_set = instructions
            return result

    # 普通对话模式
    dk = next((k for k in DIALOG_ID_KEYS if k in hdr), None)
    tk = next((k for k in TASK_KEYS if k in hdr), None)
    if tk and drows[0].get(tk):
        result.task_instruction = str(drows[0][tk] or "")

    if dk:
        for gid, group in groupby(drows, lambda r: r.get(dk, "")):
            turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in list(group)] if r and c]
            if turns:
                result.dialogs.append(ParsedDialog(scenario_label=str(gid) or "对话", turns=turns))
    else:
        turns = [{"role": r, "content": c} for r, c in [normalize_turn(r) for r in drows] if r and c]
        if turns:
            result.dialogs.append(ParsedDialog(scenario_label="对话", turns=turns))

    return result


def _parse_json(raw_text: str) -> ParseResult:
    """解析 JSON 格式"""
    result = ParseResult()
    data = json.loads(raw_text)
    result.task_instruction = extract_task_instruction(data)

    items = []
    if "dialogs" in data:
        items = data["dialogs"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for idx, item in enumerate(items):
        turns = parse_turns(item)
        if turns:
            result.dialogs.append(ParsedDialog(
                scenario_label=extract_label(item, len(result.dialogs)),
                turns=turns,
            ))

    return result


def parse_file(file_bytes: bytes, file_ext: str) -> ParseResult:
    """解析上传的对话文件

    Args:
        file_bytes: 文件原始字节
        file_ext: 文件扩展名 (csv, json, jsonl, xlsx)

    Returns:
        ParseResult 包含解析后的对话数据和任务指令
    """
    if file_ext == "xlsx":
        return _parse_xlsx(file_bytes)

    raw_text = file_bytes.decode("utf-8-sig")

    if file_ext == "csv":
        return _parse_csv(raw_text)
    elif file_ext == "jsonl":
        return _parse_jsonl(raw_text)
    else:
        return _parse_json(raw_text)
