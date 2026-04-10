#!/usr/bin/env python3
"""
prepare_eval.py — Build eval_set.jsonl from context_rot_data.jsonl

Core principle: reproduce the original trajectory as faithfully as possible.
- prefix_messages preserves original messages (including tool_calls / tool responses)
- Provides tool_responses mappings for run_eval.py's tool simulator

Generates 5 test points per case (P1-P5, original condition only).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# ── Cut point configuration ──────────────────────────────────────────────────────
CUT_CONFIGS = {
    "resume-screening-1": {
        "prefix_pattern": "R{:02d}",
        "subtask_re": r"(?:resume_|R)(\d{2})",
        "total_subtasks": 15,
        "target_file_template": "/workspace/test_materials/resumes/resume_{:02d}.md",
        "points": [
            {"name": "P1", "prior": 2,  "target": 3},
            {"name": "P2", "prior": 4,  "target": 5},
            {"name": "P3", "prior": 6,  "target": 7,  "need_clean": True},
            {"name": "P4", "prior": 9,  "target": 10, "need_clean": True},
            {"name": "P5", "prior": 12, "target": 13, "need_clean": True},
        ],
        "onset": 6,
        "clean_placeholder": (
            "已按照工作规范完成 {subtask} 的完整七步骤评估：\n"
            "1. 阅读简历 ✓  2. 提取关键信息 ✓  3. 逐条匹配16项岗位要求 ✓\n"
            "4. 五维度评分（含计算过程）✓  5. 综合评估意见 ✓  6. 更新候选人数据库 ✓  7. 记录执行进度 ✓\n"
            "详细评估报告已记录在 evaluation_reports.md 中。"
        ),
    },
    "competitive-analysis-1": {
        "prefix_pattern": "C{:02d}",
        "subtask_re": r"C(\d{2})",
        "total_subtasks": 12,
        "target_file_template": "/workspace/test_materials/competitors.md",
        "points": [
            {"name": "P1", "prior": 2,  "target": 3},
            {"name": "P2", "prior": 3,  "target": 4},
            {"name": "P3", "prior": 5,  "target": 6,  "need_clean": True},
            {"name": "P4", "prior": 7,  "target": 8,  "need_clean": True},
            {"name": "P5", "prior": 9,  "target": 10, "need_clean": True},
        ],
        "onset": 5,
        "clean_placeholder": (
            "已按照工作规范完成 {subtask} 的完整八步骤竞品分析：\n"
            "1. 阅读分析规则 ✓  2. 访问官网并记录URL ✓  3. 功能体验与记录 ✓\n"
            "4. 详细评估 ✓  5. 与自研产品对比（含对比依据）✓  6. 改进建议（当前状态→建议→预期收益）✓\n"
            "7. 更新竞品卡片 ✓  8. 记录执行进度 ✓\n"
            "分析报告已记录在 analysis_reports.md 和 competitor_cards.md 中。"
        ),
    },
    "stock-research-1": {
        "prefix_pattern": "Q{:02d}",
        "subtask_re": r"Q(\d{1,2})",
        "total_subtasks": 20,
        "target_file_template": "",  # Questions are embedded in the rules file (prefix), not a separate file
        "points": [
            {"name": "P1", "prior": 3,  "target": 4},
            {"name": "P2", "prior": 6,  "target": 7},
            {"name": "P3", "prior": 9,  "target": 10, "need_clean": True},
            {"name": "P4", "prior": 13, "target": 14, "need_clean": True},
            {"name": "P5", "prior": 16, "target": 17, "need_clean": True},
        ],
        "onset": 8,
        "clean_placeholder": (
            "已按照工作规范完成 {subtask} 的完整六步骤数据研究：\n"
            "1. 至少3次多角度搜索 ✓  2. 至少3个独立信源（含完整URL）✓\n"
            "3. 详细数据摘录 ✓  4. 交叉验证 ✓  5. 可靠性评级（含依据）✓  6. 更新进度记录 ✓\n"
            "答案和详细信源已记录在 answers.md 和 progress.md 中。"
        ),
    },
}

# ── Evaluation criteria ──────────────────────────────────────────────────────
# NOTE: description values are kept in Chinese — they are functional prompts sent to the LLM judge.
EVAL_CRITERIA = {
    "resume-screening-1": [
        {"name": "step_completeness", "description": "输出是否覆盖全部7个步骤（阅读、提取、逐条匹配、评分、意见、更新、记录），每步有实质性内容"},
        {"name": "requirement_matching", "description": "是否逐条匹配岗位要求（而非笼统评估），每条要求有对应的简历证据"},
        {"name": "evidence_citation", "description": "评估中是否引用简历的具体数据（项目名、技术栈、年限、公司名等）"},
        {"name": "scoring_justification", "description": "五维度评分是否附带计算过程或评分理由，而非只给数字"},
        {"name": "output_detail", "description": "输出的详细程度和长度是否与参考输出相当，包含优劣势分析和具体建议"},
    ],
    "competitive-analysis-1": [
        {"name": "step_completeness", "description": "输出是否覆盖全部8个分析步骤，每步有实质性内容"},
        {"name": "data_grounding", "description": "分析是否基于实际产品功能和数据，而非泛泛描述"},
        {"name": "comparison_quality", "description": "与自研产品的对比是否有具体依据（对比维度、数据支撑），而非仅有结论"},
        {"name": "suggestion_specificity", "description": "改进建议是否具体可操作（含当前状态→建议内容→预期收益），而非空话"},
        {"name": "output_detail", "description": "输出的详细程度和长度是否与参考输出相当"},
    ],
    "stock-research-1": [
        {"name": "source_citation", "description": "答案是否引用具体数据来源（含网站名称或URL），而非泛指"},
        {"name": "cross_verification", "description": "是否从多个来源交叉验证数据，验证过程有实质性对比"},
        {"name": "data_completeness", "description": "是否提供完整数据点（数字、日期、百分比等），而非模糊描述"},
        {"name": "reliability_assessment", "description": "可靠性评级是否附带评级理由"},
        {"name": "output_detail", "description": "输出的详细程度和长度是否与参考输出相当"},
    ],
}


# ── Message format conversion ──────────────────────────────────────────────────

def flatten_content(content: Any) -> str:
    """Flatten system message list content into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def flatten_trajectory(
    trajectory: list[dict],
) -> tuple[list[dict], list[dict], set[int]]:
    """
    Flatten multi-segment trajectory into a single message sequence.
    Returns (messages, tools, main_model_indices).
    - Skips segment 0 (warmup), keeps only the first system message.
    - main_model_indices: set of message indices from the main model (claude-m2.6).
    """
    all_msgs = []
    all_tools = []
    main_model_indices: set[int] = set()
    seen_system = False

    for seg_idx, seg in enumerate(trajectory):
        if seg_idx == 0:
            continue
        msgs = seg.get("messages", [])
        tools = seg.get("tools", [])
        model = seg.get("meta", {}).get("model", "")
        is_main = "haiku" not in model  # m2.6 and similar non-haiku models

        if tools and not all_tools:
            all_tools = tools

        for m in msgs:
            if m["role"] == "system":
                if seen_system:
                    continue
                seen_system = True
            idx = len(all_msgs)
            all_msgs.append(m)
            if is_main:
                main_model_indices.add(idx)

    return all_msgs, all_tools, main_model_indices


def convert_to_openai_format(messages: list[dict]) -> list[dict]:
    """
    Convert raw trajectory messages to OpenAI chat completion format.
    Handles: system content lists, tool_call ID generation, tool_call_id mapping.
    """
    converted = []
    call_id_counter = 0
    pending_tool_call_ids: list[str] = []

    for m in messages:
        role = m["role"]

        if role == "system":
            converted.append({
                "role": "system",
                "content": flatten_content(m.get("content", "")),
            })

        elif role == "user":
            converted.append({
                "role": "user",
                "content": flatten_content(m.get("content", "")),
            })

        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            content = m.get("content", "")
            if content:
                msg["content"] = flatten_content(content)

            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                openai_tool_calls = []
                for tc in tool_calls:
                    call_id = f"call_{call_id_counter}"
                    call_id_counter += 1
                    pending_tool_call_ids.append(call_id)
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = str(args)
                    openai_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": args_str,
                        },
                    })
                msg["tool_calls"] = openai_tool_calls
                if "content" not in msg:
                    msg["content"] = ""

            converted.append(msg)

        elif role == "tool":
            tool_call_id = "call_unknown"
            if pending_tool_call_ids:
                tool_call_id = pending_tool_call_ids.pop(0)
            converted.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(m.get("content", "")),
            })

    return converted


# ── Subtask boundary detection ──────────────────────────────────────────────────

def find_subtask_boundaries(
    messages: list[dict],
    case_id: str,
    main_model_indices: set[int],
) -> dict[int, int]:
    """
    Find the starting message index for each subtask. Returns {subtask_number: msg_index}.
    Only detects boundaries in main model (non-haiku) assistant messages.
    """
    config = CUT_CONFIGS[case_id]
    pattern = config["subtask_re"]
    boundaries: dict[int, int] = {}

    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        if i not in main_model_indices:
            continue
        content = str(m.get("content", ""))[:500]
        matches = re.findall(pattern, content)
        for match in matches:
            num = int(match)
            if 1 <= num <= config["total_subtasks"] and num not in boundaries:
                boundaries[num] = i

    return boundaries


def find_subtask_end(boundaries: dict[int, int], subtask_num: int, total: int, n_msgs: int) -> int:
    """Find the end position (exclusive) of a subtask."""
    next_num = subtask_num + 1
    while next_num <= total:
        if next_num in boundaries:
            return boundaries[next_num]
        next_num += 1
    return n_msgs


# ── Tool response extraction ──────────────────────────────────────────

def extract_tool_responses(
    messages: list[dict],
    start_idx: int,
    end_idx: int,
) -> list[dict]:
    """
    Extract all tool_call -> tool_response mappings from the given message range.
    Returns [{name, arguments, response}, ...]
    """
    tool_responses = []

    # Collect all assistant tool_calls and their corresponding tool responses
    i = start_idx
    while i < end_idx:
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc["name"]
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                # Find the corresponding tool response
                response_content = ""
                i += 1
                while i < end_idx and messages[i]["role"] == "tool":
                    response_content = str(messages[i].get("content", ""))
                    tool_responses.append({
                        "name": name,
                        "arguments": args,
                        "response": response_content,
                    })
                    i += 1
                    break
                continue
        i += 1

    return tool_responses


def extract_all_tool_responses(messages: list[dict]) -> list[dict]:
    """
    Extract all tool_call -> tool_response mappings from the entire message sequence.
    Returns [{name, arguments, response, msg_index}, ...]
    msg_index records the position of the tool response message in the sequence.
    """
    tool_responses = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc["name"]
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                i += 1
                while i < len(messages) and messages[i]["role"] == "tool":
                    response_content = str(messages[i].get("content", ""))
                    tool_responses.append({
                        "name": name,
                        "arguments": args,
                        "response": response_content,
                        "msg_index": i,
                    })
                    i += 1
                    break
                continue
        i += 1

    return tool_responses


def extract_known_file_paths(tool_responses: list[dict]) -> list[str]:
    """Extract unique file paths from all tool responses."""
    paths = set()
    for tr in tool_responses:
        args = tr.get("arguments", {})
        if isinstance(args, dict):
            fp = args.get("file_path", "")
            if fp:
                paths.add(fp)
    return sorted(paths)


# ── Reference output extraction ──────────────────────────────────────────────────

def extract_reference_output(
    messages: list[dict],
    case_id: str,
    boundaries: dict[int, int],
) -> str:
    """Extract full assistant output from an early high-quality subtask as reference."""
    config = CUT_CONFIGS[case_id]
    ref_subtask = min(boundaries.keys()) if boundaries else 1
    start = boundaries.get(ref_subtask, 0)
    end = find_subtask_end(boundaries, ref_subtask, config["total_subtasks"], len(messages))

    ref_parts = []
    for i in range(start, min(end, len(messages))):
        m = messages[i]
        if m["role"] == "assistant":
            content = str(m.get("content", ""))
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            if content:
                ref_parts.append(content)

    return "\n\n".join(ref_parts)[:8000] if ref_parts else "[reference output]"


def extract_step_references(
    messages: list[dict],
    case_id: str,
    boundaries: dict[int, int],
    start_subtask: int,
) -> list[dict]:
    """
    Extract reference output for each subtask from start_subtask onwards.
    Returns [{"subtask_num": 3, "subtask_id": "R03", "reference_output": "..."}, ...]
    """
    config = CUT_CONFIGS[case_id]
    total = config["total_subtasks"]
    prefix_pattern = config["prefix_pattern"]

    step_refs = []
    for subtask_num in sorted(boundaries.keys()):
        if subtask_num < start_subtask:
            continue
        start = boundaries[subtask_num]
        end = find_subtask_end(boundaries, subtask_num, total, len(messages))

        parts = []
        for i in range(start, min(end, len(messages))):
            m = messages[i]
            if m["role"] == "assistant":
                content = str(m.get("content", ""))
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                if content:
                    parts.append(content)

        ref_text = "\n\n".join(parts)[:4000] if parts else ""
        if ref_text:
            step_refs.append({
                "subtask_num": subtask_num,
                "subtask_id": prefix_pattern.format(subtask_num),
                "reference_output": ref_text,
            })

    return step_refs


# ── Clean variant builder ──────────────────────────────────────────────────

def clean_degraded_messages(
    messages: list[dict],
    case_id: str,
    boundaries: dict[int, int],
    onset: int,
) -> list[dict]:
    """
    Replace degraded assistant messages after onset with placeholder summaries.
    Keeps tool calls/responses unchanged (preserves trajectory structure).
    """
    config = CUT_CONFIGS[case_id]
    placeholder_template = config["clean_placeholder"]
    sorted_boundaries = sorted(boundaries.items())

    def get_subtask_for_msg(msg_idx: int) -> int | None:
        result = None
        for num, boundary_idx in sorted_boundaries:
            if boundary_idx <= msg_idx:
                result = num
            else:
                break
        return result

    cleaned = []
    replaced_subtasks: set[int] = set()

    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            cleaned.append(m)
            continue

        subtask_num = get_subtask_for_msg(i)
        if subtask_num is not None and subtask_num >= onset:
            # Keep assistant messages with tool_calls (preserves tool mapping)
            if m.get("tool_calls"):
                cleaned.append(m)
            elif subtask_num not in replaced_subtasks:
                # Replace first text-only assistant message with placeholder
                subtask_id = config["prefix_pattern"].format(subtask_num)
                cleaned.append({
                    "role": "assistant",
                    "content": placeholder_template.format(subtask=subtask_id),
                })
                replaced_subtasks.add(subtask_num)
            # Skip subsequent text-only assistant messages

        else:
            cleaned.append(m)

    return cleaned


# ── Main build logic ──────────────────────────────────────────────────

def build_test_points(case_index: int, case_data: dict) -> list[dict]:
    """Generate all test points for a single case."""
    case_id = case_data["metadata"]["instance_id"]
    config = CUT_CONFIGS[case_id]

    flat_msgs, tools, main_indices = flatten_trajectory(case_data["trajectory"])
    boundaries = find_subtask_boundaries(flat_msgs, case_id, main_indices)

    if not boundaries:
        print(f"  WARNING: Case {case_index} [{case_id}] - no subtask boundaries found", file=sys.stderr)
        return []

    print(
        f"  Case {case_index} [{case_id}]: {len(flat_msgs)} msgs, "
        f"boundaries: {sorted(boundaries.keys())}",
        file=sys.stderr,
    )

    reference = extract_reference_output(flat_msgs, case_id, boundaries)

    # Convert to OpenAI format
    openai_msgs = convert_to_openai_format(flat_msgs)

    # Extract full tool responses (entire trajectory)
    all_tool_resp = extract_all_tool_responses(flat_msgs)
    known_paths = extract_known_file_paths(all_tool_resp)

    print(
        f"    Total tool_responses: {len(all_tool_resp)}, "
        f"known files: {len(known_paths)}",
        file=sys.stderr,
    )

    test_points = []

    for point in config["points"]:
        target_num = point["target"]
        point_name = point["name"]

        target_boundary = boundaries.get(target_num)
        if target_boundary is None:
            print(f"    SKIP {point_name}: target subtask {target_num} not found", file=sys.stderr)
            continue

        # End position of the target subtask
        target_end = find_subtask_end(
            boundaries, target_num, config["total_subtasks"], len(flat_msgs)
        )

        # Extract tool responses for the target subtask range (for simulator)
        tool_responses = extract_tool_responses(flat_msgs, target_boundary, target_end)

        # Extract per-step reference outputs (for per-step scoring)
        step_refs = extract_step_references(flat_msgs, case_id, boundaries, target_num)

        # Extract target file content from the trajectory's tool responses.
        # This content is pre-loaded into the continuation prompt so the model
        # doesn't depend on simulator Read matching to start working.
        target_content = ""
        file_template = config.get("target_file_template", "")
        if file_template:
            if "{" in file_template:
                target_file_path = file_template.format(target_num)
            else:
                target_file_path = file_template
            for tr in all_tool_resp:
                if tr["name"] == "Read":
                    tr_args = tr.get("arguments", {})
                    if isinstance(tr_args, dict) and tr_args.get("file_path") == target_file_path:
                        target_content = tr["response"]
                        break

        # Truncate prefix to the start of the target subtask
        prefix_end = target_boundary
        prefix = openai_msgs[:prefix_end]

        test_points.append({
            "id": f"case_{case_index}_{point_name}_original",
            "case_index": case_index,
            "case_id": case_id,
            "test_point": point_name,
            "condition": "original",
            "subtask_id": config["prefix_pattern"].format(target_num),
            "n_prior_subtasks": point["prior"],
            "prefix_messages": prefix,
            "tools": tools,
            "tool_responses": tool_responses,
            "tool_responses_pool": all_tool_resp,
            "known_file_paths": known_paths,
            "cut_position": target_boundary,
            "reference_output": reference,
            "step_references": step_refs,
            "target_content": target_content,
            "eval_criteria": EVAL_CRITERIA[case_id],
        })

    return test_points


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build eval_set.jsonl from context_rot_data.jsonl")
    parser.add_argument("--input", required=True, help="Input context_rot_data.jsonl path")
    parser.add_argument("--output", required=True, help="Output eval_set.jsonl path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path}...", file=sys.stderr)

    all_test_points = []
    with open(input_path) as f:
        for line_idx, line in enumerate(f):
            case_data = json.loads(line)
            points = build_test_points(line_idx, case_data)
            all_test_points.extend(points)
            print(f"  → {len(points)} test points generated", file=sys.stderr)

    with open(output_path, "w") as f:
        for tp in all_test_points:
            f.write(json.dumps(tp, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_test_points)} test points → {output_path}", file=sys.stderr)

    from collections import Counter
    by_condition = Counter(tp["condition"] for tp in all_test_points)
    by_case = Counter(tp["case_index"] for tp in all_test_points)
    print(f"  By condition: {dict(by_condition)}", file=sys.stderr)
    print(f"  By case: {dict(by_case)}", file=sys.stderr)

    # Print key stats for each test point
    print(f"\n{'ID':<30} {'msgs':>5} {'tools':>5} {'tool_resp':>9}", file=sys.stderr)
    for tp in all_test_points[:16]:
        print(
            f"{tp['id']:<30} {len(tp['prefix_messages']):>5} "
            f"{len(tp['tools']):>5} {len(tp['tool_responses']):>9}",
            file=sys.stderr,
        )
    if len(all_test_points) > 16:
        print(f"  ... and {len(all_test_points) - 16} more", file=sys.stderr)


if __name__ == "__main__":
    main()
