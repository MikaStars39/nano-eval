#!/usr/bin/env python3
"""
run_eval.py — Context Rot evaluation runner (async concurrent)

For each test point in eval_set.jsonl:
1. Feed prefix_messages as context to the model under test
2. Enter agent loop: model output -> tool simulator response -> continue
3. Score model output per-step using an LLM judge
4. Write results to results.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running from any directory by adding this file's dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI

from simulator import ToolSimulator
from judge import judge_step


# ── Logging ──────────────────────────────────────────────────

def log(msg: str, tp_id: str = ""):
    """Print a timestamped log line to stderr."""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if tp_id:
        prefix += f" [{tp_id}]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


# ── Progress Tracker ──────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe progress tracker for concurrent test points."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = asyncio.Lock()

    async def mark_done(self, success: bool = True):
        async with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1

    @property
    def progress_str(self) -> str:
        return f"[{self.completed}/{self.total}]"


# ── Retry helper ──────────────────────────────────────────────────

async def api_call_with_retry(coro_fn, max_retries=5, base_delay=2.0, tp_id=""):
    """Retry on 429 rate-limit errors with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log(f"Rate limited, retry {attempt+1}/{max_retries} in {delay:.0f}s...", tp_id)
                await asyncio.sleep(delay)
                continue
            raise


# ── Agent Loop (async) ──────────────────────────────────────────────

async def run_agent_loop(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    simulator: ToolSimulator,
    max_turns: int = 10,
    max_tool_failures: int = 3,
    tp_id: str = "",
    target_subtask_re: re.Pattern | None = None,
    target_subtask_num: int | None = None,
) -> tuple[list[dict], str, str]:
    """Run the agent loop; returns (all_new_messages, final_text_output, exit_reason).

    exit_reason: "completed" | "tool_failure" | "max_turns" | "api_error" | "next_subtask"

    If target_subtask_re and target_subtask_num are provided, the loop exits early
    when the model starts working on a different subtask (reads a file matching a
    subtask number != target_subtask_num).
    """
    new_messages = []
    final_text = ""
    exit_reason = "max_turns"
    nudge_count = 0
    max_nudges = 2  # max retry nudges if model refuses to use tools

    wrap_up_injected = False  # whether we've injected the "stop calling tools" message

    for turn in range(max_turns):
        log(f"Agent turn {turn+1}/{max_turns} — calling model...", tp_id)

        # Near the turn limit: inject a wrap-up message and disable tools
        use_tools = tools
        if not wrap_up_injected and turn >= max_turns - 2:
            wrap_up_msg = {
                "role": "user",
                "content": (
                    "你即将达到最大交互轮次限制。请不要再调用任何工具，"
                    "直接在回复中输出你目前的评估结论和完整结果。"
                ),
            }
            messages.append(wrap_up_msg)
            new_messages.append(wrap_up_msg)
            wrap_up_injected = True
            use_tools = None
            log(f"  Injected wrap-up message (turn limit approaching)", tp_id)

        try:
            response = await api_call_with_retry(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=use_tools if use_tools else None,
                    tool_choice="auto" if use_tools else None,
                    max_tokens=8192,
                ),
                tp_id=tp_id,
            )
        except Exception as e:
            error_detail = str(e)
            if hasattr(e, 'body'):
                error_detail += f" | body: {e.body}"
            log(f"API error (after retries): {error_detail}", tp_id)
            exit_reason = "api_error"
            break

        if not response.choices:
            log(f"API returned empty choices, skipping", tp_id)
            exit_reason = "api_error"
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        if not assistant_msg:
            log(f"API returned empty message, skipping", tp_id)
            exit_reason = "api_error"
            break

        # Log token usage
        usage = response.usage
        if usage:
            log(f"  tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}", tp_id)

        # Build assistant message dict
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
            content_preview = assistant_msg.content[:100].replace("\n", " ")
            log(f"  output: {len(assistant_msg.content)} chars — \"{content_preview}...\"", tp_id)
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
            tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
            log(f"  tool_calls: {tool_names}", tp_id)

        messages.append(msg_dict)
        new_messages.append(msg_dict)

        # No tool calls -> check if we should nudge or end
        if not assistant_msg.tool_calls:
            if nudge_count < max_nudges and simulator.call_count == 0:
                # Model refused to use tools on early turns — inject a nudge
                nudge_count += 1
                nudge_msg = {
                    "role": "user",
                    "content": (
                        "请不要总结已完成的内容。你需要继续执行下一个子任务。"
                        "使用工具（Read、Edit、Write等）来实际完成工作，而不是描述你会做什么。"
                        "请立即开始。"
                    ),
                }
                messages.append(nudge_msg)
                new_messages.append(nudge_msg)
                log(f"  No tool calls yet — injecting nudge ({nudge_count}/{max_nudges})", tp_id)
                continue
            final_text = assistant_msg.content or ""
            exit_reason = "completed"
            log(f"  finish_reason={choice.finish_reason}, ending agent loop", tp_id)
            break

        # Process tool calls
        tool_failure_exit = False
        next_subtask_exit = False
        for tc in assistant_msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Detect if the model is starting a different subtask
            if (target_subtask_re and target_subtask_num is not None
                    and tc.function.name == "Read"):
                fp = args.get("file_path", "")
                m_file = target_subtask_re.search(fp)
                if m_file:
                    num = int(m_file.group(1))
                    if num != target_subtask_num:
                        log(f"  Early exit: model started subtask {num} "
                            f"(target was {target_subtask_num})", tp_id)
                        next_subtask_exit = True
                        break

            result = simulator.simulate(tc.function.name, args)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            messages.append(tool_msg)
            new_messages.append(tool_msg)

            if simulator.consecutive_fallbacks >= max_tool_failures:
                log(f"  Early exit: {max_tool_failures} consecutive tool call failures", tp_id)
                tool_failure_exit = True
                break

        if next_subtask_exit:
            exit_reason = "next_subtask"
            break

        if tool_failure_exit:
            exit_reason = "tool_failure"
            break

        # If finish_reason is stop, also end the loop
        if choice.finish_reason == "stop":
            final_text = assistant_msg.content or ""
            exit_reason = "completed"
            log(f"  finish_reason=stop, ending agent loop", tp_id)
            break

    return new_messages, final_text, exit_reason


# ── Step segmentation config ──────────────────────────────────────────────────

# Subtask file patterns for detecting step boundaries per case
STEP_FILE_PATTERNS = {
    "resume-screening-1": re.compile(r"resume_(\d{2})\.md"),
    "competitive-analysis-1": re.compile(r"competitor_(\d+)"),
    "stock-research-1": re.compile(r"question_(\d+)"),
}

# Subtask ID format strings
STEP_ID_FORMATS = {
    "resume-screening-1": "R{:02d}",
    "competitive-analysis-1": "C{:02d}",
    "stock-research-1": "Q{:02d}",
}

# Patterns for detecting subtask mentions in assistant text
STEP_TEXT_PATTERNS = {
    "resume-screening-1": re.compile(r"(?:R|resume[_\s]?)(\d{2})"),
    "competitive-analysis-1": re.compile(r"C(\d{2})"),
    "stock-research-1": re.compile(r"Q(\d{1,2})"),
}


def segment_trajectory_by_steps(
    new_messages: list[dict],
    case_id: str,
    start_subtask_num: int = 1,
) -> list[dict]:
    """
    Segment agent loop messages by subtask.
    Returns [{"subtask_id": "R03", "subtask_num": 3, "messages": [...], "text": "..."}, ...]

    Step boundary detection priority:
    1. Read tool call file_path matching the subtask file pattern
    2. Assistant text mentioning a new subtask ID (only when num >= start_subtask_num)
    """
    file_pattern = STEP_FILE_PATTERNS.get(case_id)
    text_pattern = STEP_TEXT_PATTERNS.get(case_id)
    id_format = STEP_ID_FORMATS.get(case_id, "S{:02d}")

    segments: list[dict] = []
    current_num: int | None = None
    current_msgs: list[dict] = []
    seen_nums: set[int] = set()

    def _flush():
        nonlocal current_msgs
        if current_num is not None and current_msgs:
            # Extract all assistant text in this step (strip <think> blocks)
            text_parts = []
            for m in current_msgs:
                if m["role"] == "assistant" and m.get("content"):
                    cleaned = re.sub(r"<think>.*?</think>\s*", "", m["content"], flags=re.DOTALL).strip()
                    if cleaned:
                        text_parts.append(cleaned)
            segments.append({
                "subtask_id": id_format.format(current_num),
                "subtask_num": current_num,
                "messages": current_msgs,
                "text": "\n\n".join(text_parts),
            })
        current_msgs = []

    for msg in new_messages:
        detected_num = None

        if msg["role"] == "assistant":
            # Check tool call file paths (most reliable signal)
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                if func.get("name") == "Read":
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        continue
                    fp = args.get("file_path", "")
                    if file_pattern:
                        m = file_pattern.search(fp)
                        if m:
                            num = int(m.group(1))
                            if num >= start_subtask_num and num not in seen_nums:
                                detected_num = num
                                break

            # If file path detection missed, look for first new subtask mention in text
            if detected_num is None and text_pattern and msg.get("content"):
                content = msg["content"][:300]
                matches = text_pattern.findall(content)
                for match_str in matches:
                    num = int(match_str)
                    if num >= start_subtask_num and num not in seen_nums:
                        detected_num = num
                        break

        if detected_num is not None:
            _flush()
            current_num = detected_num
            seen_nums.add(detected_num)

        current_msgs.append(msg)

    _flush()
    return segments


# ── Single test point handler ──────────────────────────────────────────────────

async def run_single_test_point(
    idx: int,
    total: int,
    tp: dict,
    client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    model: str,
    judge_model: str,
    max_turns: int,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: Path,
    traj_path: Path,
    progress: ProgressTracker,
) -> dict:
    """Process a single test point: agent loop + per-step judge scoring."""
    async with semaphore:
        tp_id = tp["id"]
        # Use full tool_responses_pool if available, fall back to legacy tool_responses
        pool = tp.get("tool_responses_pool", tp["tool_responses"])
        cut_pos = tp.get("cut_position", 0)
        known_paths = tp.get("known_file_paths", [])

        log(f"START — case={tp['case_id']} point={tp['test_point']} cond={tp['condition']} "
            f"prefix={len(tp['prefix_messages'])} msgs, pool={len(pool)} tool_resp, "
            f"cut_pos={cut_pos}, known_files={len(known_paths)}", tp_id)

        simulator = ToolSimulator(pool, cut_position=cut_pos, known_file_paths=known_paths)
        messages = list(tp["prefix_messages"])

        # Append a continuation prompt telling the model to complete exactly
        # ONE subtask.  Each test point targets a single subtask; the context-rot
        # signal comes from comparing scores across test points with increasing
        # prefix length, not from doing many subtasks in one run.
        #
        # The target file content is pre-loaded so the model doesn't depend on
        # simulator Read matching.  This levels the playing field between models
        # that rely on tools heavily vs. those that don't.
        subtask_id = tp["subtask_id"]
        target_content = tp.get("target_content", "")

        continuation_parts = [
            f"请完成子任务 {subtask_id} 的完整评估。",
        ]
        if target_content:
            continuation_parts.append(
                f"\n以下是 {subtask_id} 的待评估内容：\n\n{target_content}\n"
            )
        continuation_parts.append(
            "在输出中展示完整的评估过程（每个步骤的具体内容和结论）。"
            f"只需完成 {subtask_id}，完成后输出结论即可。"
        )
        continuation_msg = {
            "role": "user",
            "content": "".join(continuation_parts),
        }
        messages.append(continuation_msg)

        # ── Query padding injection (distance experiment) ──
        # If the test point has query_padding_messages, inject them after the
        # continuation prompt.  These dummy tool-call rounds push the user query
        # further from the model's generation position, allowing us to measure
        # the effect of user-query distance independently of SP distance.
        query_padding = tp.get("query_padding_messages", [])
        if query_padding:
            log(f"Injecting {len(query_padding)} query-padding messages", tp_id)
            messages.extend(query_padding)

        # ── Phase 1: Agent Loop ──
        # Build subtask boundary detector for early exit
        file_pattern = STEP_FILE_PATTERNS.get(tp["case_id"])
        target_num_match = re.search(r'\d+', subtask_id)
        target_num = int(target_num_match.group()) if target_num_match else None

        t0 = time.time()
        new_messages, final_text, exit_reason = await run_agent_loop(
            client=client,
            model=model,
            messages=messages,
            tools=tp["tools"],
            simulator=simulator,
            max_turns=max_turns,
            tp_id=tp_id,
            target_subtask_re=file_pattern,
            target_subtask_num=target_num,
        )
        agent_elapsed = time.time() - t0

        log(f"Agent done — {len(new_messages)} msgs, {len(final_text)} chars, "
            f"{simulator.call_count} tool calls ({simulator.match_summary}), "
            f"exit={exit_reason}, {agent_elapsed:.1f}s", tp_id)

        # ── Phase 2: Judge the single target subtask ──
        t1 = time.time()

        # Collect all model-generated content for judging.
        # Two sources: (1) assistant message text, (2) content written via
        # Edit/Write tool calls (new_string / content args).  Some models
        # output evaluation in text, others write it to files — we capture both.
        text_parts = []
        for m in new_messages:
            if m["role"] == "assistant":
                content = m.get("content", "")
                if content:
                    cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                    if cleaned:
                        text_parts.append(cleaned)
                # Extract substantive content from Edit/Write tool call args
                for tc in m.get("tool_calls", []):
                    func = tc.get("function", {})
                    fname = func.get("name", "")
                    if fname in ("Edit", "Write"):
                        try:
                            tc_args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            continue
                        written = tc_args.get("new_string", "") or tc_args.get("content", "")
                        # Only include substantial content (skip short edits like status updates)
                        if written and len(written) > 200:
                            text_parts.append(written)

        all_model_text = "\n\n".join(text_parts).strip()

        # Pick the reference for the target subtask
        step_refs_map = {r["subtask_id"]: r["reference_output"]
                         for r in tp.get("step_references", [])}
        ref = step_refs_map.get(subtask_id, tp["reference_output"])

        # Judge
        if all_model_text:
            judge_result = await judge_step(
                judge_client, judge_model,
                tp["case_id"], tp["eval_criteria"],
                all_model_text, ref, subtask_id,
                tp_id=tp_id, log_fn=log, retry_fn=api_call_with_retry,
            )
        else:
            judge_result = {"scores": {}, "overall": 0.0, "justification": "No output"}

        overall = judge_result["overall"]

        judge_elapsed = time.time() - t1
        total_elapsed = time.time() - t0

        # ── Phase 3: Write result ──
        success = overall > 0
        await progress.mark_done(success=success)

        log(f"DONE {progress.progress_str} — overall={overall:.3f} "
            f"scores={judge_result.get('scores', {})} "
            f"exit={exit_reason} "
            f"agent={agent_elapsed:.1f}s judge={judge_elapsed:.1f}s total={total_elapsed:.1f}s",
            tp_id)

        result = {
            "id": tp_id,
            "model": model,
            "case_index": tp["case_index"],
            "case_id": tp["case_id"],
            "test_point": tp["test_point"],
            "condition": tp["condition"],
            "subtask_id": tp["subtask_id"],
            "n_prior_subtasks": tp["n_prior_subtasks"],
            "prefix_msg_count": len(tp["prefix_messages"]),
            "response_length": len(all_model_text),
            "tool_calls_made": simulator.call_count,
            "tool_match_summary": simulator.match_summary,
            "elapsed_seconds": round(total_elapsed, 1),
            "exit_reason": exit_reason,
            "scores": judge_result.get("scores", {}),
            "justification": judge_result.get("justification", ""),
            "overall": overall,
            "response": all_model_text[:5000],
        }

        # Trajectory record: save all messages generated by the agent loop
        traj_record = {
            "id": tp_id,
            "model": model,
            "case_id": tp["case_id"],
            "test_point": tp["test_point"],
            "condition": tp["condition"],
            "subtask_id": tp["subtask_id"],
            "n_prior_subtasks": tp["n_prior_subtasks"],
            "prefix_msg_count": len(tp["prefix_messages"]),
            "trajectory": new_messages,
        }

        async with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            with open(traj_path, "a") as f:
                f.write(json.dumps(traj_record, ensure_ascii=False) + "\n")

        return result


# ── Main ──────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(description="Run Context Rot evaluation")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model")
    parser.add_argument("--judge-api-base", default=None, help="Judge API base URL (if different from --api-base)")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key (if different from --api-key)")
    parser.add_argument("--judge-extra-headers", default=None, help="Judge extra headers as JSON string")
    parser.add_argument("--api-base", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--input", required=True, help="Input eval_set.jsonl path")
    parser.add_argument("--output", required=True, help="Output results.jsonl path")
    parser.add_argument("--max-turns", type=int, default=128, help="Max agent loop turns")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N test points")
    parser.add_argument("--filter-id", default=None, help="Only run test points matching this ID prefix")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent test points")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    traj_path = output_path.parent / "trajectories.jsonl"

    # Print configuration
    print("=" * 70, file=sys.stderr)
    print("Context Rot Evaluation", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Model:        {args.model}", file=sys.stderr)
    print(f"  Judge:        {args.judge_model}", file=sys.stderr)
    print(f"  API base:     {args.api_base}", file=sys.stderr)
    print(f"  Judge base:   {args.judge_api_base or args.api_base}", file=sys.stderr)
    print(f"  Input:        {input_path}", file=sys.stderr)
    print(f"  Output:       {output_path}", file=sys.stderr)
    print(f"  Trajectories: {traj_path}", file=sys.stderr)
    print(f"  Concurrency:  {args.concurrency}", file=sys.stderr)
    print(f"  Max turns:    {args.max_turns}", file=sys.stderr)
    print(f"  Started at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print("=" * 70, file=sys.stderr, flush=True)

    # Initialize async model client
    client_kwargs = {}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = AsyncOpenAI(**client_kwargs)

    # Initialize judge client (independently configurable)
    judge_kwargs = {}
    judge_kwargs["base_url"] = args.judge_api_base or args.api_base
    judge_kwargs["api_key"] = args.judge_api_key or args.api_key
    if args.judge_extra_headers:
        judge_kwargs["default_headers"] = json.loads(args.judge_extra_headers)
    judge_kwargs = {k: v for k, v in judge_kwargs.items() if v is not None}
    judge_client = AsyncOpenAI(**judge_kwargs)

    # Load test points
    with open(input_path) as f:
        test_points = [json.loads(line) for line in f]

    if args.filter_id:
        test_points = [tp for tp in test_points if tp["id"].startswith(args.filter_id)]
    if args.limit:
        test_points = test_points[:args.limit]

    total = len(test_points)

    # Print test point distribution
    by_case = defaultdict(list)
    by_cond = defaultdict(int)
    for tp in test_points:
        by_case[tp["case_id"]].append(tp["test_point"])
        by_cond[tp["condition"]] += 1

    print(f"\nLoaded {total} test points:", file=sys.stderr)
    for case_id in sorted(by_case):
        points = sorted(by_case[case_id])
        print(f"  {case_id}: {', '.join(points)}", file=sys.stderr)
    print(f"  Conditions: {dict(by_cond)}", file=sys.stderr)
    print(f"\n{'─' * 70}", file=sys.stderr, flush=True)

    progress = ProgressTracker(total)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    t_start = time.time()

    tasks = [
        run_single_test_point(
            idx=idx,
            total=total,
            tp=tp,
            client=client,
            judge_client=judge_client,
            model=args.model,
            judge_model=args.judge_model,
            max_turns=args.max_turns,
            semaphore=semaphore,
            write_lock=write_lock,
            output_path=output_path,
            traj_path=traj_path,
            progress=progress,
        )
        for idx, tp in enumerate(test_points)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Final summary ──
    successful = []
    errors = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append((i, r))
        else:
            successful.append(r)

    t_total = time.time() - t_start

    print(f"\n{'=' * 70}", file=sys.stderr)
    print("Evaluation Complete", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)
    print(f"  Finished at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"  Duration:     {t_total:.1f}s ({t_total/60:.1f}min)", file=sys.stderr)
    print(f"  Results:      {len(successful)}/{total} succeeded, {len(errors)} failed", file=sys.stderr)
    print(f"  Output:       {output_path}", file=sys.stderr)
    print(f"  Trajectories: {traj_path}", file=sys.stderr)

    if successful:
        scores = [r["overall"] for r in successful]
        avg = sum(scores) / len(scores)
        print(f"  Avg score:    {avg:.3f} (min={min(scores):.3f}, max={max(scores):.3f})", file=sys.stderr)

        # Aggregate by test point (P1-P5) — this IS the quality curve
        by_point: dict[str, list[float]] = defaultdict(list)
        for r in successful:
            by_point[r["test_point"]].append(r["overall"])
        print(f"\n  Score by test point (quality curve):", file=sys.stderr)
        for p in sorted(by_point):
            vals = by_point[p]
            print(f"    {p}: {sum(vals)/len(vals):.3f} (n={len(vals)})", file=sys.stderr)

        # Aggregate by subtask_id for finer granularity
        by_subtask: dict[str, list[float]] = defaultdict(list)
        for r in successful:
            by_subtask[r["subtask_id"]].append(r["overall"])
        if len(by_subtask) > 1:
            print(f"\n  Score by subtask:", file=sys.stderr)
            for sid in sorted(by_subtask):
                vals = by_subtask[sid]
                print(f"    {sid}: {sum(vals)/len(vals):.3f} (n={len(vals)})", file=sys.stderr)

        # Aggregate by case
        by_case_score: dict[str, list[float]] = defaultdict(list)
        for r in successful:
            by_case_score[r["case_id"]].append(r["overall"])
        if len(by_case_score) > 1:
            print(f"\n  Score by case:", file=sys.stderr)
            for cid in sorted(by_case_score):
                vals = by_case_score[cid]
                print(f"    {cid}: {sum(vals)/len(vals):.3f} (n={len(vals)})", file=sys.stderr)

        # Exit reason statistics
        exit_reasons = defaultdict(int)
        for r in successful:
            exit_reasons[r.get("exit_reason", "unknown")] += 1
        print(f"\n  Exit reasons: {dict(exit_reasons)}", file=sys.stderr)

    if errors:
        print(f"\n  Errors:", file=sys.stderr)
        for i, e in errors:
            print(f"    test_point[{i}]: {e}", file=sys.stderr)

    print(f"{'=' * 70}", file=sys.stderr, flush=True)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
