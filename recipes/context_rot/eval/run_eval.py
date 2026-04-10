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
import logging
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

logger = logging.getLogger(__name__)


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
                logger.warning("Rate limited, retry %d/%d in %.0fs... [%s]",
                               attempt + 1, max_retries, delay, tp_id)
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
    """
    new_messages = []
    final_text = ""
    exit_reason = "max_turns"
    nudge_count = 0
    max_nudges = 2

    wrap_up_injected = False

    for turn in range(max_turns):
        logger.info("Agent turn %d/%d — calling model... [%s]", turn + 1, max_turns, tp_id)

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
            logger.info("  Injected wrap-up message (turn limit approaching) [%s]", tp_id)

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
            logger.error("API error (after retries): %s [%s]", error_detail, tp_id)
            exit_reason = "api_error"
            break

        if not response.choices:
            logger.warning("API returned empty choices, skipping [%s]", tp_id)
            exit_reason = "api_error"
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        if not assistant_msg:
            logger.warning("API returned empty message, skipping [%s]", tp_id)
            exit_reason = "api_error"
            break

        # Log token usage
        usage = response.usage
        if usage:
            logger.info("  tokens: prompt=%d completion=%d total=%d [%s]",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, tp_id)

        # Build assistant message dict
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
            content_preview = assistant_msg.content[:100].replace("\n", " ")
            logger.info('  output: %d chars — "%s..." [%s]',
                        len(assistant_msg.content), content_preview, tp_id)
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
            logger.info("  tool_calls: %s [%s]", tool_names, tp_id)

        messages.append(msg_dict)
        new_messages.append(msg_dict)

        # No tool calls -> check if we should nudge or end
        if not assistant_msg.tool_calls:
            if nudge_count < max_nudges and simulator.call_count == 0:
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
                logger.info("  No tool calls yet — injecting nudge (%d/%d) [%s]",
                            nudge_count, max_nudges, tp_id)
                continue
            final_text = assistant_msg.content or ""
            exit_reason = "completed"
            logger.info("  finish_reason=%s, ending agent loop [%s]", choice.finish_reason, tp_id)
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
                        logger.info("  Early exit: model started subtask %d "
                                    "(target was %d) [%s]", num, target_subtask_num, tp_id)
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
                logger.info("  Early exit: %d consecutive tool call failures [%s]",
                            max_tool_failures, tp_id)
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
            logger.info("  finish_reason=stop, ending agent loop [%s]", tp_id)
            break

    return new_messages, final_text, exit_reason


# ── Step segmentation config ──────────────────────────────────────────────────

STEP_FILE_PATTERNS = {
    "resume-screening-1": re.compile(r"resume_(\d{2})\.md"),
    "competitive-analysis-1": re.compile(r"competitor_(\d+)"),
    "stock-research-1": re.compile(r"question_(\d+)"),
}

STEP_ID_FORMATS = {
    "resume-screening-1": "R{:02d}",
    "competitive-analysis-1": "C{:02d}",
    "stock-research-1": "Q{:02d}",
}

STEP_TEXT_PATTERNS = {
    "resume-screening-1": re.compile(r"(?:R|resume[_\s]?)(\d{2})"),
    "competitive-analysis-1": re.compile(r"C(\d{2})"),
    "stock-research-1": re.compile(r"Q(\d{1,2})"),
}


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
        pool = tp.get("tool_responses_pool", tp["tool_responses"])
        cut_pos = tp.get("cut_position", 0)
        known_paths = tp.get("known_file_paths", [])

        logger.info("START — case=%s point=%s cond=%s prefix=%d msgs, pool=%d tool_resp, "
                     "cut_pos=%d, known_files=%d [%s]",
                     tp['case_id'], tp['test_point'], tp['condition'],
                     len(tp['prefix_messages']), len(pool), cut_pos, len(known_paths), tp_id)

        simulator = ToolSimulator(pool, cut_position=cut_pos, known_file_paths=known_paths)
        messages = list(tp["prefix_messages"])

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

        query_padding = tp.get("query_padding_messages", [])
        if query_padding:
            logger.info("Injecting %d query-padding messages [%s]", len(query_padding), tp_id)
            messages.extend(query_padding)

        # ── Phase 1: Agent Loop ──
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

        logger.info("Agent done — %d msgs, %d chars, %d tool calls (%s), exit=%s, %.1fs [%s]",
                     len(new_messages), len(final_text), simulator.call_count,
                     simulator.match_summary, exit_reason, agent_elapsed, tp_id)

        # ── Phase 2: Judge the single target subtask ──
        t1 = time.time()

        text_parts = []
        for m in new_messages:
            if m["role"] == "assistant":
                content = m.get("content", "")
                if content:
                    cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                    if cleaned:
                        text_parts.append(cleaned)
                for tc in m.get("tool_calls", []):
                    func = tc.get("function", {})
                    fname = func.get("name", "")
                    if fname in ("Edit", "Write"):
                        try:
                            tc_args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            continue
                        written = tc_args.get("new_string", "") or tc_args.get("content", "")
                        if written and len(written) > 200:
                            text_parts.append(written)

        all_model_text = "\n\n".join(text_parts).strip()

        step_refs_map = {r["subtask_id"]: r["reference_output"]
                         for r in tp.get("step_references", [])}
        ref = step_refs_map.get(subtask_id, tp["reference_output"])

        # Judge — pass logger-compatible functions
        def _log_fn(msg, tp_id_inner=""):
            logger.info("%s [%s]", msg, tp_id_inner)

        if all_model_text:
            judge_result = await judge_step(
                judge_client, judge_model,
                tp["case_id"], tp["eval_criteria"],
                all_model_text, ref, subtask_id,
                tp_id=tp_id, log_fn=_log_fn, retry_fn=api_call_with_retry,
            )
        else:
            judge_result = {"scores": {}, "overall": 0.0, "justification": "No output"}

        overall = judge_result["overall"]

        judge_elapsed = time.time() - t1
        total_elapsed = time.time() - t0

        # ── Phase 3: Write result ──
        success = overall > 0
        await progress.mark_done(success=success)

        logger.info("DONE %s — overall=%.3f scores=%s exit=%s "
                     "agent=%.1fs judge=%.1fs total=%.1fs [%s]",
                     progress.progress_str, overall, judge_result.get('scores', {}),
                     exit_reason, agent_elapsed, judge_elapsed, total_elapsed, tp_id)

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
    logger.info("=" * 70)
    logger.info("Context Rot Evaluation")
    logger.info("=" * 70)
    logger.info("  Model:        %s", args.model)
    logger.info("  Judge:        %s", args.judge_model)
    logger.info("  API base:     %s", args.api_base)
    logger.info("  Judge base:   %s", args.judge_api_base or args.api_base)
    logger.info("  Input:        %s", input_path)
    logger.info("  Output:       %s", output_path)
    logger.info("  Trajectories: %s", traj_path)
    logger.info("  Concurrency:  %s", args.concurrency)
    logger.info("  Max turns:    %s", args.max_turns)
    logger.info("  Started at:   %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 70)

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

    by_case = defaultdict(list)
    by_cond = defaultdict(int)
    for tp in test_points:
        by_case[tp["case_id"]].append(tp["test_point"])
        by_cond[tp["condition"]] += 1

    logger.info("Loaded %d test points:", total)
    for case_id in sorted(by_case):
        points = sorted(by_case[case_id])
        logger.info("  %s: %s", case_id, ', '.join(points))
    logger.info("  Conditions: %s", dict(by_cond))

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

    logger.info("=" * 70)
    logger.info("Evaluation Complete")
    logger.info("=" * 70)
    logger.info("  Finished at:  %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("  Duration:     %.1fs (%.1fmin)", t_total, t_total / 60)
    logger.info("  Results:      %d/%d succeeded, %d failed", len(successful), total, len(errors))
    logger.info("  Output:       %s", output_path)
    logger.info("  Trajectories: %s", traj_path)

    if successful:
        scores = [r["overall"] for r in successful]
        avg = sum(scores) / len(scores)
        logger.info("  Avg score:    %.3f (min=%.3f, max=%.3f)", avg, min(scores), max(scores))

        by_point: dict[str, list[float]] = defaultdict(list)
        for r in successful:
            by_point[r["test_point"]].append(r["overall"])
        logger.info("  Score by test point (quality curve):")
        for p in sorted(by_point):
            vals = by_point[p]
            logger.info("    %s: %.3f (n=%d)", p, sum(vals) / len(vals), len(vals))

        by_case_score: dict[str, list[float]] = defaultdict(list)
        for r in successful:
            by_case_score[r["case_id"]].append(r["overall"])
        if len(by_case_score) > 1:
            logger.info("  Score by case:")
            for cid in sorted(by_case_score):
                vals = by_case_score[cid]
                logger.info("    %s: %.3f (n=%d)", cid, sum(vals) / len(vals), len(vals))

        exit_reasons = defaultdict(int)
        for r in successful:
            exit_reasons[r.get("exit_reason", "unknown")] += 1
        logger.info("  Exit reasons: %s", dict(exit_reasons))

    if errors:
        logger.error("  Errors:")
        for i, e in errors:
            logger.error("    test_point[%d]: %s", i, e)

    logger.info("=" * 70)


def main():
    from nanoeval.utils.logging_utils import configure_logger
    configure_logger()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
