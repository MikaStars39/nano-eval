#!/usr/bin/env python3
"""
judge.py — LLM Judge scoring module

Scores model output on multiple dimensions using an LLM judge,
with automatic normalization of various response formats.
"""

import json
import re

from openai import AsyncOpenAI


JUDGE_SYSTEM = """你是一个严格的输出质量评估专家。你需要根据给定的评估标准，对模型输出进行评分。

评分规则：
- 每个标准打 0.0 到 1.0 分（0.0=完全不符合, 0.5=部分符合, 1.0=完全符合）
- 必须基于实际输出内容评分，不能因为模型"尝试了"就给高分
- 与参考输出对比，评估质量是否相当

你必须严格按照以下 JSON 格式输出，不要添加任何其他内容：

```json
{
  "scores": {
    "criterion_name_1": 0.8,
    "criterion_name_2": 0.5
  },
  "overall": 0.65,
  "justification": "简要说明评分理由"
}
```

注意：
- scores 中的 key 必须与评估标准的 name 完全一致
- 所有分数必须是 0.0 到 1.0 之间的浮点数
- overall 是所有标准的加权平均
- 只输出 JSON，不要输出其他文字"""

JUDGE_USER_TEMPLATE = """## 任务类型
{case_id}

## 评估标准
{criteria_text}

## 参考输出（高质量示例）
{reference}

## 待评估的模型输出
{response}

请按标准逐项评分，输出 JSON。"""


def _normalize_scores(result: dict) -> dict:
    """
    Normalize various judge response formats to a standard format:
    {"scores": {"name": float 0-1}, "overall": float 0-1, "justification": str}

    Supported input formats:
    1. Standard: {"scores": {"name": 0.8}, "overall": 0.75, ...}
    2. Non-standard: {"name": {"score": 6, "max_score": 10, "assessment": "..."}, ...}
    3. Simple: {"name": 0.8, ...} (no outer "scores" wrapper)
    """
    # Keys that represent aggregate scores, not individual criteria
    OVERALL_KEYS = {"overall", "overall_score", "total", "total_score", "average"}

    # Already in standard format
    if "scores" in result and isinstance(result["scores"], dict):
        # Ensure score values are floats; remove any overall-like keys that leaked in
        for k, v in list(result["scores"].items()):
            if k in OVERALL_KEYS:
                del result["scores"][k]
                continue
            if isinstance(v, dict):
                raw = v.get("score", v.get("value", 0))
                max_s = v.get("max_score", v.get("max", 10))
                if isinstance(raw, (int, float)) and isinstance(max_s, (int, float)) and max_s > 0:
                    result["scores"][k] = float(raw) / float(max_s) if raw > 1 else float(raw)
                else:
                    result["scores"][k] = 0.0
            else:
                result["scores"][k] = float(v)
    else:
        # Non-standard format: attempt to parse
        parsed_scores = {}
        justification_parts = []
        remaining = {}
        for k, v in result.items():
            if k in OVERALL_KEYS or k == "justification":
                remaining[k] = v
            elif isinstance(v, dict) and ("score" in v or "value" in v):
                raw = v.get("score", v.get("value", 0))
                max_s = v.get("max_score", v.get("max", 10))
                if isinstance(raw, (int, float)) and isinstance(max_s, (int, float)) and max_s > 0:
                    parsed_scores[k] = float(raw) / float(max_s) if raw > 1 else float(raw)
                else:
                    parsed_scores[k] = float(raw) if isinstance(raw, (int, float)) else 0.0
                if "assessment" in v:
                    justification_parts.append(f"{k}: {v['assessment']}")
            elif isinstance(v, (int, float)):
                parsed_scores[k] = float(v) / 10.0 if v > 1 else float(v)
        if parsed_scores:
            # Pick the best overall candidate from any overall-like key
            raw_overall = None
            for ok in OVERALL_KEYS:
                if ok in remaining and remaining[ok] is not None:
                    raw_overall = remaining[ok]
                    break
            result = {
                "scores": parsed_scores,
                "overall": raw_overall,
                "justification": remaining.get("justification", "; ".join(justification_parts)),
            }

    # Normalize overall: always recompute from scores average
    # The judge's self-reported overall is unreliable (wrong scale, placeholder 0, etc.)
    scores = result.get("scores", {})
    if scores:
        overall = sum(scores.values()) / len(scores)
    else:
        overall = 0.0
    result["overall"] = overall

    if "justification" not in result:
        result["justification"] = ""

    return result


async def judge_response(
    client: AsyncOpenAI,
    judge_model: str,
    test_point: dict,
    model_output: str,
    tp_id: str = "",
    log_fn=None,
    retry_fn=None,
) -> dict:
    """
    Evaluate model output using an LLM judge; returns normalized scores dict.

    Args:
        client: AsyncOpenAI client
        judge_model: judge model name
        test_point: test point data (must contain eval_criteria, reference_output, case_id)
        model_output: model output text to evaluate
        tp_id: test point ID (for logging)
        log_fn: logging function log(msg, tp_id)
        retry_fn: retry function api_call_with_retry(coro_fn, tp_id=tp_id)
    """
    def _log(msg):
        if log_fn:
            log_fn(msg, tp_id)

    criteria_text = "\n".join(
        f"- **{c['name']}**: {c['description']}"
        for c in test_point["eval_criteria"]
    )

    user_msg = JUDGE_USER_TEMPLATE.format(
        case_id=test_point["case_id"],
        criteria_text=criteria_text,
        reference=test_point["reference_output"][:4000],
        response=model_output[:6000],
    )

    _log(f"Judging with {judge_model}...")
    try:
        if retry_fn:
            response = await retry_fn(
                lambda: client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=1024,
                ),
                tp_id=tp_id,
            )
        else:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
            )

        content = response.choices[0].message.content or "{}"
        _log(f"Judge raw response ({len(content)} chars): {content[:300]}")

        # Extract JSON
        if "```json" in content:
            match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1)

        result = json.loads(content)
        result = _normalize_scores(result)

        _log(f"Judge parsed: overall={result['overall']:.3f}, scores={result.get('scores', {})}")
        return result

    except Exception as e:
        error_detail = str(e)
        if hasattr(e, 'response'):
            try:
                error_detail += f" | body: {e.response.text[:500]}"
            except Exception:
                pass
        if hasattr(e, 'body'):
            error_detail += f" | body: {e.body}"
        _log(f"Judge error: {error_detail}")
        return {"scores": {}, "overall": 0.0, "justification": f"Judge error: {error_detail}"}


async def judge_step(
    client: AsyncOpenAI,
    judge_model: str,
    case_id: str,
    eval_criteria: list[dict],
    step_output: str,
    step_reference: str,
    step_id: str,
    tp_id: str = "",
    log_fn=None,
    retry_fn=None,
) -> dict:
    """
    Evaluate a single step's output quality; returns normalized scores dict.

    Uses the same prompt and normalization logic as judge_response,
    but accepts a single step's output and reference instead of a full test point.
    """
    # Build a lightweight test_point dict for judge_response
    pseudo_tp = {
        "case_id": case_id,
        "eval_criteria": eval_criteria,
        "reference_output": step_reference,
    }
    return await judge_response(
        client=client,
        judge_model=judge_model,
        test_point=pseudo_tp,
        model_output=step_output,
        tp_id=f"{tp_id}/{step_id}",
        log_fn=log_fn,
        retry_fn=retry_fn,
    )
