from typing import Optional, Tuple
import logging

import re
try:
    from math_verify import parse, verify
except ImportError:
    raise ImportError(
        "math_verify is required for math evaluation but not installed. "
        "Falling back to string comparison is unsafe (e.g. '\\left(' vs '(' are mathematically equivalent but fail string match). "
        "Please install it: pip install math_verify"
    )

from nanoeval.reward.extract import extract_answer

logger = logging.getLogger(__name__)

def grade_answer(solution_str: str, ground_truth: str) -> Tuple[float, float]:
    try:
        ground_truth = parse(ground_truth)
        solution = parse(solution_str)
        if verify(ground_truth, solution):
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    except Exception as e:
        logger.error("grade_answer failed: %s", e)
        return 0.0, 0.0


def math_judge(
    response: str,
    label: str = "",
    **kwargs
) -> dict:
    raw_eval_res = response
    pred_ans = extract_answer(raw_eval_res)
    
    if not pred_ans:
        return {
            "pred": pred_ans,
            "pass": False
        }
    
    if pred_ans == label:
        return {
            "pred": pred_ans,
            "pass": True
        }
    else:
        score, _ = grade_answer(f"${pred_ans}$", f"${label}$")
        return {
            "pred": pred_ans,
            "pass": True if score == 1.0 else False
        }