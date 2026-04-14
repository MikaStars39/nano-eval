from typing import Optional

from nanoeval.reward.extract import extract_answer

def gpqa_judge(
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
        pred_normalized = pred_ans.strip().strip("()").strip().upper()
        label_normalized = label.strip().strip("()").strip().upper()
        if pred_normalized == label_normalized:
            return {
                "pred": pred_ans,
                "pass": True
            }
        else:
            return {
                "pred": pred_ans,
                "pass": False
            }