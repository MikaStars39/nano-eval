from typing import Dict

# ----------------------- IMPORTANT: judge router -----------------------
# rule-based judge router that manage the judging process

def judge_router(
    response: str,
    label: str = "",
    source: str = None,
    **kwargs
) -> Dict:

    source_lower = (source or "").lower()
    if "ifeval" in source_lower:
        from .if_eval.if_eval import if_judge

        #
        # ifeval return:
        # return {
        #     'instruction_count': len(instructions),
        #     'instruction_pass_cnt': instruction_pass_cnt,
        #     'pass': prompt_level_pass_flag
        # }
        #
        return if_judge(response, **kwargs)
    elif "gpqa" in source_lower or "mmlu" in source_lower:
        from .gpqa.gpqa_verify_reward import gpqa_judge

        return gpqa_judge(response, label, **kwargs)
    else:
        from .math.math_verify_reward import math_judge

        #
        # math return:
        # return {
        #     "pred": pred_ans,
        #     "pass": True if score == 1.0 else False
        # }
        #
        return math_judge(response, label, **kwargs)
