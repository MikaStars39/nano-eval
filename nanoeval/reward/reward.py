from typing import Dict
from importlib import import_module

from .math.math_verify_reward import math_judge

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
    elif any(key in source_lower for key in ("typos", "connections", "unscrambling")):
        language_module = import_module("nanoeval.reward.livebench.verify_language")
        language_judge = getattr(language_module, "language_judge")
        return language_judge(response, label, source, **kwargs)
    elif "tablejoin" in source_lower:
        table_module = import_module("nanoeval.reward.livebench.verify_table")
        table_process_results = getattr(table_module, "table_process_results")
        prompt = kwargs.get("prompt", "")
        return table_process_results(prompt, response, label)
    else:
        #
        # math return:
        # return {
        #     "pred": pred_ans,
        #     "pass": True if score == 1.0 else False
        # }
        #
        return math_judge(response, label, **kwargs)

if __name__ == "__main__":
    def _run_tests() -> None:
        ifeval_result = judge_router(
            response="Apple banana",
            source="ifeval-v1",
            instruction_id_list=["keywords:existence"],
            kwargs=[{"keywords": ["Apple"]}],
        )
        assert ifeval_result["pass"] is True
        assert ifeval_result["instruction_count"] == 1
        assert ifeval_result["instruction_pass_cnt"] == 1

        gpqa_result = judge_router(
            response="Final answer: \\boxed{A}",
            label="A",
            source="gpqa",
        )
        assert gpqa_result["pass"] is True
        assert gpqa_result["pred"] == "A"

        typo_text = "We see a simple typo check."
        language_score = judge_router(
            response=typo_text,
            label=typo_text,
            source="typos",
        )
        assert language_score == 1

        prompt = "Please convert the Input Table from csv format to csv format"
        csv_table = "col1,col2\n1,2\n"
        table_score = judge_router(
            response=csv_table,
            label=csv_table,
            source="tablejoin",
            prompt=prompt,
        )
        assert table_score == 1

        math_result = judge_router(
            response="Therefore, the answer is \\boxed{2}.",
            label="2",
            source="math",
        )
        assert math_result["pass"] is True
        assert math_result["pred"] == "2"

    _run_tests()
    print("reward.judge_router tests passed")
