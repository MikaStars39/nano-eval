from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from nanoeval.backend import run_inference
from nanoeval.reward.score import eval_results
from nanoeval.utils import (
    configure_logger,
    parse_cli_args,
    parse_task_names,
    prepare_eval_input,
)
from nanoeval.utils.args import (
    DEFAULT_FINAL_EVAL_OUTPUT,
    DEFAULT_SCORE_OUTPUT,
    DEFAULT_STEP01_OUTPUT,
    DEFAULT_STEP02_OUTPUT,
)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logger(prefix=" nanoeval")
    args = parse_cli_args(argv)
    final_summary: dict[str, object] = {"stage": args.stage}
    step01_output: Path = args.output
    step02_output: Path = args.inference_output
    score_output: Path = args.score_output
    final_eval_output: Path = args.final_eval_output

    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        if args.output == DEFAULT_STEP01_OUTPUT:
            step01_output = work_dir / DEFAULT_STEP01_OUTPUT
        if args.inference_output == DEFAULT_STEP02_OUTPUT:
            step02_output = work_dir / DEFAULT_STEP02_OUTPUT
        if args.score_output == DEFAULT_SCORE_OUTPUT:
            score_output = work_dir / DEFAULT_SCORE_OUTPUT
        if args.final_eval_output == DEFAULT_FINAL_EVAL_OUTPUT:
            final_eval_output = work_dir / DEFAULT_FINAL_EVAL_OUTPUT

    if args.stage in {"step01", "all"}:
        chat_template_model_path = args.chat_template_model_path or args.model_path
        if chat_template_model_path is None:
            raise ValueError(
                "chat template model path is required for step01. "
                "Please provide --chat-template-model-path or --model-path."
            )
        task_names = parse_task_names(tasks_arg=args.tasks, task_dir=args.task_dir)
        step01_summary = prepare_eval_input(
            task_names=task_names,
            task_dir=args.task_dir,
            pass_k=args.pass_k,
            output_path=step01_output,
            chat_template_model_path=chat_template_model_path,
            system_prompt=args.system_prompt,
        )
        final_summary["step01"] = step01_summary
        logging.info("Step 0/1 completed: %s", step01_summary)

    if args.stage in {"step02", "all"}:
        inference_input = args.input or step01_output
        step02_summary = run_inference(
            backend=args.backend,
            input_file=inference_input,
            output_file=step02_output,
            sampling_params={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
        )
        final_summary["step02"] = step02_summary
        logging.info("Step 2 completed: %s", step02_summary)

    if args.stage in {"step03", "all"}:
        eval_input = args.eval_input or step02_output
        metrics = eval_results(
            eval_output_file=eval_input,
            score_output_file=score_output,
            final_eval_output_file=final_eval_output,
            n_proc=args.n_proc,
        )
        step03_summary = {
            "input_path": str(eval_input),
            "score_output_path": str(score_output),
            "final_eval_output_path": str(final_eval_output),
            "metrics": metrics,
        }
        final_summary["step03"] = step03_summary
        logging.info("Step 3 completed: %s", step03_summary)

    print(json.dumps(final_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
