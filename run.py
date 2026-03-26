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
    parse_task_pass_k,
    prepare_eval_input,
)

def _resolve_output_path(output_path: Path, work_dir: Path | None) -> Path:
    if work_dir is None:
        return output_path
    return output_path

def main(argv: Sequence[str] | None = None) -> int:
    configure_logger(prefix=" nanoeval")
    args = parse_cli_args(argv)
    final_summary: dict[str, object] = {"stage": args.stage}
    work_dir: Path | None = args.work_dir

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    step01_output = _resolve_output_path(args.output, work_dir)
    step02_output = _resolve_output_path(args.inference_output, work_dir)
    score_output = _resolve_output_path(args.score_output, work_dir)
    final_eval_output = _resolve_output_path(args.final_eval_output, work_dir)

    if args.stage in {"step01", "all"}:
        chat_template_model_path = args.chat_template_model_path or args.model_path
        task_names, pass_k_by_task = parse_task_pass_k(
            tasks_arg=args.tasks,
            task_dir=args.task_dir,
            default_pass_k=args.pass_k,
        )
        logging.info("Step 0/1 system_prompt: %r", args.system_prompt)
        step01_summary = prepare_eval_input(
            task_names=task_names,
            task_dir=args.task_dir,
            pass_k_by_task=pass_k_by_task,
            output_path=step01_output,
            chat_template_model_path=chat_template_model_path,
            system_prompt=args.system_prompt,
        )
        final_summary["step01"] = step01_summary
        logging.info("Step 0/1 completed: %s", step01_summary)

    if args.stage in {"step02", "all"}:
        inference_input = args.input or step01_output
        sampling_params = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        optional_sampling_fields = (
            ("top_p", args.top_p),
            ("top_k", args.top_k),
            ("min_p", args.min_p),
            ("presence_penalty", args.presence_penalty),
            ("repetition_penalty", args.repetition_penalty),
            ("reasoning_effort", args.reasoning_effort),
        )
        for field_name, field_value in optional_sampling_fields:
            if field_value is not None:
                sampling_params[field_name] = field_value
        if args.system_prompt and args.backend in {"online", "online_ray"}:
            sampling_params["__system_prompt"] = args.system_prompt
        if args.enable_thinking is not None:
            if args.backend in {"online", "online_ray"}:
                sampling_params["chat_template_kwargs"] = {
                    "enable_thinking": args.enable_thinking
                }
            else:
                logging.info(
                    "--enable-thinking is ignored for backend=%s. "
                    "chat_template_kwargs is only applied to online-compatible backends.",
                    args.backend,
                )
        step02_summary = run_inference(
            backend=args.backend,
            input_file=inference_input,
            output_file=step02_output,
            sampling_params=sampling_params,
            model_path=args.model_path,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            ray_num_actors=args.ray_num_actors,
            ray_worker_concurrency=args.ray_worker_concurrency,
            online_request_timeout_s=args.online_request_timeout_s,
            online_stall_log_interval_s=args.online_stall_log_interval_s,
        )
        final_summary["step02"] = step02_summary
        logging.info("Step 2 completed: %s", step02_summary)

    if args.stage in {"step03", "all"}:
        eval_input = args.eval_input or step02_output
        metrics = eval_results(
            eval_output_file=eval_input,
            score_output_file=score_output,
            final_eval_output_file=final_eval_output,
            final_eval_csv_output_file=final_eval_output.with_suffix(".csv"),
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
