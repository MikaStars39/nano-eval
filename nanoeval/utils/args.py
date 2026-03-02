from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from .task import discover_task_names, resolve_task_file

DEFAULT_STEP01_OUTPUT = Path("artifacts/input/prepared_input.jsonl")
DEFAULT_STEP02_OUTPUT = Path("artifacts/output/inference_output.jsonl")
DEFAULT_SCORE_OUTPUT = Path("artifacts/score/scored_output.jsonl")
DEFAULT_FINAL_EVAL_OUTPUT = Path("artifacts/score/final_metrics.jsonl")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NanoEval pipeline: step01 prepare inputs, step02 inference."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["step01", "step02", "step03", "all"],
        default="step01",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names, or 'all' to load all *.jsonl under task directory.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("outputs/nano_eval"),
        help="Directory that stores task jsonl files.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=1,
        help="Number of repeated attempts per question.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for generated artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STEP01_OUTPUT,
        help="Output path for step01 prepared input jsonl.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input jsonl for step02 inference. If omitted, uses --output.",
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=DEFAULT_STEP02_OUTPUT,
        help="Output jsonl for step02 inference results.",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=DEFAULT_SCORE_OUTPUT,
        help="Output jsonl for per-instance judged results in step03.",
    )
    parser.add_argument(
        "--final-eval-output",
        type=Path,
        default=DEFAULT_FINAL_EVAL_OUTPUT,
        help="Output jsonl for aggregated metrics in step03.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Input jsonl for step03 evaluation. If omitted, uses --inference-output.",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=32,
        help="Number of processes for reward judging in step03.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "offline", "online"],
        default="mock",
        help="Inference backend for step02.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path for offline backend.",
    )
    parser.add_argument(
        "--chat-template-model-path",
        type=str,
        default=None,
        help="Model path used to apply chat template in step01. Defaults to --model-path.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt used when applying chat template in step01.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for online backend.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base url for online backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for online backend.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Online backend concurrency.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for inference.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Sampling max tokens for inference.",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def parse_task_names(tasks_arg: str, task_dir: Path) -> List[str]:
    normalized_arg = tasks_arg.strip()
    if not normalized_arg:
        raise ValueError("--tasks cannot be empty.")

    if normalized_arg.lower() == "all":
        names = discover_task_names(task_dir)
        if not names:
            raise ValueError(f"No task jsonl found under: {task_dir}")
        return names

    names = [name.strip() for name in normalized_arg.split(",") if name.strip()]
    normalized_names: List[str] = []
    for name in names:
        if name.endswith(".jsonl"):
            normalized_names.append(name[:-6])
        else:
            normalized_names.append(name)
    names = normalized_names
    if not names:
        raise ValueError("--tasks resolved to an empty list.")

    missing_names: List[str] = []
    for name in names:
        try:
            task_file = resolve_task_file(task_name=name, task_dir=task_dir)
        except ValueError:
            missing_names.append(name)
            continue
        if not task_file.exists():
            missing_names.append(name)
    if missing_names:
        raise ValueError(
            f"Missing task files under {task_dir}: {', '.join(sorted(missing_names))}"
        )
    return names
