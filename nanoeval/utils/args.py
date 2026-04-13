from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from .task import discover_task_names, resolve_task_file


def parse_task_names(tasks_arg: str, task_dir: Path) -> List[str]:
    normalized_arg = tasks_arg.strip()
    if not normalized_arg:
        raise ValueError("--tasks cannot be empty.")

    if normalized_arg.lower() == "all":
        names = discover_task_names(task_dir)
        if not names:
            raise ValueError(f"No task jsonl found under: {task_dir}")
        return names

    task_specs = _parse_task_specs(normalized_arg)
    names = [task_name for task_name, _ in task_specs]
    _validate_task_names(names=names, task_dir=task_dir)
    return names


def _parse_task_specs(tasks_arg: str) -> List[Tuple[str, str | None]]:
    task_specs: List[Tuple[str, str | None]] = []
    for task_spec in [item.strip() for item in tasks_arg.split(",") if item.strip()]:
        if "@" in task_spec:
            task_part, pass_k_part = task_spec.split("@", 1)
            task_name = task_part.strip()
            pass_k_value = pass_k_part.strip()
        else:
            task_name = task_spec
            pass_k_value = None
        if task_name.endswith(".jsonl"):
            task_name = task_name[:-6]
        task_name = task_name.strip()
        if not task_name:
            raise ValueError(f"Invalid task spec in --tasks: {task_spec}")
        task_specs.append((task_name, pass_k_value))
    return task_specs


def _validate_task_names(names: List[str], task_dir: Path) -> None:
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


def parse_task_pass_k(
    tasks_arg: str,
    task_dir: Path,
    default_pass_k: int,
) -> Tuple[List[str], Dict[str, int]]:
    if default_pass_k <= 0:
        raise ValueError("--pass-k must be a positive integer.")

    normalized_arg = tasks_arg.strip()
    if not normalized_arg:
        raise ValueError("--tasks cannot be empty.")

    if normalized_arg.lower() == "all":
        task_names = parse_task_names(tasks_arg=tasks_arg, task_dir=task_dir)
        pass_k_by_task: Dict[str, int] = {task_name: default_pass_k for task_name in task_names}
        return task_names, pass_k_by_task

    task_specs = _parse_task_specs(normalized_arg)
    task_names = [task_name for task_name, _ in task_specs]
    _validate_task_names(names=task_names, task_dir=task_dir)
    pass_k_by_task: Dict[str, int] = {task_name: default_pass_k for task_name in task_names}

    for task_name, pass_k_text in task_specs:
        if pass_k_text is None:
            continue
        if task_name not in pass_k_by_task:
            raise ValueError(f"Unknown task in --tasks: {task_name}")
        if not pass_k_text.isdigit():
            raise ValueError(
                f"Invalid pass-k value in --tasks spec '{task_name}@{pass_k_text}'. "
                "Expected a positive integer after '@'."
            )
        pass_k_value = int(pass_k_text)
        if pass_k_value <= 0:
            raise ValueError(
                f"Invalid pass-k value in --tasks spec '{task_name}@{pass_k_text}'. "
                "pass-k must be a positive integer."
            )
        pass_k_by_task[task_name] = pass_k_value

    return task_names, pass_k_by_task
