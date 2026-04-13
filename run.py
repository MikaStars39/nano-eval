"""
Ray-orchestrated eval pipeline: preprocess -> shard -> inference -> merge -> score.

Distributes inference across multiple Ray actors (offline or online) with
automatic JSONL sharding and merging.

Usage (offline, multi-node):
    python run.py --tasks "aime2025@8,math500@1" \\
        --task-dir /data/nano_eval --output-dir ./out \\
        --backend offline --model-path /path/to/model \\
        --num-actors 4 --tp-size 8

Usage (online, parallel API workers):
    python run.py --tasks "aime2025@8" \\
        --task-dir /data/nano_eval --output-dir ./out \\
        --backend online --model my-model \\
        --api-key $API_KEY --base-url https://api.example.com/v1 \\
        --num-actors 8

Usage (online, agent loop):
    python run.py --tasks "aime2025@8" \\
        --task-dir /data/nano_eval --output-dir ./out \\
        --backend online --model my-model \\
        --api-key $API_KEY --base-url https://api.example.com/v1 \\
        --agent-loop --max-turns 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ray

from nanoeval.ray import init_ray
from nanoeval.ray.actors import (
    OfflineInferenceActor,
    OnlineInferenceActor,
    PreprocessActor,
    ScoringActor,
)
from nanoeval.ray.utils import merge_jsonl, shard_jsonl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Ray-orchestrated NanoEval pipeline")

    # Data / task
    p.add_argument("--tasks", type=str, default="all",
                   help="Comma-separated task names (task@k syntax supported)")
    p.add_argument("--task-dir", type=Path, default=Path("outputs/nano_eval"))
    p.add_argument("--pass-k", type=int, default=1)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--stage", default="all",
                   choices=["preprocess", "inference", "score", "all"])

    # Preprocess
    p.add_argument("--chat-template-model-path", type=str, default=None,
                   help="Defaults to --model-path")
    p.add_argument("--system-prompt", type=str, default=None)

    # Backend
    p.add_argument("--backend", type=str, required=True,
                   choices=["offline", "online"])

    # Offline inference
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--max-inflight", type=int, default=512)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--enable-dp-attention", action="store_true")

    # Online inference
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=32)

    # Agent loop (online only)
    p.add_argument("--agent-loop", action="store_true", default=False,
                   help="Enable multi-turn agent loop with tool calling (online backend only).")
    p.add_argument("--max-turns", type=int, default=10,
                   help="Max turns per conversation in agent loop mode.")

    # Sampling
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None,
                   help="Optional presence penalty for token generation.")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="Optional repetition penalty for token generation.")
    p.add_argument("--reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="Optional reasoning effort level for models/APIs that support it.")
    p.add_argument("--enable-thinking", type=str, default=None,
                   help="true/false — set chat template thinking mode")

    # Ray / sharding
    p.add_argument("--num-actors", type=int, default=1,
                   help="Number of parallel inference actors")
    p.add_argument("--ray-address", type=str, default="auto")
    p.add_argument("--resume", action="store_true")

    # Score
    p.add_argument("--n-proc", type=int, default=32)

    a = p.parse_args()

    od = a.output_dir
    os.makedirs(od, exist_ok=True)
    prepared = os.path.join(od, "prepared.jsonl")
    inference_out = os.path.join(od, "inference.jsonl")
    score_out = os.path.join(od, "score.jsonl")
    final_out = os.path.join(od, "final_eval.jsonl")

    # ── Preprocess ──
    if a.stage in ("preprocess", "all"):
        log.info("[preprocess] preparing eval input...")

        from nanoeval.utils import parse_task_pass_k

        task_names, pass_k_by_task = parse_task_pass_k(
            tasks_arg=a.tasks,
            task_dir=a.task_dir,
            default_pass_k=a.pass_k,
        )
        chat_template_path = a.chat_template_model_path or a.model_path

        init_ray(address=a.ray_address)
        prep_actor = PreprocessActor.options(num_cpus=1).remote()
        summary = ray.get(prep_actor.run.remote(
            task_names=task_names,
            task_dir=str(a.task_dir),
            pass_k_by_task=pass_k_by_task,
            output_path=prepared,
            chat_template_model_path=chat_template_path,
            system_prompt=a.system_prompt,
        ))
        log.info("[preprocess] done: %s", summary)

    # ── Inference ──
    if a.stage in ("inference", "all"):
        init_ray(address=a.ray_address)

        # Build sampling params
        sampling: dict = {
            "temperature": a.temperature,
            "max_tokens": a.max_tokens,
        }
        for key, val in [
            ("top_p", a.top_p),
            ("top_k", a.top_k),
            ("min_p", a.min_p),
            ("presence_penalty", a.presence_penalty),
            ("repetition_penalty", a.repetition_penalty),
            ("reasoning_effort", a.reasoning_effort),
        ]:
            if val is not None:
                sampling[key] = val
        if a.enable_thinking is not None:
            sampling["chat_template_kwargs"] = {
                "enable_thinking": a.enable_thinking.lower() in ("1", "true", "yes"),
            }
        if a.system_prompt and a.backend == "online":
            sampling["__system_prompt"] = a.system_prompt

        # Shard input
        shards_dir = os.path.join(od, "shards")
        shard_inputs = shard_jsonl(prepared, a.num_actors, os.path.join(shards_dir, "input"))
        log.info("[inference] split into %d actor(s)", len(shard_inputs))

        shard_out_dir = os.path.join(shards_dir, "output")
        os.makedirs(shard_out_dir, exist_ok=True)

        futures = []
        if a.backend == "offline":
            if not a.model_path:
                raise ValueError("--model-path required for offline backend")
            num_gpus = a.tp_size * a.dp_size
            total_gpus = num_gpus * a.num_actors
            log.info("[inference] offline: %d actor(s) x (tp=%d x dp=%d) = %d GPU(s) total",
                     a.num_actors, a.tp_size, a.dp_size, total_gpus)
            for i, si in enumerate(shard_inputs):
                so = os.path.join(shard_out_dir, f"shard_{i:05d}.jsonl")
                actor = OfflineInferenceActor.options(num_gpus=num_gpus).remote(
                    model_path=a.model_path,
                    tp_size=a.tp_size,
                    dp_size=a.dp_size,
                    max_inflight=a.max_inflight,
                    mem_fraction_static=a.mem_fraction_static,
                    enable_dp_attention=a.enable_dp_attention,
                )
                futures.append(actor.run.remote(si, so, sampling, resume=a.resume))

        elif a.backend == "online":
            if not a.api_key or not a.base_url or not a.model:
                raise ValueError("--api-key, --base-url, --model required for online backend")
            for i, si in enumerate(shard_inputs):
                so = os.path.join(shard_out_dir, f"shard_{i:05d}.jsonl")
                actor = OnlineInferenceActor.options(num_cpus=1).remote(
                    api_key=a.api_key,
                    base_url=a.base_url,
                    model=a.model,
                    concurrency=a.concurrency,
                )
                if a.agent_loop:
                    futures.append(actor.run_agent_loop.remote(
                        si, so, sampling, max_turns=a.max_turns,
                    ))
                else:
                    futures.append(actor.run.remote(si, so, sampling))

        log.info("[inference] dispatched %d actors, waiting...", len(futures))
        shard_outputs = ray.get(futures)
        total = merge_jsonl(shard_outputs, inference_out)
        log.info("[inference] merged %d lines -> %s", total, inference_out)

    # ── Score ──
    if a.stage in ("score", "all"):
        log.info("[score] evaluating...")
        init_ray(address=a.ray_address)
        scorer = ScoringActor.options(num_cpus=a.n_proc).remote(n_proc=a.n_proc)
        metrics = ray.get(scorer.run.remote(
            eval_output_file=inference_out,
            score_output_file=score_out,
            final_eval_output_file=final_out,
            final_eval_csv_output_file=str(Path(final_out).with_suffix(".csv")),
        ))
        for task, m in metrics.items():
            log.info("  %s: avg_k=%.4f  pass_k=%.4f",
                     task, m.get("avg_k", 0), m.get("pass_k", 0))

    log.info("[done]")


if __name__ == "__main__":
    main()
