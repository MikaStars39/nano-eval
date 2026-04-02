"""
Profiling pipeline: preprocess -> Ray inference -> score.

Usage:
    python run.py --input data.jsonl --output-dir ./out \
        --model-path /path/to/model --num-nodes 4 \
        --prompt-key messages --label-key label
"""

import argparse
import asyncio
import json
import math
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import ray
from transformers import AutoTokenizer

from nanoeval.reward.score import eval_results


# ------------------------ Step 1: Preprocess ------------------------

def _preprocess(
    input_file: str, 
    output_file: str, 
    tokenizer_path: str,
    prompt_key: str, 
    label_key: str, 
    n_examples: int, 
    num_workers: int
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _process(args):
        idx, line = args
        if not line.strip():
            return None
        data = json.loads(line)
        results = []
        for oi in range(n_examples):
            prompt = tokenizer.apply_chat_template(
                data[prompt_key], tokenize=False, add_generation_prompt=True)
            results.append(json.dumps({
                "question_id": f"{idx}",
                "output_idx": oi,
                "label": data[label_key],
                "source": data.get("source", "default"),
                "prompt": prompt,
            }, ensure_ascii=False))
        return results

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for batch in Pool(num_workers).imap(_process, enumerate(fin), chunksize=64):
            if batch:
                for line in batch:
                    fout.write(line + "\n")
                    total += 1
    print(f"[preprocess] {total} entries -> {output_file}")


# ------------------------ Step 2: Ray Inference ------------------------

def _split(
    input_file: str, 
    num_shards: int, 
    out_dir: str
) -> list[str]:

    os.makedirs(out_dir, exist_ok=True)
    lines = Path(input_file).read_text("utf-8").splitlines()
    size = math.ceil(len(lines) / num_shards)
    paths = []
    for i in range(num_shards):
        chunk = lines[i * size:(i + 1) * size]
        if not chunk:
            continue
        p = os.path.join(out_dir, f"shard_{i:05d}.jsonl")
        Path(p).write_text("\n".join(chunk) + "\n", "utf-8")
        paths.append(p)
    return paths


def _merge(
    shard_paths: list[str], 
    output_file: str
) -> int:

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for p in sorted(shard_paths):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


@ray.remote(num_gpus=1)
def _run_shard(
    shard_in: str, 
    shard_out: str, 
    model_path: str,
    tp: int, 
    dp: int, 
    max_inflight: int, 
    mem_frac: float,
    dp_attn: bool, 
    sampling: dict, 
    resume: bool
) -> str:

    from nanoeval.backend.offline import BatchInferenceEngine

    async def _go():
        async with BatchInferenceEngine(
            model_path=model_path, tp_size=tp, dp_size=dp,
            max_inflight=max_inflight, mem_fraction_static=mem_frac,
            enable_dp_attention=dp_attn,
        ) as engine:
            await engine.run(input_file=shard_in, output_file=shard_out,
                             sampling_params=sampling, resume=resume)
    asyncio.run(_go())
    return shard_out


# ------------------------ Step 3: Score ------------------------

def _score(inference_output: str, score_output: str, final_output: str, n_proc: int):
    metrics = eval_results(
        eval_output_file=Path(inference_output),
        score_output_file=Path(score_output),
        final_eval_output_file=Path(final_output),
        final_eval_csv_output_file=Path(final_output).with_suffix(".csv"),
        n_proc=n_proc,
    )
    for task, m in metrics.items():
        print(f"  {task}: avg_k={m.get('avg_k', 0):.4f}  pass_k={m.get('pass_k', 0):.4f}")


# ------------------------ Main ------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", default="all", choices=["preprocess", "inference", "score", "all"])
    # preprocess
    p.add_argument("--prompt-key", default="messages")
    p.add_argument("--label-key", default="label")
    p.add_argument("--tokenizer", default=None, help="Defaults to --model-path")
    p.add_argument("--num-examples", type=int, default=1)
    p.add_argument("--preprocess-workers", type=int, default=max(1, cpu_count() - 2))
    # inference
    p.add_argument("--model-path", default=None)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--ray-address", default="auto")
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--max-inflight", type=int, default=512)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--enable-dp-attention", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--resume", action="store_true")
    # score
    p.add_argument("--n-proc", type=int, default=32)
    a = p.parse_args()

    od = a.output_dir
    os.makedirs(od, exist_ok=True)
    prepared = os.path.join(od, "prepared.jsonl")
    inference_out = os.path.join(od, "inference.jsonl")
    score_out = os.path.join(od, "score.jsonl")
    final_out = os.path.join(od, "final_eval.jsonl")

    if a.stage in ("preprocess", "all"):
        tok = a.tokenizer or a.model_path
        if not tok:
            raise ValueError("--tokenizer or --model-path required for preprocess")
        _preprocess(a.input, prepared, tok,
                    a.prompt_key, a.label_key, a.num_examples, a.preprocess_workers)

    if a.stage in ("inference", "all"):
        if not a.model_path:
            raise ValueError("--model-path required for inference")
        shards_dir = os.path.join(od, "shards")
        shard_inputs = _split(prepared, a.num_nodes, os.path.join(shards_dir, "input"))
        print(f"[split] {len(shard_inputs)} shards")

        ray.init(address=a.ray_address)
        sampling = {"temperature": a.temperature, "top_p": a.top_p, "max_new_tokens": a.max_new_tokens}
        out_dir = os.path.join(shards_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

        futures = []
        for i, si in enumerate(shard_inputs):
            so = os.path.join(out_dir, f"shard_{i:05d}.jsonl")
            f = _run_shard.options(num_gpus=a.tp_size * a.dp_size).remote(
                si, so, a.model_path, a.tp_size, a.dp_size, a.max_inflight,
                a.mem_fraction_static, a.enable_dp_attention, sampling, a.resume)
            futures.append(f)

        print(f"[ray] dispatched {len(futures)} tasks, waiting...")
        shard_outputs = ray.get(futures)
        total = _merge(shard_outputs, inference_out)
        print(f"[merge] {total} lines -> {inference_out}")
        ray.shutdown()

    if a.stage in ("score", "all"):
        print("[score] evaluating...")
        _score(inference_out, score_out, final_out, a.n_proc)


if __name__ == "__main__":
    main()
