import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from transformers import AutoTokenizer

from nanoeval.backend.offline import BatchInferenceEngine


def _make_chat_formatter(model_path: str) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _format(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return _format


def _extract_last_human_message(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ""
    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip().lower()
        if role in {"human", "user"}:
            return str(message.get("value", "")).strip()
    return ""


def prepare_shard_input(
    shard_source_jsonl: Path,
    shard_prepared_jsonl: Path,
    prompt_formatter: Optional[Callable[[str], str]] = None,
) -> Tuple[int, int]:
    shard_prepared_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    valid = 0

    with shard_source_jsonl.open("r", encoding="utf-8") as fin, shard_prepared_jsonl.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(item.get("id", f"line-{idx}"))
            question = _extract_last_human_message(item.get("conversations"))
            if not question:
                continue
            prompt = prompt_formatter(question) if prompt_formatter else question
            fout.write(json.dumps({"id": item_id, "prompt": prompt}, ensure_ascii=False) + "\n")
            valid += 1
    return total, valid


def _replace_last_assistant_message(conversations: Any, rewritten_text: str) -> list:
    conv_list = conversations if isinstance(conversations, list) else []
    copied = [dict(msg) if isinstance(msg, dict) else msg for msg in conv_list]
    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("from", "")).strip().lower()
        if role in {"gpt", "assistant"}:
            msg["value"] = rewritten_text
            return copied
    copied.append({"from": "gpt", "value": rewritten_text})
    return copied


def merge_shard_output(shard_source_jsonl: Path, shard_model_output_jsonl: Path, shard_final_jsonl: Path) -> Tuple[int, int]:
    shard_final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    source_map: Dict[str, Dict[str, Any]] = {}
    written = 0
    skipped = 0

    with shard_source_jsonl.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            item = json.loads(line)
            source_map[str(item.get("id", f"line-{idx}"))] = item

    with shard_model_output_jsonl.open("r", encoding="utf-8") as fin, shard_final_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id"))
            response = str(item.get("response", "")).strip()
            if not response:
                skipped += 1
                continue

            source_item = source_map.get(item_id, {})
            output = dict(source_item) if isinstance(source_item, dict) else {"id": item_id}
            output["id"] = item_id
            output["conversations"] = _replace_last_assistant_message(output.get("conversations"), response)
            output.pop("question", None)
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


async def run_offline(
    shard_prepared_jsonl: Path,
    shard_model_output_jsonl: Path,
    model_path: str,
    tp_size: int,
    dp_size: int,
    max_inflight: int,
    enable_dp_attention: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    resume: bool,
) -> None:
    sampling_params = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens}
    async with BatchInferenceEngine(
        model_path=model_path,
        tp_size=tp_size,
        dp_size=dp_size,
        max_inflight=max_inflight,
        enable_dp_attention=enable_dp_attention,
    ) as engine:
        await engine.run(
            input_file=str(shard_prepared_jsonl),
            output_file=str(shard_model_output_jsonl),
            sampling_params=sampling_params,
            resume=resume,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 shard runner: generate thinking from question only.")
    parser.add_argument("--shard-source-jsonl", required=True)
    parser.add_argument("--shard-prepared-jsonl", required=True)
    parser.add_argument("--shard-model-output-jsonl", required=True)
    parser.add_argument("--shard-final-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=512)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat-template-model-path", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_formatter: Optional[Callable[[str], str]] = None
    if args.apply_chat_template:
        tokenizer_path = args.chat_template_model_path or args.model_path
        try:
            prompt_formatter = _make_chat_formatter(tokenizer_path)
            print(f"[prepare] chat template enabled with tokenizer={tokenizer_path}")
        except Exception as exc:
            print(f"[prepare] chat template unavailable ({exc}), fallback to raw prompt.")

    shard_source = Path(args.shard_source_jsonl)
    shard_prepared = Path(args.shard_prepared_jsonl)
    shard_model_output = Path(args.shard_model_output_jsonl)
    shard_final = Path(args.shard_final_jsonl)

    if args.skip_prepare:
        if not shard_prepared.exists():
            raise FileNotFoundError(f"--skip-prepare is set, but file not found: {shard_prepared}")
        print(f"[prepare] skipped, reuse={shard_prepared}")
    else:
        total, valid = prepare_shard_input(shard_source, shard_prepared, prompt_formatter=prompt_formatter)
        print(f"[prepare] total={total}, valid={valid}, output={shard_prepared}")
        if valid == 0:
            print("[run] no valid samples, skip.")
            return

    if not args.merge_only:
        asyncio.run(
            run_offline(
                shard_prepared_jsonl=shard_prepared,
                shard_model_output_jsonl=shard_model_output,
                model_path=args.model_path,
                tp_size=args.tp_size,
                dp_size=args.dp_size,
                max_inflight=args.max_inflight,
                enable_dp_attention=args.enable_dp_attention,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
            )
        )

    merged, skipped = merge_shard_output(shard_source, shard_model_output, shard_final)
    print(f"[merge] written={merged}, skipped_empty={skipped}, output={shard_final}")


if __name__ == "__main__":
    main()

