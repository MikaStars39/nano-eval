import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from transformers import AutoTokenizer

from nanoeval.backend.offline import BatchInferenceEngine


def _make_chat_formatter(
    model_path: str,
    enable_thinking: bool,
) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _format(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # For Stage 2, we allow toggling thinking mode because DeepSeek models
        # may follow rewrite constraints better with thinking enabled.
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Fallback for tokenizers that do not support enable_thinking.
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return formatted

    return _format


def _text_len_for_quality(text: str) -> int:
    compact = re.sub(r"\s+", "", text)
    return len(compact)


def _build_rewrite_prompt(
    question: str,
    thinking_answer_text: str,
    min_length_ratio: float,
    min_length_floor: int,
) -> str:
    cleaned_thinking_text = _strip_think_tokens(thinking_answer_text)
    src_len = _text_len_for_quality(cleaned_thinking_text)
    target_min_len = max(min_length_floor, int(src_len * min_length_ratio))
    return (
        "Rewrite the following thinking-style answer into a non-thinking CoT answer.\n"
        "Requirements:\n"
        "1) Preserve all reasoning steps and key calculations. Do not shorten significantly.\n"
        "2) Keep the final answer and make sure it is consistent with the reasoning.\n"
        "3) Use the same language as the source (English in -> English out, Chinese in -> Chinese out).\n"
        "4) Output only the rewritten text. No JSON, no extra explanation.\n\n"
        "5) Do NOT evaluate whether requirements are met. Do NOT output phrases like "
        "\"The rewritten text meets requirements\". Output the rewritten answer itself only.\n\n"
        "6) Use a structured rewritten style like:\n"
        "# Step 1: ...\n"
        "# Step 2: ...\n"
        "# Step 3: ...\n"
        "Finish with the final answer/conclusion as the last step.\n\n"
        f"Length constraint: source length is about {src_len} chars (whitespace removed). "
        f"Your output must be around {src_len} chars (whitespace removed).\n"
        "Do not compress into a short summary.\n\n"
        f"Question:\n{question}\n\n"
        f"Content to rewrite:\n{cleaned_thinking_text}\n"
        "Requirements:\n"
        "1) Preserve all reasoning steps and key calculations. Do not shorten significantly.\n"
        "2) Keep the final answer and make sure it is consistent with the reasoning.\n"
        "3) Use the same language as the source (English in -> English out, Chinese in -> Chinese out).\n"
        "4) Output only the rewritten text. No JSON, no extra explanation.\n\n"
        "5) Do NOT evaluate whether requirements are met. Do NOT output phrases like "
        "\"The rewritten text meets requirements\". Output the rewritten answer itself only.\n\n"
        "6) Use a structured rewritten style like:\n"
        "# Step 1: ...\n"
        "# Step 2: ...\n"
        "# Step 3: ...\n"
        "Finish with the final answer/conclusion as the last step.\n\n"
        f"Length constraint: source length is about {src_len} chars (whitespace removed). "
        f"Your output must be around {src_len} chars (whitespace removed).\n"
        "Do not compress into a short summary!! As long as possible! As much as possible!\n\n"
    )


def _strip_think_tokens(text: str) -> str:
    """Remove any leaked <think>...</think> blocks or stray </think> tokens."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|/?(?:begin_of_thought|end_of_thought)\|>", "", text)
    return text.strip()


def prepare_shard_input(
    shard_source_jsonl: Path,
    shard_prepared_jsonl: Path,
    prompt_formatter: Optional[Callable[[str], str]] = None,
    min_length_ratio: float = 0.9,
    min_length_floor: int = 120,
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
            question = str(item.get("question", "")).strip()
            text = str(item.get("thinking_answer_text", "")).strip()
            if not question or not text:
                continue
            raw_prompt = _build_rewrite_prompt(
                question=question,
                thinking_answer_text=text,
                min_length_ratio=min_length_ratio,
                min_length_floor=min_length_floor,
            )
            prompt = prompt_formatter(raw_prompt) if prompt_formatter else raw_prompt
            fout.write(json.dumps({"id": item_id, "prompt": prompt}, ensure_ascii=False) + "\n")
            valid += 1
    return total, valid


def _build_delivery_conversations(conversations: Any, rewritten_text: str) -> list:
    # Keep only system/human messages from source, then append one rewritten gpt.
    conv_list = conversations if isinstance(conversations, list) else []
    kept = []
    for msg in conv_list:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("from", "")).strip().lower()
        value = str(msg.get("value", ""))
        if role == "system":
            kept.append({"from": "system", "value": value})
        elif role in {"human", "user"}:
            kept.append({"from": "human", "value": value})
    kept.append({"from": "gpt", "value": rewritten_text})
    return kept


def merge_shard_output(
    shard_source_jsonl: Path,
    shard_model_output_jsonl: Path,
    shard_final_jsonl: Path,
    min_length_ratio: float,
    min_length_floor: int,
    fallback_to_source_on_short: bool,
) -> Tuple[int, int]:
    shard_final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    source_map: Dict[str, Dict[str, Any]] = {}
    written = 0
    skipped = 0
    replaced_short = 0

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
            response = _strip_think_tokens(str(item.get("response", "")))
            if not response:
                skipped += 1
                continue
            source_item = source_map.get(item_id, {})
            src_text = _strip_think_tokens(str(source_item.get("thinking_answer_text", "")))
            src_len = _text_len_for_quality(src_text)
            response_len = _text_len_for_quality(response)
            target_min_len = max(min_length_floor, int(src_len * min_length_ratio)) if src_len > 0 else min_length_floor
            if fallback_to_source_on_short and src_text and response_len < target_min_len:
                response = src_text
                replaced_short += 1
            output = {
                "conversations": _build_delivery_conversations(
                    source_item.get("conversations") if isinstance(source_item, dict) else [],
                    response,
                )
            }
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    print(f"[merge] short_replaced={replaced_short}")
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
    parser = argparse.ArgumentParser(description="Stage2 shard runner: rewrite thinking answer into non-thinking answer.")
    parser.add_argument("--shard-source-jsonl", required=True)
    parser.add_argument("--shard-prepared-jsonl", required=True)
    parser.add_argument("--shard-model-output-jsonl", required=True)
    parser.add_argument("--shard-final-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=512)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat-template-model-path", default="")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to let chat template enter thinking mode for generation.",
    )
    parser.add_argument("--min-length-ratio", type=float, default=0.9)
    parser.add_argument("--min-length-floor", type=int, default=120)
    parser.add_argument("--fallback-to-source-on-short", action=argparse.BooleanOptionalAction, default=True)
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
            prompt_formatter = _make_chat_formatter(
                tokenizer_path,
                enable_thinking=args.enable_thinking,
            )
            print(
                "[prepare] chat template enabled "
                f"(enable_thinking={args.enable_thinking}) "
                f"tokenizer={tokenizer_path}"
            )
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
        total, valid = prepare_shard_input(
            shard_source,
            shard_prepared,
            prompt_formatter=prompt_formatter,
            min_length_ratio=args.min_length_ratio,
            min_length_floor=args.min_length_floor,
        )
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

    merged, skipped = merge_shard_output(
        shard_source,
        shard_model_output,
        shard_final,
        min_length_ratio=args.min_length_ratio,
        min_length_floor=args.min_length_floor,
        fallback_to_source_on_short=args.fallback_to_source_on_short,
    )
    print(f"[merge] written={merged}, skipped_empty={skipped}, output={shard_final}")


if __name__ == "__main__":
    main()
