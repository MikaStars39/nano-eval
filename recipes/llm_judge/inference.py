import argparse

import ray

from nanoeval.ray import init_ray
from nanoeval.ray.actors import OfflineInferenceActor


def main():
    parser = argparse.ArgumentParser(description="Inference: Run LLM batch processing via Ray.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--enable_dp_attention", action="store_true", help="Enable DP attention for multi-GPU inference")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address")

    args = parser.parse_args()

    sampling_params = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
    }

    init_ray(address=args.ray_address)

    num_gpus = args.tp_size * args.dp_size
    actor = OfflineInferenceActor.options(num_gpus=num_gpus).remote(
        model_path=args.model_path,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        max_inflight=args.max_concurrency,
        mem_fraction_static=args.gpu_mem,
        enable_dp_attention=args.enable_dp_attention,
    )
    ray.get(actor.run.remote(
        args.input, args.output, sampling_params, resume=args.resume,
    ))


if __name__ == "__main__":
    main()
