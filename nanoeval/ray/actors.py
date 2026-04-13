"""
Ray actors wrapping core nanoeval operations.

Each actor encapsulates a single pipeline operation (inference, scoring,
preprocessing) and can be scheduled by Ray with appropriate resource requests.

Usage pattern::

    import ray
    from nanoeval.ray import init_ray, OfflineInferenceActor

    init_ray()
    actor = OfflineInferenceActor.options(num_gpus=8).remote(
        model_path="/path/to/model", tp_size=8,
    )
    output = ray.get(actor.run.remote("in.jsonl", "out.jsonl", {"temperature": 0.6}))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None


logger = logging.getLogger(__name__)


def _require_ray():
    if ray is None:
        raise RuntimeError("ray is required. Install it with: pip install ray")


# ---------------------------------------------------------------------------
# 1. OfflineInferenceActor
# ---------------------------------------------------------------------------

if ray is not None:

    @ray.remote
    class OfflineInferenceActor:
        """GPU-based inference using SGLang ``BatchInferenceEngine``.

        Resource scheduling is set by the caller via
        ``.options(num_gpus=tp_size * dp_size)``.
        """

        def __init__(
            self,
            model_path: str,
            tp_size: int = 1,
            dp_size: int = 1,
            max_inflight: int = 512,
            mem_fraction_static: float = 0.90,
            enable_dp_attention: bool = False,
        ):
            self.model_path = model_path
            self.tp_size = tp_size
            self.dp_size = dp_size
            self.max_inflight = max_inflight
            self.mem_fraction_static = mem_fraction_static
            self.enable_dp_attention = enable_dp_attention

        async def run(
            self,
            input_file: str,
            output_file: str,
            sampling_params: dict,
            resume: bool = False,
        ) -> str:
            """Run batch inference. Returns *output_file* path on completion."""
            from nanoeval.backend.offline import BatchInferenceEngine

            engine_kwargs: Dict[str, Any] = dict(
                model_path=self.model_path,
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                max_inflight=self.max_inflight,
                mem_fraction_static=self.mem_fraction_static,
            )
            if self.enable_dp_attention:
                engine_kwargs["enable_dp_attention"] = True

            async with BatchInferenceEngine(**engine_kwargs) as engine:
                await engine.run(
                    input_file=input_file,
                    output_file=output_file,
                    sampling_params=sampling_params,
                    resume=resume,
                )
            return output_file

    # -----------------------------------------------------------------------
    # 2. OnlineInferenceActor
    # -----------------------------------------------------------------------

    @ray.remote
    class OnlineInferenceActor:
        """API-based inference using ``OnlineBatchInferenceEngine``.

        For distributed online inference, create multiple actors and split
        the input across them using ``shard_jsonl`` / ``merge_jsonl``.

        Resource scheduling: ``.options(num_cpus=1)`` (I/O-bound).
        """

        def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            concurrency: int = 32,
        ):
            self.api_key = api_key
            self.base_url = base_url.rstrip("/")
            self.model = model
            self.concurrency = concurrency

        async def run(
            self,
            input_file: str,
            output_file: str,
            sampling_params: dict,
        ) -> str:
            """Run online inference. Returns *output_file* path on completion."""
            from nanoeval.backend.online import APIConfig, OnlineBatchInferenceEngine

            config = APIConfig(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            )
            engine = OnlineBatchInferenceEngine(config, concurrency=self.concurrency)
            await engine.run(
                input_file=input_file,
                output_file=output_file,
                sampling_params=sampling_params,
            )
            return output_file

    # -----------------------------------------------------------------------
    # 3. ScoringActor
    # -----------------------------------------------------------------------

    @ray.remote
    class ScoringActor:
        """Wraps ``eval_results`` for reward scoring / judging.

        Resource scheduling: ``.options(num_cpus=n_proc)`` to reserve cores
        for the internal multiprocessing pool.
        """

        def __init__(self, n_proc: int = 32):
            self.n_proc = n_proc

        def run(
            self,
            eval_output_file: str,
            score_output_file: str,
            final_eval_output_file: str,
            final_eval_csv_output_file: Optional[str] = None,
        ) -> dict:
            """Score inference results. Returns a metrics dict."""
            from pathlib import Path

            from nanoeval.reward.score import eval_results

            return eval_results(
                eval_output_file=Path(eval_output_file),
                score_output_file=Path(score_output_file),
                final_eval_output_file=Path(final_eval_output_file),
                final_eval_csv_output_file=(
                    Path(final_eval_csv_output_file)
                    if final_eval_csv_output_file
                    else None
                ),
                n_proc=self.n_proc,
            )

    # -----------------------------------------------------------------------
    # 4. PreprocessActor
    # -----------------------------------------------------------------------

    @ray.remote
    class PreprocessActor:
        """Wraps ``prepare_eval_input`` for input data preparation.

        Resource scheduling: ``.options(num_cpus=1)``.
        """

        def run(
            self,
            task_names: List[str],
            task_dir: str,
            pass_k_by_task: Dict[str, int],
            output_path: str,
            chat_template_model_path: Optional[str] = None,
            system_prompt: Optional[str] = None,
        ) -> dict:
            """Prepare eval input. Returns a summary dict."""
            from pathlib import Path

            from nanoeval.utils.task import prepare_eval_input

            return prepare_eval_input(
                task_names=task_names,
                task_dir=Path(task_dir),
                pass_k_by_task=pass_k_by_task,
                output_path=Path(output_path),
                chat_template_model_path=chat_template_model_path,
                system_prompt=system_prompt,
            )

else:
    # Stubs when ray is not installed — allows importing the module for
    # type-checking or documentation without requiring ray.
    class OfflineInferenceActor:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            _require_ray()

    class OnlineInferenceActor:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            _require_ray()

    class ScoringActor:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            _require_ray()

    class PreprocessActor:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            _require_ray()
