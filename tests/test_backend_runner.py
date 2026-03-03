from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoeval.backend import run_inference


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestBackendRunner(unittest.TestCase):
    def test_run_inference_mock_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            _write_jsonl(
                input_path,
                [
                    {"id": "row_1", "prompt": "What is 1+1?", "label": "2"},
                    {"id": "row_2", "prompt": "What is 2+2?", "label": "4"},
                ],
            )

            summary = run_inference(
                backend="mock",
                input_file=input_path,
                output_file=output_path,
            )
            self.assertEqual(summary["backend"], "mock")
            self.assertEqual(summary["input_count"], 2)
            self.assertEqual(summary["output_count"], 2)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file_obj:
                rows = [json.loads(line) for line in file_obj if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertIn("response", rows[0])

    def test_run_inference_online_ray_backend_with_mock_engine(self) -> None:
        class _FakeOnlineRayBatchInferenceEngine:
            def __init__(
                self,
                api_config,
                num_actors: int,
                worker_concurrency: int,
                request_timeout_s: float = 120.0,
                stall_log_interval_s: float = 15.0,
            ):
                self.api_config = api_config
                self.num_actors = num_actors
                self.worker_concurrency = worker_concurrency
                self.request_timeout_s = request_timeout_s
                self.stall_log_interval_s = stall_log_interval_s

            async def run(self, input_file: str, output_file: str, sampling_params: dict) -> None:
                with open(input_file, "r", encoding="utf-8") as file_obj:
                    rows = [json.loads(line) for line in file_obj if line.strip()]
                with open(output_file, "w", encoding="utf-8") as file_obj:
                    for row in rows:
                        item = dict(row)
                        item["response"] = f"[FAKE_RAY] {row.get('prompt', '')}"
                        item["sampling_echo"] = dict(sampling_params)
                        file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            _write_jsonl(
                input_path,
                [
                    {"id": "row_1", "prompt": "What is 1+1?", "label": "2"},
                    {"id": "row_2", "prompt": "What is 2+2?", "label": "4"},
                ],
            )

            with patch(
                "nanoeval.backend.online_ray.OnlineRayBatchInferenceEngine",
                _FakeOnlineRayBatchInferenceEngine,
            ):
                summary = run_inference(
                    backend="online_ray",
                    input_file=input_path,
                    output_file=output_path,
                    sampling_params={"temperature": 0.3, "max_tokens": 64},
                    api_key="dummy",
                    base_url="https://example.com/v1",
                    model="dummy-model",
                    concurrency=16,
                    ray_num_actors=4,
                    ray_worker_concurrency=6,
                )

            self.assertEqual(summary["backend"], "online_ray")
            self.assertEqual(summary["ray_num_actors"], 4)
            self.assertEqual(summary["ray_worker_concurrency"], 6)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8") as file_obj:
                rows = [json.loads(line) for line in file_obj if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertIn("response", rows[0])
            self.assertEqual(rows[0]["sampling_echo"]["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
