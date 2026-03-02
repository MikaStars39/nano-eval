from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
