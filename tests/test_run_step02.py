from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestRunStep02(unittest.TestCase):
    def test_run_main_step02_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "prepared_input.jsonl"
            output_path = tmp_path / "inference_output.jsonl"
            _write_jsonl(input_path, [{"id": "q1_0", "prompt": "test prompt", "label": "A"}])

            exit_code = main(
                [
                    "--stage",
                    "step02",
                    "--input",
                    str(input_path),
                    "--backend",
                    "mock",
                    "--inference-output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file_obj:
                rows = [json.loads(line) for line in file_obj if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "q1_0")
            self.assertIn("response", rows[0])


if __name__ == "__main__":
    unittest.main()
