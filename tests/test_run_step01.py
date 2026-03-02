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


class TestRunStep01(unittest.TestCase):
    def test_run_main_generates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_dir = tmp_path / "tasks"
            output_path = tmp_path / "output" / "prepared.jsonl"
            _write_jsonl(task_dir / "aime2024.jsonl", [{"prompt": "1+1=?", "label": "2"}])

            exit_code = main(
                [
                    "--task-dir",
                    str(task_dir),
                    "--tasks",
                    "aime2024",
                    "--pass-k",
                    "3",
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file_obj:
                records = [json.loads(line) for line in file_obj if line.strip()]

            self.assertEqual(len(records), 3)
            self.assertEqual([record["sample_index"] for record in records], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
