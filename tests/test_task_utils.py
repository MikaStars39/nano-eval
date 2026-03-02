from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanoeval.utils.task import discover_task_names, prepare_eval_input


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestTaskUtils(unittest.TestCase):
    def test_prepare_eval_input_with_pass_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_dir = tmp_path / "tasks"
            output_path = tmp_path / "artifacts" / "prepared.jsonl"

            _write_jsonl(
                task_dir / "aime2024.jsonl",
                [
                    {"question_id": "q1", "prompt": "2+2=?", "label": "4"},
                    {"question_id": "q2", "prompt": "3+5=?", "label": "8"},
                ],
            )

            summary = prepare_eval_input(
                task_names=["aime2024"],
                task_dir=task_dir,
                pass_k=2,
                output_path=output_path,
            )
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(summary["instance_count"], 4)
            self.assertEqual(summary["task_sizes"]["aime2024"], 4)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file_obj:
                lines = [json.loads(line) for line in file_obj if line.strip()]

            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0]["id"], "q1_0")
            self.assertEqual(lines[1]["id"], "q1_1")
            self.assertEqual(lines[2]["id"], "q2_0")
            self.assertEqual(lines[3]["id"], "q2_1")
            self.assertTrue(all(item["source"] == "aime2024" for item in lines))

    def test_discover_task_names_uses_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_dir = tmp_path / "tasks"
            _write_jsonl(task_dir / "aime2024.jsonl", [{"prompt": "x", "label": "y"}])
            _write_jsonl(task_dir / "not_in_mapping.jsonl", [{"prompt": "x", "label": "y"}])

            discovered = discover_task_names(task_dir)
            self.assertEqual(discovered, ["aime2024"])

    def test_expand_records_uses_id_as_question_id_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_dir = tmp_path / "tasks"
            output_path = tmp_path / "artifacts" / "prepared.jsonl"
            _write_jsonl(task_dir / "aime2024.jsonl", [{"id": "origin_1", "prompt": "x", "label": "y"}])

            prepare_eval_input(
                task_names=["aime2024"],
                task_dir=task_dir,
                pass_k=2,
                output_path=output_path,
            )

            with output_path.open("r", encoding="utf-8") as file_obj:
                rows = [json.loads(line) for line in file_obj if line.strip()]

            self.assertEqual(rows[0]["question_id"], "origin_1")
            self.assertEqual(rows[0]["id"], "origin_1_0")


if __name__ == "__main__":
    unittest.main()
