from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanoeval.utils.args import parse_task_names


class TestArgs(unittest.TestCase):
    def test_parse_task_names_supports_jsonl_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            (task_dir / "aime2024.jsonl").write_text("{}", encoding="utf-8")

            names = parse_task_names(tasks_arg="aime2024.jsonl", task_dir=task_dir)
            self.assertEqual(names, ["aime2024"])


if __name__ == "__main__":
    unittest.main()
