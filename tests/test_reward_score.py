from __future__ import annotations

import csv
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoeval.reward.reward import judge_router
from nanoeval.reward.score import eval_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestRewardScore(unittest.TestCase):
    def test_eval_results_outputs_metrics_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_input = tmp_path / "eval_input.jsonl"
            score_output = tmp_path / "score_output.jsonl"
            metrics_output = tmp_path / "final_metrics.jsonl"
            metrics_csv_output = tmp_path / "final_metrics.csv"
            _write_jsonl(
                eval_input,
                [
                    {"source": "aime2024", "question_id": "q1", "response": "ok", "thinking": "plan now", "label": "x"},
                    {"source": "aime2024", "question_id": "q1", "response": "bad", "label": "x"},
                    {"source": "aime2024", "question_id": "q2", "response": "bad", "thinking": "deep think.", "label": "x"},
                    {"source": "aime2025", "question_id": "q3", "response": "ok", "thinking": "h", "label": "x"},
                ],
            )

            def _fake_judge_router(response: str, label: str = "", source: str | None = None, **kwargs: object) -> dict:
                return {"pass": response == "ok", "pred": response}

            with patch("nanoeval.reward.score.judge_router", side_effect=_fake_judge_router):
                metrics = eval_results(
                    eval_output_file=eval_input,
                    score_output_file=score_output,
                    final_eval_output_file=metrics_output,
                    final_eval_csv_output_file=metrics_csv_output,
                    n_proc=1,
                )

            self.assertAlmostEqual(metrics["aime2024"]["avg_k"], 1 / 3)
            self.assertAlmostEqual(metrics["aime2024"]["pass_k"], 1 / 2)
            self.assertAlmostEqual(metrics["aime2025"]["avg_k"], 1.0)
            self.assertAlmostEqual(metrics["aime2025"]["pass_k"], 1.0)
            self.assertAlmostEqual(metrics["overall"]["avg_k"], 0.5)
            self.assertAlmostEqual(metrics["overall"]["pass_k"], 2 / 3)
            self.assertAlmostEqual(metrics["aime2024"]["avg_total_tokens"], 8 / 3)
            self.assertAlmostEqual(metrics["aime2024"]["avg_thinking_tokens"], 5 / 3)
            self.assertAlmostEqual(metrics["aime2024"]["max_thinking_tokens"], 3.0)
            self.assertAlmostEqual(metrics["aime2024"]["min_thinking_tokens"], 0.0)
            self.assertAlmostEqual(metrics["aime2025"]["avg_total_tokens"], 2.0)
            self.assertAlmostEqual(metrics["aime2025"]["avg_thinking_tokens"], 1.0)
            self.assertAlmostEqual(metrics["aime2025"]["max_thinking_tokens"], 1.0)
            self.assertAlmostEqual(metrics["aime2025"]["min_thinking_tokens"], 1.0)
            self.assertAlmostEqual(metrics["overall"]["avg_total_tokens"], 2.5)
            self.assertAlmostEqual(metrics["overall"]["avg_thinking_tokens"], 1.5)
            self.assertAlmostEqual(metrics["overall"]["max_thinking_tokens"], 3.0)
            self.assertAlmostEqual(metrics["overall"]["min_thinking_tokens"], 0.0)

            with score_output.open("r", encoding="utf-8") as file_obj:
                scored_rows = [json.loads(line) for line in file_obj if line.strip()]
            self.assertEqual(len(scored_rows), 4)
            q1_rows = [row for row in scored_rows if row["source"] == "aime2024" and row["question_id"] == "q1"]
            self.assertTrue(all(row["pass_at_k"] for row in q1_rows))

            with metrics_csv_output.open("r", encoding="utf-8") as file_obj:
                csv_rows = list(csv.DictReader(file_obj))
            self.assertTrue(any(row["task"] == "aime2024" for row in csv_rows))
            self.assertTrue(any(row["task"] == "overall" for row in csv_rows))
            aime2024_row = next(row for row in csv_rows if row["task"] == "aime2024")
            self.assertAlmostEqual(float(aime2024_row["avg_total_tokens"]), 8 / 3)
            self.assertAlmostEqual(float(aime2024_row["avg_thinking_tokens"]), 5 / 3)
            self.assertAlmostEqual(float(aime2024_row["max_thinking_tokens"]), 3.0)
            self.assertAlmostEqual(float(aime2024_row["min_thinking_tokens"]), 0.0)

    def test_eval_results_strips_thinking_from_response_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_input = tmp_path / "eval_input.jsonl"
            score_output = tmp_path / "score_output.jsonl"
            metrics_output = tmp_path / "final_metrics.jsonl"
            metrics_csv_output = tmp_path / "final_metrics.csv"
            _write_jsonl(
                eval_input,
                [
                    {
                        "source": "ifeval",
                        "question_id": "q1",
                        "response": (
                            "Thinking Process:\n"
                            "I should ensure the format is strict.\n\n"
                            "Final answer: My answer is no."
                        ),
                        "instruction_id_list": ["detectable_format:constrained_response"],
                        "kwargs": [{}],
                        "label": "",
                    }
                ],
            )

            def _fake_judge_router(response: str, label: str = "", source: str | None = None, **kwargs: object) -> dict:
                return {"pass": response == "My answer is no.", "pred": response}

            with patch("nanoeval.reward.score.judge_router", side_effect=_fake_judge_router):
                metrics = eval_results(
                    eval_output_file=eval_input,
                    score_output_file=score_output,
                    final_eval_output_file=metrics_output,
                    final_eval_csv_output_file=metrics_csv_output,
                    n_proc=1,
                )

            self.assertAlmostEqual(metrics["ifeval"]["avg_k"], 1.0)
            self.assertGreater(metrics["ifeval"]["avg_thinking_tokens"], 0.0)

            with score_output.open("r", encoding="utf-8") as file_obj:
                scored_rows = [json.loads(line) for line in file_obj if line.strip()]
            self.assertEqual(len(scored_rows), 1)
            self.assertEqual(scored_rows[0]["response"], "My answer is no.")
            self.assertIn("Thinking Process", scored_rows[0].get("thinking", ""))
            self.assertTrue(scored_rows[0]["pass"])

    def test_judge_router_routes_to_expected_branches(self) -> None:
        fake_ifeval_module = types.ModuleType("nanoeval.reward.if_eval.if_eval")
        fake_ifeval_module.if_judge = lambda response, **kwargs: {"pass": True, "pred": "ifeval"}
        fake_gpqa_module = types.ModuleType("nanoeval.reward.gpqa.gpqa_verify_reward")
        fake_gpqa_module.gpqa_judge = lambda response, label="", **kwargs: {"pass": True, "pred": "gpqa"}

        with patch.dict(
            "sys.modules",
            {
                "nanoeval.reward.if_eval.if_eval": fake_ifeval_module,
                "nanoeval.reward.gpqa.gpqa_verify_reward": fake_gpqa_module,
            },
        ):
            ifeval_result = judge_router(response="x", source="ifeval-v1")
            gpqa_result = judge_router(response="x", label="A", source="gpqa")

        self.assertIsInstance(ifeval_result, dict)
        self.assertTrue(ifeval_result["pass"])
        self.assertEqual(ifeval_result["pred"], "ifeval")
        self.assertIsInstance(gpqa_result, dict)
        self.assertTrue(gpqa_result["pass"])
        self.assertEqual(gpqa_result["pred"], "gpqa")

        with patch("nanoeval.reward.reward.math_judge", return_value={"pass": True, "pred": "math"}) as mock_math:
            math_result = judge_router(response="x", label="2", source="math")
            mock_math.assert_called_once()
        self.assertIsInstance(math_result, dict)
        self.assertTrue(math_result["pass"])
        self.assertEqual(math_result["pred"], "math")


if __name__ == "__main__":
    unittest.main()
