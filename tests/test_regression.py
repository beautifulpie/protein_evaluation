"""End-to-end regression and multiprocessing consistency tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from complex_eval.cli import main as cli_main

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmark"


class RegressionTests(unittest.TestCase):
    def test_regression_fixture_matches_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "results"
            exit_code = cli_main(
                [
                    "--manifest",
                    str(FIXTURE_DIR / "manifest.csv"),
                    "--out_dir",
                    str(out_dir),
                    "--workers",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)

            metrics_df = pd.read_csv(out_dir / "per_sample_metrics.csv").sort_values("sample_id").reset_index(drop=True)
            expected = json.loads((FIXTURE_DIR / "expected_per_sample.json").read_text(encoding="utf-8"))
            for row, expected_row in zip(metrics_df.to_dict(orient="records"), expected):
                self.assertEqual(row["sample_id"], expected_row["sample_id"])
                self.assertEqual(row["status"], expected_row["status"])
                self.assertAlmostEqual(float(row["fnat"]), expected_row["fnat"], places=6)
                self.assertAlmostEqual(float(row["dockq"]), expected_row["dockq"], places=6)
                self.assertEqual(int(row["num_matched_residues"]), expected_row["num_matched_residues"])
                self.assertAlmostEqual(float(row["matched_residue_fraction"]), expected_row["matched_residue_fraction"], places=6)

            summary = json.loads((out_dir / "summary_top1.json").read_text(encoding="utf-8"))
            expected_summary = json.loads((FIXTURE_DIR / "expected_summary_top1.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["count"], expected_summary["count"])
            self.assertEqual(summary["status_counts"], expected_summary["status_counts"])

    def test_workers_1_and_2_produce_identical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_one = tmp / "one"
            out_two = tmp / "two"
            exit_one = cli_main(
                [
                    "--manifest",
                    str(FIXTURE_DIR / "manifest.csv"),
                    "--out_dir",
                    str(out_one),
                    "--workers",
                    "1",
                ]
            )
            exit_two = cli_main(
                [
                    "--manifest",
                    str(FIXTURE_DIR / "manifest.csv"),
                    "--out_dir",
                    str(out_two),
                    "--workers",
                    "2",
                ]
            )
            self.assertEqual(exit_one, 0)
            self.assertEqual(exit_two, 0)

            columns = ["sample_id", "target_id", "rank", "status", "dockq", "fnat", "num_matched_residues"]
            df_one = pd.read_csv(out_one / "per_sample_metrics.csv")[columns].sort_values("sample_id").reset_index(drop=True)
            df_two = pd.read_csv(out_two / "per_sample_metrics.csv")[columns].sort_values("sample_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(df_one, df_two, check_exact=False, atol=1e-8, rtol=1e-8)

            failures_one = pd.read_csv(out_one / "failures.csv")
            failures_two = pd.read_csv(out_two / "failures.csv")
            pd.testing.assert_frame_equal(failures_one, failures_two)


if __name__ == "__main__":
    unittest.main()
