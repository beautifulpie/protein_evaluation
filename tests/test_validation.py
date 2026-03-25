"""Validation module tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from complex_eval.validation import parse_dockq_output, validate_against_dockq

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmark"


class ValidationTests(unittest.TestCase):
    def test_parse_dockq_output(self) -> None:
        parsed = parse_dockq_output("DockQ 1.0\nFnat 1.0\niRMSD 0.0\nLRMSD 0.0\n")
        self.assertEqual(parsed, {"dockq": 1.0, "fnat": 1.0, "irmsd": 0.0, "lrmsd": 0.0})

    def test_validation_gracefully_handles_missing_dockq(self) -> None:
        comparisons, summary = validate_against_dockq(pd.DataFrame(), dockq_executable="/definitely/missing/DockQ")
        self.assertTrue(comparisons.empty)
        self.assertEqual(summary["status"], "unavailable")

    def test_validation_against_fake_dockq(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dockq_path = tmp / "fake_dockq.py"
            dockq_path.write_text(
                "#!/usr/bin/env python3\n"
                "print('DockQ 1.0')\n"
                "print('Fnat 1.0')\n"
                "print('iRMSD 0.0')\n"
                "print('LRMSD 0.0')\n",
                encoding="utf-8",
            )
            os.chmod(dockq_path, 0o755)
            metrics_df = pd.DataFrame(
                [
                    {
                        "sample_id": "identical",
                        "target_id": "T001",
                        "rank": 1,
                        "status": "success",
                        "pred_path": str(FIXTURE_DIR / "pred_identical.pdb"),
                        "gt_path": str(FIXTURE_DIR / "native.pdb"),
                        "pred_receptor_chains": "A",
                        "pred_ligand_chains": "B",
                        "gt_receptor_chains": "A",
                        "gt_ligand_chains": "B",
                        "dockq": 1.0,
                        "fnat": 1.0,
                        "irmsd": 0.0,
                        "lrmsd": 0.0,
                    }
                ]
            )
            comparisons, summary = validate_against_dockq(metrics_df, dockq_executable=str(dockq_path))
            self.assertEqual(comparisons.iloc[0]["status"], "validated")
            self.assertAlmostEqual(float(comparisons.iloc[0]["dockq_diff"]), 0.0, places=6)
            self.assertEqual(summary["status"], "ok")
            self.assertTrue(summary["pass"])


if __name__ == "__main__":
    unittest.main()
