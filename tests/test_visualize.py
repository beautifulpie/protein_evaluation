"""Regression tests for visualization outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complex_eval.cli import main as cli_main

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmark"


class VisualizationTests(unittest.TestCase):
    def test_visualization_outputs_are_written_by_default(self) -> None:
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

            report_path = out_dir / "report.html"
            self.assertTrue(report_path.exists())
            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("complex_eval diagnostic report", report_html)
            self.assertIn("Prediction performance overview", report_html)
            self.assertIn("Benchmark performance summary", report_html)
            self.assertIn("Top predicted samples", report_html)
            self.assertIn("Detailed metric gallery", report_html)
            self.assertIn("plots/status_counts.svg", report_html)
            self.assertIn("plots/metric_ca_rmsd.svg", report_html)
            self.assertIn("plots/metric_dockq.svg", report_html)
            self.assertIn("plots/metric_interface_f1.svg", report_html)

            expected_plot_files = [
                "status_counts.svg",
                "confidence_label_counts.svg",
                "diagnostic_tag_counts.svg",
                "performance_snapshot.svg",
                "method_mean_dockq.svg",
                "mapping_confidence_vs_dockq.svg",
                "interface_precision_vs_recall.svg",
                "metric_ca_rmsd.svg",
                "metric_all_atom_rmsd.svg",
                "metric_irmsd.svg",
                "metric_lrmsd.svg",
                "metric_fnat.svg",
                "metric_dockq.svg",
                "metric_lddt_ca.svg",
                "metric_clash_count.svg",
                "metric_clashes_per_1000_atoms.svg",
                "metric_interface_precision.svg",
                "metric_interface_recall.svg",
                "metric_interface_f1.svg",
            ]
            for filename in expected_plot_files:
                plot_path = out_dir / "plots" / filename
                self.assertTrue(plot_path.exists(), msg=f"missing plot {filename}")
                self.assertIn("<svg", plot_path.read_text(encoding="utf-8"))

    def test_visualizations_can_be_disabled(self) -> None:
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
                    "--no-write_visualizations",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse((out_dir / "report.html").exists())
            self.assertFalse((out_dir / "plots").exists())

    def test_visualizations_still_work_without_explainability_fields_in_csv(self) -> None:
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
                    "--no-include_explainability_fields",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "report.html").exists())
            self.assertTrue((out_dir / "plots" / "mapping_confidence_vs_dockq.svg").exists())


if __name__ == "__main__":
    unittest.main()
