"""Explainability and diagnostics tests."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from complex_eval.evaluate import EvaluationConfig, safe_evaluate_record
from complex_eval.metrics import interface_contact_metrics


def _atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    insertion_code: str = "",
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}{(insertion_code or ' '):1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def _write_backbone_pdb(
    path: Path,
    residues: list[tuple[str, int, str, str, tuple[float, float, float], tuple[str, ...]]],
) -> None:
    lines: list[str] = []
    serial = 1
    atom_offsets = {
        "N": (-0.5, 0.0, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.5, 0.0, 0.0),
        "O": (1.0, 0.0, 0.0),
    }
    for chain_id, resseq, insertion_code, resname, base_xyz, atoms in residues:
        for atom_name in atoms:
            dx, dy, dz = atom_offsets[atom_name]
            lines.append(
                _atom_line(
                    serial,
                    atom_name,
                    resname,
                    chain_id,
                    resseq,
                    base_xyz[0] + dx,
                    base_xyz[1] + dy,
                    base_xyz[2] + dz,
                    insertion_code=insertion_code,
                )
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DiagnosticsTests(unittest.TestCase):
    def test_mapping_confidence_perfect_mapping_is_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            _write_backbone_pdb(gt_path, residues)
            _write_backbone_pdb(pred_path, residues)
            outcome = safe_evaluate_record(
                {
                    "sample_id": "s1",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "A",
                    "pred_ligand_chains": "B",
                    "gt_receptor_chains": "A",
                    "gt_ligand_chains": "B",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )
            assert outcome.metrics is not None
            self.assertEqual(outcome.metrics["mapping_confidence_label"], "high")
            self.assertAlmostEqual(float(outcome.metrics["mapping_confidence_score"]), 1.0, places=6)
            self.assertIn("clean_mapping", outcome.metrics["diagnostic_tags"])

    def test_numbering_offset_sequence_resolved_is_lower_than_perfect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            pred_residues = [
                ("A", 101, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 102, "", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 101, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)

            perfect = safe_evaluate_record(
                {
                    "sample_id": "perfect",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": gt_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "A",
                    "pred_ligand_chains": "B",
                    "gt_receptor_chains": "A",
                    "gt_ligand_chains": "B",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )
            rescued = safe_evaluate_record(
                {
                    "sample_id": "rescued",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "A",
                    "pred_ligand_chains": "B",
                    "gt_receptor_chains": "A",
                    "gt_ligand_chains": "B",
                },
                config=EvaluationConfig(sequence_fallback=True),
                manifest_dir=tmp,
            )
            assert perfect.metrics is not None
            assert rescued.metrics is not None
            self.assertLess(
                float(rescued.metrics["mapping_confidence_score"]),
                float(perfect.metrics["mapping_confidence_score"]),
            )
            self.assertEqual(rescued.metrics["mapping_confidence_label"], "high")
            self.assertIn("used_sequence_fallback", rescued.metrics["diagnostic_tags"])
            self.assertIn("likely_numbering_offset", rescued.metrics["diagnostic_tags"])

    def test_chain_mismatch_with_poor_coverage_is_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "GLY", (6.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("C", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            pred_residues = [
                ("X", 1, "", "GLY", (6.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("Y", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("Z", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)
            outcome = safe_evaluate_record(
                {
                    "sample_id": "swap",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "X,Y",
                    "pred_ligand_chains": "Z",
                    "gt_receptor_chains": "A,B",
                    "gt_ligand_chains": "C",
                },
                config=EvaluationConfig(sequence_fallback=True),
                manifest_dir=tmp,
            )
            assert outcome.metrics is not None
            self.assertLess(float(outcome.metrics["mapping_confidence_score"]), 0.60)
            self.assertEqual(outcome.metrics["mapping_confidence_label"], "low")
            self.assertIn("ambiguous_chain_mapping", outcome.metrics["diagnostic_tags"])

    def test_sparse_atom_coverage_is_tagged_and_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            pred_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("CA",)),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("CA",)),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("CA",)),
            ]
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)
            outcome = safe_evaluate_record(
                {
                    "sample_id": "sparse",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "A",
                    "pred_ligand_chains": "B",
                    "gt_receptor_chains": "A",
                    "gt_ligand_chains": "B",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )
            assert outcome.metrics is not None
            self.assertLess(float(outcome.metrics["mapping_confidence_score"]), 0.60)
            self.assertIn("sparse_atom_coverage", outcome.metrics["diagnostic_tags"])
            self.assertIn("used_ca_fallback_irmsd", outcome.metrics["diagnostic_tags"])

    def test_interface_metrics_perfect_partial_false_and_empty(self) -> None:
        perfect = interface_contact_metrics({("A", "B"), ("C", "D")}, {("A", "B"), ("C", "D")})
        self.assertEqual(perfect["interface_native_contact_count"], 2)
        self.assertEqual(perfect["interface_pred_contact_count"], 2)
        self.assertEqual(perfect["interface_false_contact_count"], 0)
        self.assertAlmostEqual(float(perfect["interface_precision"]), 1.0, places=6)
        self.assertAlmostEqual(float(perfect["interface_recall"]), 1.0, places=6)
        self.assertAlmostEqual(float(perfect["interface_f1"]), 1.0, places=6)

        partial = interface_contact_metrics({("A", "B"), ("C", "D")}, {("A", "B")})
        self.assertEqual(partial["interface_missing_contact_count"], 1)
        self.assertAlmostEqual(float(partial["interface_precision"]), 1.0, places=6)
        self.assertAlmostEqual(float(partial["interface_recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(partial["interface_f1"]), 2.0 / 3.0, places=6)

        false_heavy = interface_contact_metrics({("A", "B")}, {("A", "B"), ("C", "D"), ("E", "F")})
        self.assertEqual(false_heavy["interface_false_contact_count"], 2)
        self.assertAlmostEqual(float(false_heavy["interface_precision"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(false_heavy["interface_recall"]), 1.0, places=6)

        empty = interface_contact_metrics(set(), set())
        self.assertTrue(math.isnan(float(empty["interface_precision"])))
        self.assertTrue(math.isnan(float(empty["interface_recall"])))
        self.assertTrue(math.isnan(float(empty["interface_f1"])))


if __name__ == "__main__":
    unittest.main()
