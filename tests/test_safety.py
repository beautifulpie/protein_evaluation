"""Safety-focused tests for chain/residue mapping and failure visibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complex_eval.evaluate import EvaluationConfig, safe_evaluate_record


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


class SafetyTests(unittest.TestCase):
    def test_identical_structures_are_success(self) -> None:
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
            self.assertTrue(outcome.ok)
            assert outcome.metrics is not None
            self.assertEqual(outcome.metrics["status"], "success")
            self.assertAlmostEqual(float(outcome.metrics["dockq"]), 1.0, places=6)
            self.assertEqual(int(outcome.metrics["num_matched_residues"]), 3)

    def test_chain_name_mismatch_is_visible_but_allowed_for_single_chain_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "TYR", (0.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            pred_residues = [
                ("X", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("Y", 1, "", "TYR", (0.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)
            outcome = safe_evaluate_record(
                {
                    "sample_id": "s1",
                    "target_id": "t1",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_receptor_chains": "X",
                    "pred_ligand_chains": "Y",
                    "gt_receptor_chains": "A",
                    "gt_ligand_chains": "B",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )
            assert outcome.metrics is not None
            self.assertEqual(outcome.metrics["status"], "success")
            self.assertTrue(bool(outcome.metrics["receptor_used_positional_chain_mapping"]))
            self.assertTrue(bool(outcome.metrics["ligand_used_positional_chain_mapping"]))

    def test_residue_numbering_offset_requires_sequence_fallback(self) -> None:
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

            without_fallback = safe_evaluate_record(
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
                config=EvaluationConfig(sequence_fallback=False),
                manifest_dir=tmp,
            )
            with_fallback = safe_evaluate_record(
                {
                    "sample_id": "s2",
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
            assert without_fallback.metrics is not None
            assert with_fallback.metrics is not None
            self.assertEqual(without_fallback.metrics["status"], "low_confidence_mapping")
            self.assertEqual(with_fallback.metrics["status"], "success")
            self.assertTrue(bool(with_fallback.metrics["used_sequence_fallback"]))

    def test_insertion_codes_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 2, "A", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
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
            self.assertEqual(outcome.metrics["status"], "success")
            self.assertEqual(int(outcome.metrics["receptor_num_matched_residues"]), 2)

    def test_missing_residue_is_flagged_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("A", 3, "", "SER", (6.0, 0.0, 0.0), ("N", "CA", "C", "O")),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("N", "CA", "C", "O")),
            ]
            pred_residues = gt_residues[:-2] + gt_residues[-1:]
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)
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
            self.assertEqual(outcome.metrics["status"], "low_confidence_mapping")

    def test_missing_backbone_atoms_trigger_ca_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            gt_residues = [
                ("A", 1, "", "ALA", (0.0, 0.0, 0.0), ("CA",)),
                ("A", 2, "", "GLY", (3.0, 0.0, 0.0), ("CA",)),
                ("B", 1, "", "TYR", (3.0, 0.0, 4.0), ("CA",)),
            ]
            pred_residues = gt_residues
            _write_backbone_pdb(gt_path, gt_residues)
            _write_backbone_pdb(pred_path, pred_residues)
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
            self.assertTrue(bool(outcome.metrics["used_ca_fallback_for_irmsd"]))
            self.assertTrue(bool(outcome.metrics["used_ca_fallback_for_lrmsd"]))

    def test_multi_chain_positional_mapping_is_low_confidence(self) -> None:
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
                    "sample_id": "s1",
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
            self.assertEqual(outcome.metrics["status"], "low_confidence_mapping")
            self.assertIn("receptor_multi_chain_positional_mapping", outcome.metrics["mapping_low_confidence_reasons"])

    def test_malformed_manifest_row_fails_loudly(self) -> None:
        outcome = safe_evaluate_record(
            {
                "sample_id": "s1",
                "target_id": "t1",
                "rank": "not-an-int",
                "pred_path": "",
                "gt_path": "missing.pdb",
                "pred_receptor_chains": "A",
                "pred_ligand_chains": "B",
                "gt_receptor_chains": "A",
                "gt_ligand_chains": "B",
            },
            config=EvaluationConfig(),
            manifest_dir=Path("."),
        )
        self.assertFalse(outcome.ok)
        assert outcome.failure is not None
        self.assertEqual(outcome.failure["status"], "failed")
        self.assertIn("blank required values", outcome.failure["error_message"])


if __name__ == "__main__":
    unittest.main()
