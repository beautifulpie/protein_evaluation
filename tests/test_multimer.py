"""Multimer evaluation tests."""

from __future__ import annotations

import math
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
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def _write_backbone_pdb(path: Path, residues: list[tuple[str, str, tuple[float, float, float]]]) -> None:
    lines: list[str] = []
    serial = 1
    atom_offsets = {
        "N": (-0.5, 0.0, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.5, 0.0, 0.0),
        "O": (1.0, 0.0, 0.0),
    }
    for chain_id, resname, base_xyz in residues:
        for atom_name, (dx, dy, dz) in atom_offsets.items():
            lines.append(
                _atom_line(
                    serial=serial,
                    atom_name=atom_name,
                    resname=resname,
                    chain_id=chain_id,
                    resseq=1,
                    x=base_xyz[0] + dx,
                    y=base_xyz[1] + dy,
                    z=base_xyz[2] + dz,
                )
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class MultimerTests(unittest.TestCase):
    def test_three_group_multimer_reports_pairwise_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            residues = [
                ("A", "ALA", (0.0, 0.0, 0.0)),
                ("B", "GLY", (0.0, 4.0, 0.0)),
                ("C", "SER", (3.0, 2.0, 0.0)),
            ]
            _write_backbone_pdb(gt_path, residues)
            _write_backbone_pdb(pred_path, residues)

            outcome = safe_evaluate_record(
                {
                    "sample_id": "m1",
                    "target_id": "Tmulti",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_chain_groups": "A|B|C",
                    "gt_chain_groups": "A|B|C",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )

            self.assertTrue(outcome.ok)
            assert outcome.metrics is not None
            self.assertEqual(outcome.metrics["status"], "success")
            self.assertEqual(outcome.metrics["evaluation_mode"], "multimer")
            self.assertEqual(int(outcome.metrics["num_chain_groups"]), 3)
            self.assertEqual(int(outcome.metrics["pairwise_interface_count"]), 3)
            self.assertAlmostEqual(float(outcome.metrics["pairwise_dockq_mean"]), 1.0, places=6)
            self.assertAlmostEqual(float(outcome.metrics["fnat"]), 1.0, places=6)
            self.assertTrue(math.isnan(float(outcome.metrics["dockq"])))

    def test_two_group_chain_group_manifest_uses_binary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "gt.pdb"
            pred_path = tmp / "pred.pdb"
            residues = [
                ("A", "ALA", (0.0, 0.0, 0.0)),
                ("B", "TYR", (0.0, 0.0, 4.0)),
            ]
            _write_backbone_pdb(gt_path, residues)
            _write_backbone_pdb(pred_path, residues)

            outcome = safe_evaluate_record(
                {
                    "sample_id": "b1",
                    "target_id": "Tbinary",
                    "rank": 1,
                    "pred_path": pred_path.name,
                    "gt_path": gt_path.name,
                    "pred_chain_groups": "A|B",
                    "gt_chain_groups": "A|B",
                },
                config=EvaluationConfig(),
                manifest_dir=tmp,
            )

            self.assertTrue(outcome.ok)
            assert outcome.metrics is not None
            self.assertEqual(outcome.metrics["evaluation_mode"], "binary")
            self.assertEqual(outcome.metrics["pred_chain_groups"], "A|B")
            self.assertAlmostEqual(float(outcome.metrics["dockq"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
