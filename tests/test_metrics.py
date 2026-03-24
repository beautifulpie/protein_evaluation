"""Unit tests for alignment and core metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from complex_eval.align import apply_transform, compute_rmsd, kabsch_superimpose
from complex_eval.io_utils import AtomRecord, ResidueRecord
from complex_eval.metrics import (
    ComplexMatchResult,
    ResidueMatch,
    SideMatchResult,
    collect_interface_contacts,
    compute_fnat,
    compute_lddt_ca,
    dockq_score,
)


def make_atom(name: str, coord: tuple[float, float, float], element: str) -> AtomRecord:
    return AtomRecord(name=name, element=element, coord=np.array(coord, dtype=float), occupancy=1.0)


def make_residue(
    chain_id: str,
    resseq: int,
    resname: str,
    atom_specs: list[tuple[str, tuple[float, float, float], str]],
) -> ResidueRecord:
    atoms = {name: make_atom(name, coord, element) for name, coord, element in atom_specs}
    return ResidueRecord(
        chain_id=chain_id,
        resseq=resseq,
        insertion_code="",
        resname=resname,
        atoms=atoms,
        sequence_index=resseq - 1,
    )


def build_match_result(
    pred_receptor: list[ResidueRecord],
    gt_receptor: list[ResidueRecord],
    pred_ligand: list[ResidueRecord],
    gt_ligand: list[ResidueRecord],
) -> ComplexMatchResult:
    receptor_pairs = [
        ResidueMatch(pred=pred, gt=gt, side="receptor", pred_chain_id=pred.chain_id, gt_chain_id=gt.chain_id)
        for pred, gt in zip(pred_receptor, gt_receptor)
    ]
    ligand_pairs = [
        ResidueMatch(pred=pred, gt=gt, side="ligand", pred_chain_id=pred.chain_id, gt_chain_id=gt.chain_id)
        for pred, gt in zip(pred_ligand, gt_ligand)
    ]
    receptor = SideMatchResult(
        side="receptor",
        chain_mapping=[("A", "A")],
        matched_pairs=receptor_pairs,
        total_gt_residues=len(gt_receptor),
        total_pred_residues=len(pred_receptor),
        total_gt_heavy_atoms=sum(len(residue.heavy_atoms()) for residue in gt_receptor),
    )
    ligand = SideMatchResult(
        side="ligand",
        chain_mapping=[("B", "B")],
        matched_pairs=ligand_pairs,
        total_gt_residues=len(gt_ligand),
        total_pred_residues=len(pred_ligand),
        total_gt_heavy_atoms=sum(len(residue.heavy_atoms()) for residue in gt_ligand),
    )
    return ComplexMatchResult(receptor=receptor, ligand=ligand)


class AlignmentTests(unittest.TestCase):
    def test_kabsch_superimpose_recovers_rotation_and_translation(self) -> None:
        target = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([2.5, -1.0, 0.5])
        mobile = target @ rotation + translation

        result = kabsch_superimpose(mobile, target)
        aligned = apply_transform(mobile, result.rotation, result.translation)

        self.assertLess(result.rmsd, 1e-6)
        np.testing.assert_allclose(aligned, target, atol=1e-6)

    def test_compute_rmsd(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        b = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
        self.assertAlmostEqual(compute_rmsd(a, b), math.sqrt(0.5))


class MetricTests(unittest.TestCase):
    def test_fnat(self) -> None:
        gt_receptor = [
            make_residue(
                "A",
                1,
                "ALA",
                [
                    ("N", (0.0, 0.0, 0.0), "N"),
                    ("CA", (0.5, 0.0, 0.0), "C"),
                ],
            )
        ]
        gt_ligand = [
            make_residue(
                "B",
                1,
                "GLY",
                [
                    ("N", (4.0, 0.0, 0.0), "N"),
                    ("CA", (4.5, 0.0, 0.0), "C"),
                ],
            )
        ]
        pred_receptor = gt_receptor
        pred_ligand = gt_ligand

        native_contacts = collect_interface_contacts(gt_receptor, gt_ligand, cutoff=5.0)
        match_result = build_match_result(pred_receptor, gt_receptor, pred_ligand, gt_ligand)
        fnat, native_count, recovered_count = compute_fnat(native_contacts, match_result, contact_cutoff=5.0)

        self.assertEqual(native_count, 1)
        self.assertEqual(recovered_count, 1)
        self.assertEqual(fnat, 1.0)

    def test_dockq_formula(self) -> None:
        expected = (0.5 + 1.0 / (1.0 + (1.5 / 1.5) ** 2) + 1.0 / (1.0 + (8.5 / 8.5) ** 2)) / 3.0
        self.assertAlmostEqual(dockq_score(0.5, 1.5, 8.5), expected)

    def test_lddt_ca(self) -> None:
        gt_receptor = [
            make_residue("A", 1, "ALA", [("CA", (0.0, 0.0, 0.0), "C")]),
            make_residue("A", 2, "GLY", [("CA", (5.0, 0.0, 0.0), "C")]),
        ]
        gt_ligand = [
            make_residue("B", 1, "SER", [("CA", (0.0, 5.0, 0.0), "C")]),
        ]
        pred_receptor = [
            make_residue("A", 1, "ALA", [("CA", (0.1, 0.0, 0.0), "C")]),
            make_residue("A", 2, "GLY", [("CA", (5.1, 0.0, 0.0), "C")]),
        ]
        pred_ligand = [
            make_residue("B", 1, "SER", [("CA", (0.0, 5.1, 0.0), "C")]),
        ]

        match_result = build_match_result(pred_receptor, gt_receptor, pred_ligand, gt_ligand)
        self.assertAlmostEqual(compute_lddt_ca(match_result), 1.0)


if __name__ == "__main__":
    unittest.main()
