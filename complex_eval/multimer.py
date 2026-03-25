"""Multimer-safe evaluation helpers built on top of binary metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable

from .io_utils import StructureRecord
from .metrics import (
    ResidueMatch,
    SideMatchResult,
    _merge_warnings,
    compute_all_heavy_atom_rmsd,
    compute_global_ca_rmsd,
    compute_lddt_ca,
    count_steric_clashes,
    evaluate_binary_complex,
    match_side_residues,
)


@dataclass(frozen=True)
class MatchCollection:
    """Minimal match container for global multimer metrics."""

    all_pairs: list[ResidueMatch]


def evaluate_multimer_complex(
    pred_structure: StructureRecord,
    gt_structure: StructureRecord,
    pred_chain_groups: list[list[str]],
    gt_chain_groups: list[list[str]],
    contact_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    include_all_atom_rmsd: bool = True,
    include_lddt: bool = True,
    include_clashes: bool = True,
    sequence_fallback: bool = False,
) -> dict[str, object]:
    """Evaluate a multimer using full-complex metrics plus pairwise interfaces.

    Scientific safety note:
    For complexes with more than two manifest-defined groups, there is no single
    canonical DockQ/iRMSD/LRMSD value implemented here. This function therefore
    reports:

    - full-complex metrics: CA RMSD, all-atom RMSD, lDDT-CA, clashes, coverage
    - pairwise interface metrics for each group pair
    - aggregate pairwise summaries such as mean/min DockQ across group pairs

    The legacy scalar fields `dockq`, `irmsd`, and `lrmsd` are left as NaN in
    true multimer mode to avoid implying a validated single-number equivalence.
    """

    if len(pred_chain_groups) != len(gt_chain_groups):
        raise ValueError(
            "Prediction and ground-truth multimer chain groups must have the same number of groups; "
            f"received pred={len(pred_chain_groups)} gt={len(gt_chain_groups)}."
        )
    if len(pred_chain_groups) < 3:
        raise ValueError(
            "evaluate_multimer_complex is intended for three or more chain groups. "
            "Use the binary evaluation path for two-group problems."
        )

    group_results: list[SideMatchResult] = []
    all_pairs: list[ResidueMatch] = []
    total_gt_residues = 0
    total_gt_heavy_atoms = 0

    for group_index, (pred_group, gt_group) in enumerate(zip(pred_chain_groups, gt_chain_groups), start=1):
        side_result = match_side_residues(
            pred_structure=pred_structure,
            gt_structure=gt_structure,
            pred_chain_ids=pred_group,
            gt_chain_ids=gt_group,
            side=f"group_{group_index}",
            allow_sequence_fallback=sequence_fallback,
        )
        group_results.append(side_result)
        all_pairs.extend(side_result.matched_pairs)
        total_gt_residues += side_result.total_gt_residues
        total_gt_heavy_atoms += side_result.total_gt_heavy_atoms

    match_collection = MatchCollection(all_pairs=all_pairs)
    num_matched_residues = len(all_pairs)
    matched_residue_fraction = num_matched_residues / max(1, total_gt_residues)

    heavy_rmsd, num_matched_atoms = compute_all_heavy_atom_rmsd(match_collection)
    matched_atom_fraction = num_matched_atoms / max(1, total_gt_heavy_atoms)

    pred_all_residues = pred_structure.get_residues(_flatten_groups(pred_chain_groups))
    clash_count, clashes_per_1000_atoms = (
        count_steric_clashes(pred_all_residues) if include_clashes else (math.nan, math.nan)
    )

    pairwise_rows = _evaluate_pairwise_interfaces(
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        pred_chain_groups=pred_chain_groups,
        gt_chain_groups=gt_chain_groups,
        contact_cutoff=contact_cutoff,
        interface_cutoff=interface_cutoff,
        sequence_fallback=sequence_fallback,
    )
    pairwise_summary = _summarize_pairwise_rows(pairwise_rows)
    group_mapping = _aggregate_group_mapping(group_results)

    return {
        "evaluation_mode": "multimer",
        "num_chain_groups": len(pred_chain_groups),
        "ca_rmsd": compute_global_ca_rmsd(match_collection),
        "all_atom_rmsd": heavy_rmsd if include_all_atom_rmsd else math.nan,
        "irmsd": math.nan,
        "lrmsd": math.nan,
        "fnat": pairwise_summary["overall_fnat"],
        "dockq": math.nan,
        "dockq_acceptable": False,
        "dockq_medium": False,
        "dockq_high": False,
        "dockq_decomposition_fnat_term": math.nan,
        "dockq_decomposition_irmsd_term": math.nan,
        "dockq_decomposition_lrmsd_term": math.nan,
        "lddt_ca": compute_lddt_ca(match_collection) if include_lddt else math.nan,
        "clash_count": clash_count,
        "clashes_per_1000_atoms": clashes_per_1000_atoms,
        "matched_residue_fraction": matched_residue_fraction,
        "matched_atom_fraction": matched_atom_fraction,
        "num_matched_residues": num_matched_residues,
        "num_matched_atoms": num_matched_atoms,
        "num_native_contacts": pairwise_summary["total_native_contacts"],
        "num_recovered_contacts": pairwise_summary["total_recovered_contacts"],
        "used_sequence_fallback": any(result.used_sequence_fallback for result in group_results),
        "used_ca_fallback_for_irmsd": any(bool(row["used_ca_fallback_for_irmsd"]) for row in pairwise_rows),
        "used_ca_fallback_for_lrmsd": any(bool(row["used_ca_fallback_for_lrmsd"]) for row in pairwise_rows),
        "parse_warning": _merge_warnings(
            pred_structure.warnings
            + gt_structure.warnings
            + [warning for result in group_results for warning in result.warnings]
        ),
        "pairwise_interface_count": len(pairwise_rows),
        "pairwise_interfaces_with_native_contacts": pairwise_summary["interfaces_with_native_contacts"],
        "pairwise_fnat_mean": pairwise_summary["pairwise_fnat_mean"],
        "pairwise_irmsd_mean": pairwise_summary["pairwise_irmsd_mean"],
        "pairwise_lrmsd_mean": pairwise_summary["pairwise_lrmsd_mean"],
        "pairwise_dockq_mean": pairwise_summary["pairwise_dockq_mean"],
        "pairwise_dockq_min": pairwise_summary["pairwise_dockq_min"],
        "pairwise_dockq_acceptable_fraction": pairwise_summary["pairwise_dockq_acceptable_fraction"],
        "interface_native_contact_count": pairwise_summary["total_native_contacts"],
        "interface_pred_contact_count": pairwise_summary["total_pred_contacts"],
        "interface_recovered_contact_count": pairwise_summary["total_recovered_contacts"],
        "interface_missing_contact_count": pairwise_summary["total_missing_contacts"],
        "interface_false_contact_count": pairwise_summary["total_false_contacts"],
        "interface_precision": pairwise_summary["overall_interface_precision"],
        "interface_recall": pairwise_summary["overall_interface_recall"],
        "interface_f1": pairwise_summary["overall_interface_f1"],
        "pairwise_interface_metrics_json": json.dumps(
            _json_safe_rows(pairwise_rows),
            sort_keys=True,
            allow_nan=False,
        ),
        **group_mapping,
    }


def _evaluate_pairwise_interfaces(
    pred_structure: StructureRecord,
    gt_structure: StructureRecord,
    pred_chain_groups: list[list[str]],
    gt_chain_groups: list[list[str]],
    contact_cutoff: float,
    interface_cutoff: float,
    sequence_fallback: bool,
) -> list[dict[str, object]]:
    """Evaluate all unique group-group interfaces using the binary metric path."""

    rows: list[dict[str, object]] = []
    for left_index in range(len(pred_chain_groups)):
        for right_index in range(left_index + 1, len(pred_chain_groups)):
            pair_metrics = evaluate_binary_complex(
                pred_structure=pred_structure,
                gt_structure=gt_structure,
                pred_receptor_chains=pred_chain_groups[left_index],
                pred_ligand_chains=pred_chain_groups[right_index],
                gt_receptor_chains=gt_chain_groups[left_index],
                gt_ligand_chains=gt_chain_groups[right_index],
                contact_cutoff=contact_cutoff,
                interface_cutoff=interface_cutoff,
                include_all_atom_rmsd=False,
                include_lddt=False,
                include_clashes=False,
                sequence_fallback=sequence_fallback,
            )
            rows.append(
                {
                    "group_pair": f"{left_index + 1}-{right_index + 1}",
                    "pred_group_left": ",".join(pred_chain_groups[left_index]),
                    "pred_group_right": ",".join(pred_chain_groups[right_index]),
                    "gt_group_left": ",".join(gt_chain_groups[left_index]),
                    "gt_group_right": ",".join(gt_chain_groups[right_index]),
                    "fnat": pair_metrics["fnat"],
                    "irmsd": pair_metrics["irmsd"],
                    "lrmsd": pair_metrics["lrmsd"],
                    "dockq": pair_metrics["dockq"],
                    "num_native_contacts": pair_metrics["num_native_contacts"],
                    "num_recovered_contacts": pair_metrics["num_recovered_contacts"],
                    "interface_native_contact_count": pair_metrics["interface_native_contact_count"],
                    "interface_pred_contact_count": pair_metrics["interface_pred_contact_count"],
                    "interface_recovered_contact_count": pair_metrics["interface_recovered_contact_count"],
                    "interface_missing_contact_count": pair_metrics["interface_missing_contact_count"],
                    "interface_false_contact_count": pair_metrics["interface_false_contact_count"],
                    "interface_precision": pair_metrics["interface_precision"],
                    "interface_recall": pair_metrics["interface_recall"],
                    "interface_f1": pair_metrics["interface_f1"],
                    "used_ca_fallback_for_irmsd": pair_metrics["used_ca_fallback_for_irmsd"],
                    "used_ca_fallback_for_lrmsd": pair_metrics["used_ca_fallback_for_lrmsd"],
                    "mapping_confidence_score": pair_metrics.get("mapping_confidence_score", math.nan),
                }
            )
    return rows


def _summarize_pairwise_rows(pairwise_rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Aggregate pairwise interface metrics across all group pairs."""

    rows = list(pairwise_rows)
    total_native_contacts = sum(int(row.get("num_native_contacts", 0) or 0) for row in rows)
    total_recovered_contacts = sum(int(row.get("num_recovered_contacts", 0) or 0) for row in rows)
    total_pred_contacts = sum(int(row.get("interface_pred_contact_count", 0) or 0) for row in rows)
    total_missing_contacts = sum(int(row.get("interface_missing_contact_count", 0) or 0) for row in rows)
    total_false_contacts = sum(int(row.get("interface_false_contact_count", 0) or 0) for row in rows)
    overall_fnat = total_recovered_contacts / total_native_contacts if total_native_contacts > 0 else math.nan
    overall_interface_precision = (
        total_recovered_contacts / total_pred_contacts if total_pred_contacts > 0 else math.nan
    )
    overall_interface_recall = (
        total_recovered_contacts / total_native_contacts if total_native_contacts > 0 else math.nan
    )
    if (
        math.isnan(overall_interface_precision)
        or math.isnan(overall_interface_recall)
        or overall_interface_precision + overall_interface_recall == 0.0
    ):
        overall_interface_f1 = math.nan
    else:
        overall_interface_f1 = (
            2.0 * overall_interface_precision * overall_interface_recall
            / (overall_interface_precision + overall_interface_recall)
        )

    pairwise_dockq_values = _finite_values(row.get("dockq") for row in rows)
    pairwise_fnat_values = _finite_values(row.get("fnat") for row in rows)
    pairwise_irmsd_values = _finite_values(row.get("irmsd") for row in rows)
    pairwise_lrmsd_values = _finite_values(row.get("lrmsd") for row in rows)

    return {
        "total_native_contacts": total_native_contacts,
        "total_recovered_contacts": total_recovered_contacts,
        "total_pred_contacts": total_pred_contacts,
        "total_missing_contacts": total_missing_contacts,
        "total_false_contacts": total_false_contacts,
        "overall_fnat": overall_fnat,
        "overall_interface_precision": overall_interface_precision,
        "overall_interface_recall": overall_interface_recall,
        "overall_interface_f1": overall_interface_f1,
        "interfaces_with_native_contacts": sum(int(row.get("num_native_contacts", 0) or 0) > 0 for row in rows),
        "pairwise_fnat_mean": _mean_or_nan(pairwise_fnat_values),
        "pairwise_irmsd_mean": _mean_or_nan(pairwise_irmsd_values),
        "pairwise_lrmsd_mean": _mean_or_nan(pairwise_lrmsd_values),
        "pairwise_dockq_mean": _mean_or_nan(pairwise_dockq_values),
        "pairwise_dockq_min": min(pairwise_dockq_values) if pairwise_dockq_values else math.nan,
        "pairwise_dockq_acceptable_fraction": (
            sum(value >= 0.23 for value in pairwise_dockq_values) / len(pairwise_dockq_values)
            if pairwise_dockq_values
            else math.nan
        ),
    }


def _aggregate_group_mapping(group_results: Iterable[SideMatchResult]) -> dict[str, object]:
    """Aggregate mapping diagnostics across manifest-defined multimer groups."""

    results = list(group_results)
    per_group_fractions: list[float] = []
    per_group_sequence_identities: list[float] = []
    per_group_length_differences: list[int] = []
    summaries: list[str] = []

    for index, group_result in enumerate(results, start=1):
        matched_count = len(group_result.matched_pairs)
        matched_fraction = matched_count / max(1, group_result.total_gt_residues)
        per_group_fractions.append(matched_fraction)

        chain_sequence_identities = [
            diagnostic.sequence_identity for diagnostic in group_result.chain_diagnostics
        ]
        if chain_sequence_identities:
            per_group_sequence_identities.append(min(chain_sequence_identities))

        chain_length_differences = [
            abs(diagnostic.length_difference) for diagnostic in group_result.chain_diagnostics
        ]
        per_group_length_differences.extend(chain_length_differences)

        summaries.append(
            (
                f"group_{index}:matched={matched_count}"
                f"/gt={group_result.total_gt_residues}"
                f"/pred={group_result.total_pred_residues}"
                f"/frac={matched_fraction:.3f}"
                f"/strategy={group_result.chain_mapping_strategy}"
                f"/positional={int(group_result.used_positional_chain_mapping)}"
            )
        )

    mapping_confidence_candidates = [
        *per_group_fractions,
        *per_group_sequence_identities,
    ]
    finite_candidates = [value for value in mapping_confidence_candidates if not math.isnan(value)]

    confidence_floor = min(finite_candidates) if finite_candidates else math.nan
    return {
        "group_num_matched_residues": sum(len(result.matched_pairs) for result in results),
        "group_num_unmatched_gt_residues": sum(
            max(0, result.total_gt_residues - len(result.matched_pairs)) for result in results
        ),
        "group_num_unmatched_pred_residues": sum(
            max(0, result.total_pred_residues - len(result.matched_pairs)) for result in results
        ),
        "group_min_matched_residue_fraction": min(per_group_fractions) if per_group_fractions else math.nan,
        "group_min_chain_sequence_identity": (
            min(per_group_sequence_identities) if per_group_sequence_identities else math.nan
        ),
        "group_max_chain_length_difference": (
            max(per_group_length_differences) if per_group_length_differences else math.nan
        ),
        "group_used_sequence_fallback_any": any(result.used_sequence_fallback for result in results),
        "group_used_positional_chain_mapping_any": any(
            result.used_positional_chain_mapping for result in results
        ),
        "group_mapping_summary": ";".join(summaries),
        "mapping_confidence_score": confidence_floor,
        "mapping_confidence_floor_signal": confidence_floor,
    }


def _flatten_groups(chain_groups: Iterable[list[str]]) -> list[str]:
    """Flatten ordered chain groups into one ordered chain list."""

    flattened: list[str] = []
    for group in chain_groups:
        flattened.extend(group)
    return flattened


def _finite_values(values: Iterable[object]) -> list[float]:
    """Return finite numeric values."""

    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(numeric):
            finite.append(numeric)
    return finite


def _mean_or_nan(values: list[float]) -> float:
    """Return the mean of finite values or NaN."""

    if not values:
        return math.nan
    return sum(values) / len(values)


def _json_safe_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Convert NaN-containing metric rows to JSON-safe values."""

    safe_rows: list[dict[str, object]] = []
    for row in rows:
        safe_row: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                safe_row[key] = None
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)
    return safe_rows
