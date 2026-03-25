"""Manifest-row evaluation orchestration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .io_utils import load_structure, parse_chain_groups, parse_chain_list, resolve_input_path
from .metrics import evaluate_binary_complex
from .multimer import evaluate_multimer_complex

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "target_id",
    "rank",
    "pred_path",
    "gt_path",
)
LEGACY_BINARY_MANIFEST_COLUMNS: tuple[str, ...] = (
    "pred_receptor_chains",
    "pred_ligand_chains",
    "gt_receptor_chains",
    "gt_ligand_chains",
)
MULTIMER_GROUP_MANIFEST_COLUMNS: tuple[str, ...] = (
    "pred_chain_groups",
    "gt_chain_groups",
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for complex evaluation."""

    contact_cutoff: float = 5.0
    interface_cutoff: float = 10.0
    include_all_atom_rmsd: bool = True
    include_lddt: bool = True
    include_clashes: bool = True
    ignore_hydrogens: bool = True
    sequence_fallback: bool = False
    strict_mapping: bool = True
    min_matched_residue_fraction: float = 0.7
    min_sequence_identity: float = 0.5
    max_chain_length_difference: int = 10


@dataclass(frozen=True)
class EvaluationOutcome:
    """Success or failure result for one manifest row."""

    ok: bool
    metrics: dict[str, object] | None = None
    failure: dict[str, object] | None = None


def validate_manifest_record(record: Mapping[str, object]) -> None:
    """Validate a manifest row before evaluation."""

    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Manifest row is missing required columns: {', '.join(missing)}")
    blank_columns = [column for column in REQUIRED_MANIFEST_COLUMNS if _is_missing_value(record.get(column))]
    if blank_columns:
        raise ValueError(f"Manifest row contains blank required values: {', '.join(blank_columns)}")
    if _record_uses_group_manifest(record):
        group_blank_columns = [
            column for column in MULTIMER_GROUP_MANIFEST_COLUMNS if _is_missing_value(record.get(column))
        ]
        if group_blank_columns:
            raise ValueError(
                "Manifest row uses multimer chain groups but contains blank group values: "
                f"{', '.join(group_blank_columns)}"
            )
        return

    missing_binary = [column for column in LEGACY_BINARY_MANIFEST_COLUMNS if column not in record]
    if missing_binary:
        raise ValueError(
            "Manifest must contain either multimer group columns "
            f"({', '.join(MULTIMER_GROUP_MANIFEST_COLUMNS)}) or legacy binary columns "
            f"({', '.join(LEGACY_BINARY_MANIFEST_COLUMNS)}). Missing: {', '.join(missing_binary)}"
        )
    blank_binary = [column for column in LEGACY_BINARY_MANIFEST_COLUMNS if _is_missing_value(record.get(column))]
    if blank_binary:
        raise ValueError(f"Manifest row contains blank required values: {', '.join(blank_binary)}")


def validate_manifest_columns(columns: Iterable[str]) -> None:
    """Validate manifest columns at the DataFrame level."""

    column_set = set(columns)
    missing_base = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in column_set]
    if missing_base:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing_base)}")

    has_legacy_binary = all(column in column_set for column in LEGACY_BINARY_MANIFEST_COLUMNS)
    has_group_columns = all(column in column_set for column in MULTIMER_GROUP_MANIFEST_COLUMNS)
    if not has_legacy_binary and not has_group_columns:
        raise ValueError(
            "Manifest must include either legacy binary chain columns "
            f"({', '.join(LEGACY_BINARY_MANIFEST_COLUMNS)}) or multimer group columns "
            f"({', '.join(MULTIMER_GROUP_MANIFEST_COLUMNS)})."
        )


def evaluate_record(
    record: Mapping[str, object],
    config: EvaluationConfig,
    manifest_dir: str | Path,
) -> dict[str, object]:
    """Evaluate one prediction/ground-truth pair from the manifest."""

    validate_manifest_record(record)

    pred_path = resolve_input_path(str(record["pred_path"]), manifest_dir)
    gt_path = resolve_input_path(str(record["gt_path"]), manifest_dir)

    try:
        rank = int(record["rank"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest rank must be an integer; received {record['rank']!r}.") from exc

    pred_structure = load_structure(pred_path, ignore_hydrogens=config.ignore_hydrogens)
    gt_structure = load_structure(gt_path, ignore_hydrogens=config.ignore_hydrogens)
    evaluation_mode, chain_context, metrics = _evaluate_record_metrics(
        record=record,
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        config=config,
    )
    mapping_reasons = _mapping_low_confidence_reasons(metrics, config)
    status = "low_confidence_mapping" if mapping_reasons else "success"
    error_type = "LowConfidenceMapping" if mapping_reasons else ""
    error_message = "; ".join(mapping_reasons)

    return {
        "sample_id": str(record["sample_id"]),
        "target_id": str(record["target_id"]),
        "rank": rank,
        "pred_path": str(record["pred_path"]),
        "gt_path": str(record["gt_path"]),
        "evaluation_mode": evaluation_mode,
        **chain_context,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "mapping_low_confidence_reasons": error_message,
        **metrics,
    }


def safe_evaluate_record(
    record: Mapping[str, object],
    config: EvaluationConfig,
    manifest_dir: str | Path,
) -> EvaluationOutcome:
    """Evaluate a manifest row and capture failures as structured output."""

    try:
        metrics = evaluate_record(record=record, config=config, manifest_dir=manifest_dir)
    except Exception as exc:
        failure = {
            "sample_id": str(record.get("sample_id", "")),
            "target_id": str(record.get("target_id", "")),
            "rank": record.get("rank", ""),
            "pred_path": str(record.get("pred_path", "")),
            "gt_path": str(record.get("gt_path", "")),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "mapping_low_confidence_reasons": "",
            "config": asdict(config),
        }
        return EvaluationOutcome(ok=False, failure=failure)
    return EvaluationOutcome(ok=True, metrics=metrics)


def _is_missing_value(value: object) -> bool:
    """Return whether a manifest field should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _mapping_low_confidence_reasons(metrics: Mapping[str, object], config: EvaluationConfig) -> list[str]:
    """Return explicit reasons for low-confidence residue or chain mapping."""

    evaluation_mode = str(metrics.get("evaluation_mode", "") or "")
    reasons: list[str] = []
    if evaluation_mode == "multimer":
        matched_fraction = float(metrics.get("group_min_matched_residue_fraction", math.nan))
        if not math.isnan(matched_fraction) and matched_fraction < config.min_matched_residue_fraction:
            reasons.append(
                f"group_matched_fraction_below_threshold({matched_fraction:.3f}<"
                f"{config.min_matched_residue_fraction:.3f})"
            )

        sequence_identity = float(metrics.get("group_min_chain_sequence_identity", math.nan))
        if not math.isnan(sequence_identity) and sequence_identity < config.min_sequence_identity:
            reasons.append(
                f"group_sequence_identity_below_threshold({sequence_identity:.3f}<"
                f"{config.min_sequence_identity:.3f})"
            )

        length_difference = float(metrics.get("group_max_chain_length_difference", math.nan))
        if not math.isnan(length_difference) and abs(length_difference) > config.max_chain_length_difference:
            reasons.append(
                f"group_chain_length_difference_above_threshold({int(length_difference)}>"
                f"{config.max_chain_length_difference})"
            )

        if bool(metrics.get("group_used_positional_chain_mapping_any", False)):
            reasons.append("group_positional_chain_mapping_present")
        return reasons

    for side_name in ("receptor", "ligand"):
        matched_fraction = float(metrics.get(f"{side_name}_matched_residue_fraction", math.nan))
        if not math.isnan(matched_fraction) and matched_fraction < config.min_matched_residue_fraction:
            reasons.append(
                f"{side_name}_matched_fraction_below_threshold({matched_fraction:.3f}<"
                f"{config.min_matched_residue_fraction:.3f})"
            )

        sequence_identity = float(metrics.get(f"{side_name}_min_chain_sequence_identity", math.nan))
        if not math.isnan(sequence_identity) and sequence_identity < config.min_sequence_identity:
            reasons.append(
                f"{side_name}_sequence_identity_below_threshold({sequence_identity:.3f}<"
                f"{config.min_sequence_identity:.3f})"
            )

        length_difference = float(metrics.get(f"{side_name}_max_chain_length_difference", math.nan))
        if not math.isnan(length_difference) and abs(length_difference) > config.max_chain_length_difference:
            reasons.append(
                f"{side_name}_chain_length_difference_above_threshold({int(length_difference)}>"
                f"{config.max_chain_length_difference})"
            )

        if (
            bool(metrics.get(f"{side_name}_used_positional_chain_mapping", False))
            and int(metrics.get(f"{side_name}_num_chain_pairs", 0)) > 1
        ):
            reasons.append(f"{side_name}_multi_chain_positional_mapping")

    return reasons


def _record_uses_group_manifest(record: Mapping[str, object]) -> bool:
    """Return whether a manifest row uses explicit multimer chain groups."""

    if any(column not in record for column in MULTIMER_GROUP_MANIFEST_COLUMNS):
        return False
    return any(not _is_missing_value(record.get(column)) for column in MULTIMER_GROUP_MANIFEST_COLUMNS)


def _evaluate_record_metrics(
    record: Mapping[str, object],
    pred_structure,
    gt_structure,
    config: EvaluationConfig,
) -> tuple[str, dict[str, object], dict[str, object]]:
    """Evaluate one record in binary or multimer mode."""

    if _record_uses_group_manifest(record):
        pred_chain_groups = parse_chain_groups(record.get("pred_chain_groups"))
        gt_chain_groups = parse_chain_groups(record.get("gt_chain_groups"))
        if len(pred_chain_groups) != len(gt_chain_groups):
            raise ValueError(
                "Prediction and ground-truth multimer group counts must match; "
                f"received pred={len(pred_chain_groups)} gt={len(gt_chain_groups)}."
            )
        if len(pred_chain_groups) < 2:
            raise ValueError("Multimer evaluation requires at least two chain groups.")
        if any(not group for group in pred_chain_groups) or any(not group for group in gt_chain_groups):
            raise ValueError("Manifest chain groups must not contain empty groups.")

        chain_context = {
            "pred_chain_groups": _serialize_chain_groups(pred_chain_groups),
            "gt_chain_groups": _serialize_chain_groups(gt_chain_groups),
            "pred_receptor_chains": (
                ",".join(pred_chain_groups[0]) if len(pred_chain_groups) == 2 else ""
            ),
            "pred_ligand_chains": (
                ",".join(pred_chain_groups[1]) if len(pred_chain_groups) == 2 else ""
            ),
            "gt_receptor_chains": ",".join(gt_chain_groups[0]) if len(gt_chain_groups) == 2 else "",
            "gt_ligand_chains": ",".join(gt_chain_groups[1]) if len(gt_chain_groups) == 2 else "",
        }

        if len(pred_chain_groups) == 2:
            metrics = evaluate_binary_complex(
                pred_structure=pred_structure,
                gt_structure=gt_structure,
                pred_receptor_chains=pred_chain_groups[0],
                pred_ligand_chains=pred_chain_groups[1],
                gt_receptor_chains=gt_chain_groups[0],
                gt_ligand_chains=gt_chain_groups[1],
                contact_cutoff=config.contact_cutoff,
                interface_cutoff=config.interface_cutoff,
                include_all_atom_rmsd=config.include_all_atom_rmsd,
                include_lddt=config.include_lddt,
                include_clashes=config.include_clashes,
                sequence_fallback=config.sequence_fallback,
            )
            metrics["evaluation_mode"] = "binary"
            metrics["num_chain_groups"] = 2
            return "binary", chain_context, metrics

        metrics = evaluate_multimer_complex(
            pred_structure=pred_structure,
            gt_structure=gt_structure,
            pred_chain_groups=pred_chain_groups,
            gt_chain_groups=gt_chain_groups,
            contact_cutoff=config.contact_cutoff,
            interface_cutoff=config.interface_cutoff,
            include_all_atom_rmsd=config.include_all_atom_rmsd,
            include_lddt=config.include_lddt,
            include_clashes=config.include_clashes,
            sequence_fallback=config.sequence_fallback,
        )
        return "multimer", chain_context, metrics

    pred_receptor_chains = parse_chain_list(record["pred_receptor_chains"])
    pred_ligand_chains = parse_chain_list(record["pred_ligand_chains"])
    gt_receptor_chains = parse_chain_list(record["gt_receptor_chains"])
    gt_ligand_chains = parse_chain_list(record["gt_ligand_chains"])

    if not pred_receptor_chains or not pred_ligand_chains:
        raise ValueError("Prediction receptor and ligand chain lists must both be non-empty.")
    if not gt_receptor_chains or not gt_ligand_chains:
        raise ValueError("Ground-truth receptor and ligand chain lists must both be non-empty.")

    metrics = evaluate_binary_complex(
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        pred_receptor_chains=pred_receptor_chains,
        pred_ligand_chains=pred_ligand_chains,
        gt_receptor_chains=gt_receptor_chains,
        gt_ligand_chains=gt_ligand_chains,
        contact_cutoff=config.contact_cutoff,
        interface_cutoff=config.interface_cutoff,
        include_all_atom_rmsd=config.include_all_atom_rmsd,
        include_lddt=config.include_lddt,
        include_clashes=config.include_clashes,
        sequence_fallback=config.sequence_fallback,
    )
    metrics["evaluation_mode"] = "binary"
    metrics["num_chain_groups"] = 2
    return (
        "binary",
        {
            "pred_receptor_chains": ",".join(pred_receptor_chains),
            "pred_ligand_chains": ",".join(pred_ligand_chains),
            "gt_receptor_chains": ",".join(gt_receptor_chains),
            "gt_ligand_chains": ",".join(gt_ligand_chains),
            "pred_chain_groups": "",
            "gt_chain_groups": "",
        },
        metrics,
    )


def _serialize_chain_groups(chain_groups: Iterable[list[str]]) -> str:
    """Serialize pipe-separated chain groups."""

    return "|".join(",".join(group) for group in chain_groups)
