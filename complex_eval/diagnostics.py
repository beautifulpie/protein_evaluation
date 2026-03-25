"""Explainability, confidence, and diagnostic reporting helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

EXPLAINABILITY_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "mapping_confidence_score",
        "mapping_confidence_label",
        "mapping_confidence_floor_signal",
        "diagnostic_tags",
        "dockq_decomposition_fnat_term",
        "dockq_decomposition_irmsd_term",
        "dockq_decomposition_lrmsd_term",
        "interface_native_contact_count",
        "interface_pred_contact_count",
        "interface_recovered_contact_count",
        "interface_missing_contact_count",
        "interface_false_contact_count",
        "interface_precision",
        "interface_recall",
        "interface_f1",
        "failure_category",
    }
)


@dataclass(frozen=True)
class ExplainabilityRecord:
    """Flat explainability layer added on top of core metrics."""

    mapping_confidence_score: float
    mapping_confidence_label: str
    diagnostic_tags: list[str]
    dockq_decomposition_fnat_term: float
    dockq_decomposition_irmsd_term: float
    dockq_decomposition_lrmsd_term: float
    interface_native_contact_count: int
    interface_pred_contact_count: int
    interface_recovered_contact_count: int
    interface_missing_contact_count: int
    interface_false_contact_count: int
    interface_precision: float
    interface_recall: float
    interface_f1: float

    def to_row_fields(self) -> dict[str, object]:
        """Return CSV-friendly flat fields."""

        fields = asdict(self)
        fields["diagnostic_tags"] = ";".join(self.diagnostic_tags)
        return fields


def build_explainability_record(
    metrics: Mapping[str, object],
    mode: str = "heuristic",
) -> ExplainabilityRecord:
    """Build an explainability layer from flat metric outputs."""

    score = calculate_mapping_confidence_score(metrics, mode=mode)
    label = mapping_confidence_label(score)
    tags = diagnostic_tags(metrics)

    return ExplainabilityRecord(
        mapping_confidence_score=score,
        mapping_confidence_label=label,
        diagnostic_tags=tags,
        dockq_decomposition_fnat_term=_safe_float(metrics.get("dockq_decomposition_fnat_term")),
        dockq_decomposition_irmsd_term=_safe_float(metrics.get("dockq_decomposition_irmsd_term")),
        dockq_decomposition_lrmsd_term=_safe_float(metrics.get("dockq_decomposition_lrmsd_term")),
        interface_native_contact_count=int(metrics.get("interface_native_contact_count", 0) or 0),
        interface_pred_contact_count=int(metrics.get("interface_pred_contact_count", 0) or 0),
        interface_recovered_contact_count=int(metrics.get("interface_recovered_contact_count", 0) or 0),
        interface_missing_contact_count=int(metrics.get("interface_missing_contact_count", 0) or 0),
        interface_false_contact_count=int(metrics.get("interface_false_contact_count", 0) or 0),
        interface_precision=_safe_float(metrics.get("interface_precision")),
        interface_recall=_safe_float(metrics.get("interface_recall")),
        interface_f1=_safe_float(metrics.get("interface_f1")),
    )


def calculate_mapping_confidence_score(
    metrics: Mapping[str, object],
    mode: str = "heuristic",
) -> float:
    """Return a deterministic mapping confidence score in [0, 1].

    Heuristic design:

    1. Build a base score from directly interpretable coverage signals:
       - residue match coverage
       - sequence identity
       - atom match coverage
       - chain length agreement
    2. Apply transparent penalties for recovery mechanisms and warning signals:
       - sequence fallback
       - ambiguous positional mapping
       - sparse atom coverage
       - parse warnings
       - CA fallback for interface metrics

    This intentionally favors visibility and interpretability over statistical
    sophistication. A score near 1.0 means "clean and well-supported mapping",
    not "guaranteed correctness".
    """

    if mode != "heuristic":
        raise ValueError(f"Unsupported mapping confidence mode: {mode}")

    matched_fraction = _primary_matched_fraction(metrics)
    sequence_identity = _primary_sequence_identity(metrics)
    atom_fraction = _safe_float(metrics.get("matched_atom_fraction"))
    atom_fraction = 0.0 if math.isnan(atom_fraction) else atom_fraction
    chain_length_difference = _primary_chain_length_difference(metrics)
    chain_length_score = max(0.0, 1.0 - min(abs(chain_length_difference), 20.0) / 20.0)

    base_score = (
        0.35 * matched_fraction
        + 0.25 * sequence_identity
        + 0.25 * atom_fraction
        + 0.15 * chain_length_score
    )

    penalty = 1.0
    if bool(metrics.get("used_sequence_fallback", False)):
        penalty *= 0.92
    if _uses_ambiguous_positional_mapping(metrics):
        penalty *= 0.75
    if atom_fraction < 0.50:
        penalty *= 0.75
    if atom_fraction < 0.25:
        penalty *= 0.80
    if _has_non_fallback_parse_warning(metrics):
        penalty *= 0.90
    if bool(metrics.get("used_ca_fallback_for_irmsd", False)):
        penalty *= 0.95
    if bool(metrics.get("used_ca_fallback_for_lrmsd", False)):
        penalty *= 0.97

    return max(0.0, min(1.0, base_score * penalty))


def mapping_confidence_label(score: float) -> str:
    """Convert a confidence score into a coarse label."""

    if math.isnan(score):
        return "low"
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def diagnostic_tags(metrics: Mapping[str, object]) -> list[str]:
    """Assign deterministic diagnostic tags for explainability."""

    tags: list[str] = []
    matched_fraction = _primary_matched_fraction(metrics)
    sequence_identity = _primary_sequence_identity(metrics)
    chain_length_difference = abs(_primary_chain_length_difference(metrics))
    atom_fraction = _safe_float(metrics.get("matched_atom_fraction"))
    native_contacts = int(metrics.get("interface_native_contact_count", 0) or 0)
    recovered_contacts = int(metrics.get("interface_recovered_contact_count", 0) or 0)
    false_contacts = int(metrics.get("interface_false_contact_count", 0) or 0)

    if bool(metrics.get("used_sequence_fallback", False)):
        tags.append("used_sequence_fallback")
    if matched_fraction < 0.80:
        tags.append("low_matched_residue_fraction")
    if sequence_identity < 0.80:
        tags.append("low_sequence_identity")
    if chain_length_difference > 5:
        tags.append("chain_length_mismatch")
    if _uses_ambiguous_positional_mapping(metrics):
        tags.append("ambiguous_chain_mapping")
    if not math.isnan(atom_fraction) and atom_fraction < 0.70:
        tags.append("sparse_atom_coverage")
    if bool(metrics.get("used_ca_fallback_for_irmsd", False)):
        tags.append("used_ca_fallback_irmsd")
    if bool(metrics.get("used_ca_fallback_for_lrmsd", False)):
        tags.append("used_ca_fallback_lrmsd")
    if _has_non_fallback_parse_warning(metrics):
        tags.append("parse_warning_present")
    if native_contacts > 0 and 0 < recovered_contacts < native_contacts:
        tags.append("partial_interface_observed")
    if false_contacts > 0:
        tags.append("false_interface_contacts_present")
    if _likely_chain_swap(metrics):
        tags.append("likely_chain_swap")
    if _likely_numbering_offset(metrics):
        tags.append("likely_numbering_offset")

    if not tags:
        tags.append("clean_mapping")
    return tags


def classify_failure(exc: Exception) -> str:
    """Map exceptions onto normalized failure categories."""

    message = str(exc).lower()
    error_type = type(exc).__name__.lower()

    if "parse" in message or "no protein residues" in message or "does not contain any models" in message:
        return "parse_error"
    if (
        "missing required" in message
        or "blank required" in message
        or "rank must be an integer" in message
        or "must include either" in message
        or "must contain either" in message
    ):
        return "invalid_input"
    if (
        "chain list is empty" in message
        or "chain groups" in message
        or "receptor and ligand chain lists" in message
        or "ground-truth chain" in message
        or "prediction chain" in message
        or "multimer evaluation requires" in message
    ):
        return "mapping_error"
    if "rmsd" in message or "dockq" in message or "must have shape" in message or "must contain at least one point" in message:
        return "metric_computation_error"
    if "valueerror" in error_type:
        return "invalid_input"
    return "unknown_error"


def strip_explainability_fields(row: Mapping[str, object]) -> dict[str, object]:
    """Return a copy without explainability-specific flat fields."""

    return {key: value for key, value in row.items() if key not in EXPLAINABILITY_FIELD_NAMES}


def write_per_sample_diagnostics_jsonl(
    rows: Iterable[Mapping[str, object]],
    path: str | Path,
    include_explainability_fields: bool = True,
) -> None:
    """Write a structured per-sample diagnostics JSONL report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = build_diagnostics_json_record(row, include_explainability_fields=include_explainability_fields)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def build_diagnostics_json_record(
    row: Mapping[str, object],
    include_explainability_fields: bool = True,
) -> dict[str, object]:
    """Build a nested JSON diagnostics record from a flat CSV row."""

    evaluation_mode = str(row.get("evaluation_mode", "binary"))
    interface_json = row.get("pairwise_interface_metrics_json", "")
    parsed_pairwise_metrics = None
    if evaluation_mode == "multimer" and str(interface_json).strip():
        try:
            parsed_pairwise_metrics = json.loads(str(interface_json))
        except json.JSONDecodeError:
            parsed_pairwise_metrics = None

    record = {
        "sample_id": row.get("sample_id", ""),
        "target_id": row.get("target_id", ""),
        "rank": row.get("rank", ""),
        "method": row.get("method", ""),
        "evaluation_mode": evaluation_mode,
        "status": row.get("status", ""),
        "failure_category": row.get("failure_category", ""),
        "error_type": row.get("error_type", ""),
        "error_message": row.get("error_message", ""),
        "paths": {
            "pred_path": row.get("pred_path", ""),
            "gt_path": row.get("gt_path", ""),
        },
        "chain_context": {
            "pred_receptor_chains": row.get("pred_receptor_chains", ""),
            "pred_ligand_chains": row.get("pred_ligand_chains", ""),
            "gt_receptor_chains": row.get("gt_receptor_chains", ""),
            "gt_ligand_chains": row.get("gt_ligand_chains", ""),
            "pred_chain_groups": row.get("pred_chain_groups", ""),
            "gt_chain_groups": row.get("gt_chain_groups", ""),
        },
        "core_metrics": _json_safe_mapping(
            {
                "ca_rmsd": row.get("ca_rmsd"),
                "all_atom_rmsd": row.get("all_atom_rmsd"),
                "irmsd": row.get("irmsd"),
                "lrmsd": row.get("lrmsd"),
                "fnat": row.get("fnat"),
                "dockq": row.get("dockq"),
                "lddt_ca": row.get("lddt_ca"),
                "clash_count": row.get("clash_count"),
                "clashes_per_1000_atoms": row.get("clashes_per_1000_atoms"),
                "matched_residue_fraction": row.get("matched_residue_fraction"),
                "matched_atom_fraction": row.get("matched_atom_fraction"),
                "num_matched_residues": row.get("num_matched_residues"),
                "num_matched_atoms": row.get("num_matched_atoms"),
            }
        ),
        "mapping_diagnostics": _json_safe_mapping(
            {
                "mapping_confidence_score": row.get("mapping_confidence_score"),
                "mapping_confidence_label": row.get("mapping_confidence_label", ""),
                "mapping_confidence_floor_signal": row.get("mapping_confidence_floor_signal"),
                "receptor_matched_residue_fraction": row.get("receptor_matched_residue_fraction"),
                "ligand_matched_residue_fraction": row.get("ligand_matched_residue_fraction"),
                "group_min_matched_residue_fraction": row.get("group_min_matched_residue_fraction"),
                "receptor_min_chain_sequence_identity": row.get("receptor_min_chain_sequence_identity"),
                "ligand_min_chain_sequence_identity": row.get("ligand_min_chain_sequence_identity"),
                "group_min_chain_sequence_identity": row.get("group_min_chain_sequence_identity"),
                "receptor_max_chain_length_difference": row.get("receptor_max_chain_length_difference"),
                "ligand_max_chain_length_difference": row.get("ligand_max_chain_length_difference"),
                "group_max_chain_length_difference": row.get("group_max_chain_length_difference"),
                "used_sequence_fallback": row.get("used_sequence_fallback"),
                "used_ca_fallback_for_irmsd": row.get("used_ca_fallback_for_irmsd"),
                "used_ca_fallback_for_lrmsd": row.get("used_ca_fallback_for_lrmsd"),
                "receptor_chain_mapping_strategy": row.get("receptor_chain_mapping_strategy", ""),
                "ligand_chain_mapping_strategy": row.get("ligand_chain_mapping_strategy", ""),
                "group_mapping_summary": row.get("group_mapping_summary", ""),
            }
        ),
        "interface_diagnostics": _json_safe_mapping(
            {
                "interface_native_contact_count": row.get("interface_native_contact_count"),
                "interface_pred_contact_count": row.get("interface_pred_contact_count"),
                "interface_recovered_contact_count": row.get("interface_recovered_contact_count"),
                "interface_missing_contact_count": row.get("interface_missing_contact_count"),
                "interface_false_contact_count": row.get("interface_false_contact_count"),
                "interface_precision": row.get("interface_precision"),
                "interface_recall": row.get("interface_recall"),
                "interface_f1": row.get("interface_f1"),
                "pairwise_interface_metrics": parsed_pairwise_metrics,
            }
        ),
        "warnings": {
            "parse_warning": row.get("parse_warning", ""),
            "mapping_low_confidence_reasons": _split_semicolon_field(row.get("mapping_low_confidence_reasons", "")),
        },
    }
    if include_explainability_fields:
        record["explainability"] = _json_safe_mapping(
            {
                "mapping_confidence_score": row.get("mapping_confidence_score"),
                "mapping_confidence_label": row.get("mapping_confidence_label", ""),
                "diagnostic_tags": _split_semicolon_field(row.get("diagnostic_tags", "")),
                "dockq_decomposition_fnat_term": row.get("dockq_decomposition_fnat_term"),
                "dockq_decomposition_irmsd_term": row.get("dockq_decomposition_irmsd_term"),
                "dockq_decomposition_lrmsd_term": row.get("dockq_decomposition_lrmsd_term"),
            }
        )
    return record


def _primary_matched_fraction(metrics: Mapping[str, object]) -> float:
    """Return the main matched-residue coverage signal."""

    if str(metrics.get("evaluation_mode", "binary")) == "multimer":
        return _nan_to_zero(metrics.get("group_min_matched_residue_fraction"))
    receptor = _nan_to_zero(metrics.get("receptor_matched_residue_fraction"))
    ligand = _nan_to_zero(metrics.get("ligand_matched_residue_fraction"))
    return min(receptor, ligand)


def _primary_sequence_identity(metrics: Mapping[str, object]) -> float:
    """Return the main sequence-identity signal."""

    if str(metrics.get("evaluation_mode", "binary")) == "multimer":
        return _nan_to_zero(metrics.get("group_min_chain_sequence_identity"), default=1.0)
    receptor = _nan_to_zero(metrics.get("receptor_min_chain_sequence_identity"), default=1.0)
    ligand = _nan_to_zero(metrics.get("ligand_min_chain_sequence_identity"), default=1.0)
    return min(receptor, ligand)


def _primary_chain_length_difference(metrics: Mapping[str, object]) -> float:
    """Return the main chain-length mismatch signal."""

    if str(metrics.get("evaluation_mode", "binary")) == "multimer":
        return _safe_float(metrics.get("group_max_chain_length_difference"))
    receptor = _safe_float(metrics.get("receptor_max_chain_length_difference"))
    ligand = _safe_float(metrics.get("ligand_max_chain_length_difference"))
    finite = [abs(value) for value in (receptor, ligand) if not math.isnan(value)]
    return max(finite) if finite else 0.0


def _uses_ambiguous_positional_mapping(metrics: Mapping[str, object]) -> bool:
    """Return whether positional chain mapping suggests ambiguity."""

    if str(metrics.get("evaluation_mode", "binary")) == "multimer":
        return bool(metrics.get("group_used_positional_chain_mapping_any", False))
    receptor_ambiguous = bool(metrics.get("receptor_used_positional_chain_mapping", False)) and int(
        metrics.get("receptor_num_chain_pairs", 0) or 0
    ) > 1
    ligand_ambiguous = bool(metrics.get("ligand_used_positional_chain_mapping", False)) and int(
        metrics.get("ligand_num_chain_pairs", 0) or 0
    ) > 1
    return receptor_ambiguous or ligand_ambiguous


def _likely_chain_swap(metrics: Mapping[str, object]) -> bool:
    """Return whether the pattern looks like a likely chain swap."""

    matched_fraction = _primary_matched_fraction(metrics)
    sequence_identity = _primary_sequence_identity(metrics)
    chain_length_difference = abs(_primary_chain_length_difference(metrics))
    return _uses_ambiguous_positional_mapping(metrics) and matched_fraction < 0.80 and sequence_identity < 0.95 and chain_length_difference <= 1.0


def _likely_numbering_offset(metrics: Mapping[str, object]) -> bool:
    """Return whether sequence fallback likely rescued a numbering offset."""

    matched_fraction = _primary_matched_fraction(metrics)
    sequence_identity = _primary_sequence_identity(metrics)
    chain_length_difference = abs(_primary_chain_length_difference(metrics))
    return (
        bool(metrics.get("used_sequence_fallback", False))
        and matched_fraction >= 0.90
        and sequence_identity >= 0.90
        and chain_length_difference <= 1.0
    )


def _has_non_fallback_parse_warning(metrics: Mapping[str, object]) -> bool:
    """Return whether parse_warning contains more than expected fallback notices."""

    warnings = [
        item.strip()
        for item in str(metrics.get("parse_warning", "") or "").split("|")
        if item.strip()
    ]
    if not warnings:
        return False
    return any("Used sequence-based residue matching" not in warning for warning in warnings)


def _split_semicolon_field(value: object) -> list[str]:
    """Split a semicolon-delimited field into a stable list."""

    text = str(value or "").strip()
    if not text:
        return []
    return [item for item in text.split(";") if item]


def _json_safe_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Convert mappings into JSON-safe scalars recursively."""

    safe: dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            safe[key] = _json_safe_mapping(value)
        elif isinstance(value, list):
            safe[key] = [_json_safe_value(item) for item in value]
        else:
            safe[key] = _json_safe_value(value)
    return safe


def _json_safe_value(value: object) -> object:
    """Convert scalar values into JSON-safe values."""

    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _safe_float(value: object) -> float:
    """Convert objects to float or NaN."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric


def _nan_to_zero(value: object, default: float = 0.0) -> float:
    """Return float(value), replacing NaN with a deterministic default."""

    numeric = _safe_float(value)
    return default if math.isnan(numeric) else numeric
