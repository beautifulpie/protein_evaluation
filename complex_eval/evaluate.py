"""Manifest-row evaluation orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .io_utils import load_structure, parse_chain_list, resolve_input_path
from .metrics import evaluate_binary_complex

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "target_id",
    "rank",
    "pred_path",
    "gt_path",
    "pred_receptor_chains",
    "pred_ligand_chains",
    "gt_receptor_chains",
    "gt_ligand_chains",
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


def evaluate_record(
    record: Mapping[str, object],
    config: EvaluationConfig,
    manifest_dir: str | Path,
) -> dict[str, object]:
    """Evaluate one prediction/ground-truth pair from the manifest."""

    validate_manifest_record(record)

    pred_path = resolve_input_path(str(record["pred_path"]), manifest_dir)
    gt_path = resolve_input_path(str(record["gt_path"]), manifest_dir)

    pred_receptor_chains = parse_chain_list(record["pred_receptor_chains"])
    pred_ligand_chains = parse_chain_list(record["pred_ligand_chains"])
    gt_receptor_chains = parse_chain_list(record["gt_receptor_chains"])
    gt_ligand_chains = parse_chain_list(record["gt_ligand_chains"])

    if not pred_receptor_chains or not pred_ligand_chains:
        raise ValueError("Prediction receptor and ligand chain lists must both be non-empty.")
    if not gt_receptor_chains or not gt_ligand_chains:
        raise ValueError("Ground-truth receptor and ligand chain lists must both be non-empty.")

    pred_structure = load_structure(pred_path, ignore_hydrogens=config.ignore_hydrogens)
    gt_structure = load_structure(gt_path, ignore_hydrogens=config.ignore_hydrogens)

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

    return {
        "sample_id": str(record["sample_id"]),
        "target_id": str(record["target_id"]),
        "rank": int(record["rank"]),
        "pred_path": str(record["pred_path"]),
        "gt_path": str(record["gt_path"]),
        "pred_receptor_chains": ",".join(pred_receptor_chains),
        "pred_ligand_chains": ",".join(pred_ligand_chains),
        "gt_receptor_chains": ",".join(gt_receptor_chains),
        "gt_ligand_chains": ",".join(gt_ligand_chains),
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
            "error": str(exc),
            "config": asdict(config),
        }
        return EvaluationOutcome(ok=False, failure=failure)
    return EvaluationOutcome(ok=True, metrics=metrics)
