"""Validation helpers against trusted external tools when available."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import pearsonr, spearmanr

DEFAULT_DOCKQ_THRESHOLDS: dict[str, float] = {
    "dockq_mean_abs_diff_max": 0.02,
    "fnat_mean_abs_diff_max": 0.05,
    "irmsd_mean_abs_diff_max": 0.5,
    "lrmsd_mean_abs_diff_max": 0.5,
    "dockq_pearson_min": 0.95,
    "dockq_spearman_min": 0.95,
}
DOCKQ_REFERENCE_FIELDS: tuple[str, ...] = ("dockq", "fnat", "irmsd", "lrmsd")


def write_validation_outputs(
    metrics_df: pd.DataFrame,
    out_dir: str | Path,
    mode: str,
    dockq_executable: str | None = None,
    include_invalid_rows: bool = False,
) -> dict[str, object]:
    """Run a validation mode and write stable per-sample and summary outputs."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if mode == "none":
        summary = {"status": "disabled", "mode": "none"}
        (out_path / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if mode != "dockq":
        raise ValueError(f"Unsupported validation mode: {mode}")

    comparisons, summary = validate_against_dockq(
        metrics_df=metrics_df,
        dockq_executable=dockq_executable,
        include_invalid_rows=include_invalid_rows,
    )
    comparisons.to_csv(out_path / "validation_dockq_per_sample.csv", index=False)
    (out_path / "validation_dockq_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_against_dockq(
    metrics_df: pd.DataFrame,
    dockq_executable: str | None = None,
    include_invalid_rows: bool = False,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare internal metrics to the official DockQ executable when available."""

    threshold_values = dict(DEFAULT_DOCKQ_THRESHOLDS)
    if thresholds is not None:
        threshold_values.update(thresholds)

    executable = resolve_dockq_executable(dockq_executable)
    if executable is None:
        summary = {
            "status": "unavailable",
            "mode": "dockq",
            "message": "DockQ executable was not found on PATH. Install DockQ or pass --dockq_executable.",
            "thresholds": threshold_values,
        }
        return pd.DataFrame(columns=["sample_id", "target_id", "status", "validation_message"]), summary

    rows: list[dict[str, object]] = []
    for record in metrics_df.sort_values(["target_id", "rank", "sample_id"]).to_dict(orient="records"):
        record_status = str(record.get("status", "success"))
        if not include_invalid_rows and record_status != "success":
            rows.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "target_id": record.get("target_id", ""),
                    "status": "skipped",
                    "validation_message": f"Skipped because evaluation status is {record_status}.",
                }
            )
            continue

        applicability_reason = _dockq_applicability_reason(record)
        if applicability_reason:
            rows.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "target_id": record.get("target_id", ""),
                    "status": "skipped",
                    "validation_message": applicability_reason,
                }
            )
            continue

        try:
            reference = run_dockq(
                pred_path=str(record["pred_path"]),
                gt_path=str(record["gt_path"]),
                executable=executable,
            )
            row: dict[str, object] = {
                "sample_id": record.get("sample_id", ""),
                "target_id": record.get("target_id", ""),
                "status": "validated",
                "validation_message": "",
            }
            for field in DOCKQ_REFERENCE_FIELDS:
                row[f"{field}_reference"] = reference[field]
                row[f"{field}_internal"] = _safe_float(record.get(field))
                row[f"{field}_diff"] = _diff(_safe_float(record.get(field)), reference[field])
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "target_id": record.get("target_id", ""),
                    "status": "failed",
                    "validation_message": str(exc),
                }
            )

    comparisons = pd.DataFrame(rows)
    summary = _summarize_dockq_validation(comparisons, threshold_values)
    return comparisons, summary


def resolve_dockq_executable(explicit_path: str | None = None) -> str | None:
    """Resolve a DockQ executable path."""

    candidates = [explicit_path] if explicit_path else ["DockQ", "DockQ.py"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if resolved:
            return resolved
    return None


def run_dockq(pred_path: str, gt_path: str, executable: str) -> dict[str, float]:
    """Run DockQ and parse the main reported metrics."""

    command = _dockq_command(executable, pred_path, gt_path)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"DockQ failed with exit code {completed.returncode}: {stderr}")
    parsed = parse_dockq_output(completed.stdout)
    if parsed is None:
        raise ValueError("DockQ output did not contain parseable DockQ/Fnat/iRMSD/LRMSD metrics.")
    return parsed


def parse_dockq_output(stdout: str) -> dict[str, float] | None:
    """Parse DockQ stdout into canonical metric keys."""

    patterns = {
        "dockq": [r"DockQ(?:-score)?\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "fnat": [r"Fnat\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "irmsd": [r"iRMSD?\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "lrmsd": [r"LRMSD?\s*[:=]?\s*([0-9]*\.?[0-9]+)", r"LRMS\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
    }
    parsed: dict[str, float] = {}
    for key, key_patterns in patterns.items():
        for pattern in key_patterns:
            match = re.search(pattern, stdout, flags=re.IGNORECASE)
            if match:
                parsed[key] = float(match.group(1))
                break
    return parsed if set(parsed) == set(DOCKQ_REFERENCE_FIELDS) else None


def _dockq_command(executable: str, pred_path: str, gt_path: str) -> list[str]:
    """Build a DockQ command compatible with executable scripts and Python entrypoints."""

    if executable.endswith(".py"):
        return [sys.executable, executable, pred_path, gt_path]
    return [executable, pred_path, gt_path]


def _dockq_applicability_reason(record: dict[str, Any]) -> str:
    """Return a reason why DockQ validation should be skipped for a row."""

    for key in (
        "pred_receptor_chains",
        "pred_ligand_chains",
        "gt_receptor_chains",
        "gt_ligand_chains",
    ):
        chain_count = len([item for item in str(record.get(key, "")).split(",") if item.strip()])
        if chain_count != 1:
            return "DockQ validation currently supports only one-chain-per-side examples."
    return ""


def _summarize_dockq_validation(comparisons: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, object]:
    """Summarize validation diffs and correlations."""

    validated = comparisons[comparisons["status"] == "validated"].copy()
    summary: dict[str, object] = {
        "status": "ok" if not validated.empty else "no_comparable_rows",
        "mode": "dockq",
        "num_rows": int(len(comparisons)),
        "num_validated_rows": int(len(validated)),
        "status_counts": comparisons["status"].value_counts().sort_index().to_dict() if not comparisons.empty else {},
        "thresholds": thresholds,
        "metrics": {},
    }
    if validated.empty:
        return summary

    correlations: dict[str, object] = {}
    passes: list[bool] = []
    for field in DOCKQ_REFERENCE_FIELDS:
        diff_series = pd.to_numeric(validated[f"{field}_diff"], errors="coerce").abs()
        internal_series = pd.to_numeric(validated[f"{field}_internal"], errors="coerce")
        reference_series = pd.to_numeric(validated[f"{field}_reference"], errors="coerce")
        field_summary = {
            "mean_abs_diff": _safe_float(diff_series.mean()),
            "max_abs_diff": _safe_float(diff_series.max()),
        }
        summary["metrics"][field] = field_summary

        if field in {"dockq"} and len(validated) >= 2 and internal_series.nunique(dropna=True) > 1 and reference_series.nunique(dropna=True) > 1:
            correlations["dockq_pearson"] = _safe_float(pearsonr(internal_series, reference_series).statistic)
            correlations["dockq_spearman"] = _safe_float(spearmanr(internal_series, reference_series).statistic)

    passes.append(_metric_within_threshold(summary["metrics"]["dockq"]["mean_abs_diff"], thresholds["dockq_mean_abs_diff_max"]))
    passes.append(_metric_within_threshold(summary["metrics"]["fnat"]["mean_abs_diff"], thresholds["fnat_mean_abs_diff_max"]))
    passes.append(_metric_within_threshold(summary["metrics"]["irmsd"]["mean_abs_diff"], thresholds["irmsd_mean_abs_diff_max"]))
    passes.append(_metric_within_threshold(summary["metrics"]["lrmsd"]["mean_abs_diff"], thresholds["lrmsd_mean_abs_diff_max"]))
    if "dockq_pearson" in correlations and correlations["dockq_pearson"] is not None:
        passes.append(float(correlations["dockq_pearson"]) >= thresholds["dockq_pearson_min"])
    if "dockq_spearman" in correlations and correlations["dockq_spearman"] is not None:
        passes.append(float(correlations["dockq_spearman"]) >= thresholds["dockq_spearman_min"])

    summary["correlations"] = correlations
    summary["pass"] = all(passes)
    return summary


def _safe_float(value: object) -> float | None:
    """Convert values to JSON-safe floats."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _diff(internal: float | None, reference: float) -> float | None:
    """Return internal-reference difference when both are finite."""

    if internal is None:
        return None
    return internal - reference


def _metric_within_threshold(value: float | None, threshold: float) -> bool:
    """Return whether a summary metric satisfies a threshold."""

    return value is not None and value <= threshold
