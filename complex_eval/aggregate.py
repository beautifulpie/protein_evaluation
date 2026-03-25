"""Aggregation helpers for per-sample and per-target summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "ca_rmsd",
    "all_atom_rmsd",
    "irmsd",
    "lrmsd",
    "fnat",
    "dockq",
    "pairwise_fnat_mean",
    "pairwise_irmsd_mean",
    "pairwise_lrmsd_mean",
    "pairwise_dockq_mean",
    "pairwise_dockq_min",
    "lddt_ca",
    "clash_count",
    "clashes_per_1000_atoms",
    "matched_residue_fraction",
    "matched_atom_fraction",
    "num_matched_residues",
    "num_matched_atoms",
    "num_native_contacts",
    "num_recovered_contacts",
    "mapping_confidence_score",
)
VALID_SUMMARY_STATUSES: frozenset[str] = frozenset({"success"})


def write_aggregate_outputs(
    metrics_df: pd.DataFrame,
    out_dir: str | Path,
    topk: int,
    include_invalid_rows_in_summary: bool = False,
    all_rows_df: pd.DataFrame | None = None,
    summary_by_method: bool | None = None,
    summary_by_confidence: bool = True,
) -> dict[str, object]:
    """Write per-sample outputs and aggregate summaries."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ordered_metrics = metrics_df.copy()
    all_rows = all_rows_df.copy() if all_rows_df is not None else metrics_df.copy()
    for column, dtype in (("target_id", "object"), ("rank", "float64"), ("sample_id", "object")):
        if column not in ordered_metrics.columns:
            ordered_metrics[column] = pd.Series(dtype=dtype)
        if column not in all_rows.columns:
            all_rows[column] = pd.Series(dtype=dtype)
    ordered_metrics["_rank_sort"] = pd.to_numeric(ordered_metrics["rank"], errors="coerce")
    ordered_metrics = ordered_metrics.sort_values(["target_id", "_rank_sort", "sample_id"]).reset_index(drop=True)
    ordered_metrics = ordered_metrics.drop(columns=["_rank_sort"], errors="ignore")
    all_rows["_rank_sort"] = pd.to_numeric(all_rows["rank"], errors="coerce")
    all_rows = all_rows.sort_values(["target_id", "_rank_sort", "sample_id"]).reset_index(drop=True)
    all_rows = all_rows.drop(columns=["_rank_sort"], errors="ignore")
    ordered_metrics.to_csv(out_path / "per_sample_metrics.csv", index=False)

    summary_input = _filter_summary_rows(ordered_metrics, include_invalid_rows=include_invalid_rows_in_summary)
    top1_df = select_top1(summary_input)
    best_of_k_df = select_best_of_k(summary_input, topk=topk)

    best_of_k_df.to_csv(out_path / "per_target_best_of_k.csv", index=False)

    summary_top1 = summarize_subset(top1_df, all_rows=ordered_metrics, include_invalid_rows=include_invalid_rows_in_summary)
    summary_best_of_k = summarize_subset(
        best_of_k_df,
        all_rows=ordered_metrics,
        include_invalid_rows=include_invalid_rows_in_summary,
    )

    (out_path / "summary_top1.json").write_text(json.dumps(summary_top1, indent=2), encoding="utf-8")
    (out_path / "summary_best_of_k.json").write_text(json.dumps(summary_best_of_k, indent=2), encoding="utf-8")
    summary_diagnostics = summarize_diagnostics(
        ordered_metrics=ordered_metrics,
        all_rows=all_rows,
        topk=topk,
        include_invalid_rows=include_invalid_rows_in_summary,
        summary_by_method=summary_by_method,
        summary_by_confidence=summary_by_confidence,
    )
    (out_path / "summary_diagnostics.json").write_text(
        json.dumps(summary_diagnostics, indent=2),
        encoding="utf-8",
    )
    return summary_diagnostics


def select_top1(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select the top-ranked sample per target."""

    if metrics_df.empty:
        return metrics_df.copy()
    ordered = metrics_df.copy()
    ordered["_rank_sort"] = pd.to_numeric(ordered["rank"], errors="coerce")
    ordered = ordered.sort_values(["target_id", "_rank_sort", "sample_id"])
    ordered = ordered.drop(columns=["_rank_sort"], errors="ignore")
    return ordered.groupby("target_id", as_index=False).first()


def select_best_of_k(metrics_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    """Select the best-scoring sample within top-k ranks per target."""

    if metrics_df.empty:
        return metrics_df.copy()

    candidates = metrics_df.copy()
    candidates["_rank_sort"] = pd.to_numeric(candidates["rank"], errors="coerce")
    candidates = candidates[candidates["_rank_sort"] <= topk].copy()
    if candidates.empty:
        return candidates

    candidates["_dockq_sort"] = _selection_metric(
        candidates,
        primary_column="dockq",
        fallback_column="pairwise_dockq_mean",
        fill_value=-1.0,
    )
    candidates["_fnat_sort"] = _selection_metric(
        candidates,
        primary_column="fnat",
        fallback_column="pairwise_fnat_mean",
        fill_value=-1.0,
    )
    candidates["_irmsd_sort"] = _selection_metric(
        candidates,
        primary_column="irmsd",
        fallback_column="pairwise_irmsd_mean",
        fill_value=float("inf"),
    )

    ordered = candidates.sort_values(
        ["target_id", "_dockq_sort", "_fnat_sort", "_irmsd_sort", "rank", "sample_id"],
        ascending=[True, False, False, True, True, True],
    )
    best = ordered.groupby("target_id", as_index=False).first()
    return best.drop(columns=["_dockq_sort", "_fnat_sort", "_irmsd_sort", "_rank_sort"], errors="ignore")


def summarize_subset(
    metrics_df: pd.DataFrame,
    all_rows: pd.DataFrame | None = None,
    include_invalid_rows: bool = False,
) -> dict[str, object]:
    """Summarize a metrics subset for JSON export."""

    source_rows = all_rows if all_rows is not None else metrics_df
    summary: dict[str, object] = {
        "count": int(len(metrics_df)),
        "total_rows_seen": int(len(source_rows)),
        "include_invalid_rows": bool(include_invalid_rows),
        "status_counts": (
            source_rows["status"].fillna("success").value_counts().sort_index().to_dict()
            if "status" in source_rows.columns
            else {"success": int(len(source_rows))}
        ),
        "metrics": {},
        "success_rates": {
            "dockq_ge_0_23": _rate(metrics_df, "dockq", lambda series: series >= 0.23),
            "dockq_ge_0_49": _rate(metrics_df, "dockq", lambda series: series >= 0.49),
            "dockq_ge_0_80": _rate(metrics_df, "dockq", lambda series: series >= 0.80),
            "irmsd_le_2_0": _rate(metrics_df, "irmsd", lambda series: series <= 2.0),
        },
        "average_lddt_ca": _mean(metrics_df, "lddt_ca"),
        "average_clash_count": _mean(metrics_df, "clash_count"),
        "average_clashes_per_1000_atoms": _mean(metrics_df, "clashes_per_1000_atoms"),
    }

    for column in SUMMARY_METRIC_COLUMNS:
        if column not in metrics_df.columns:
            continue
        series = pd.to_numeric(metrics_df[column], errors="coerce")
        summary["metrics"][column] = {
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std(ddof=0)),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
        }
    return summary


def _rate(metrics_df: pd.DataFrame, column: str, fn) -> float | None:
    """Compute a rate over the full subset, treating NaNs as failures."""

    if metrics_df.empty or column not in metrics_df.columns:
        return None
    series = pd.to_numeric(metrics_df[column], errors="coerce")
    mask = fn(series.fillna(float("inf") if "irmsd" in column else float("-inf")))
    return float(mask.mean())


def _mean(metrics_df: pd.DataFrame, column: str) -> float | None:
    """Compute a column mean or return None."""

    if metrics_df.empty or column not in metrics_df.columns:
        return None
    return _safe_float(pd.to_numeric(metrics_df[column], errors="coerce").mean())


def _safe_float(value: object) -> float | None:
    """Convert pandas numeric results to JSON-friendly floats."""

    if pd.isna(value):
        return None
    return float(value)


def _filter_summary_rows(metrics_df: pd.DataFrame, include_invalid_rows: bool) -> pd.DataFrame:
    """Filter rows used for summary statistics."""

    if include_invalid_rows or "status" not in metrics_df.columns:
        return metrics_df.copy()
    return metrics_df[metrics_df["status"].fillna("success").isin(VALID_SUMMARY_STATUSES)].copy()


def _selection_metric(
    metrics_df: pd.DataFrame,
    primary_column: str,
    fallback_column: str,
    fill_value: float,
) -> pd.Series:
    """Return a selection metric with an explicit fallback column."""

    if primary_column in metrics_df.columns:
        primary = pd.to_numeric(metrics_df[primary_column], errors="coerce")
    else:
        primary = pd.Series(float("nan"), index=metrics_df.index, dtype="float64")
    if fallback_column in metrics_df.columns:
        fallback = pd.to_numeric(metrics_df[fallback_column], errors="coerce")
    else:
        fallback = pd.Series(float("nan"), index=metrics_df.index, dtype="float64")
    return primary.fillna(fallback).fillna(fill_value)


def summarize_diagnostics(
    ordered_metrics: pd.DataFrame,
    all_rows: pd.DataFrame,
    topk: int,
    include_invalid_rows: bool,
    summary_by_method: bool | None,
    summary_by_confidence: bool,
) -> dict[str, object]:
    """Write benchmark-friendly diagnostic summaries."""

    filtered_all_rows = _filter_summary_rows(all_rows, include_invalid_rows=include_invalid_rows)
    filtered_metrics = _filter_summary_rows(ordered_metrics, include_invalid_rows=include_invalid_rows)
    top1_df = select_top1(filtered_metrics)
    best_of_k_df = select_best_of_k(filtered_metrics, topk=topk)
    top1_all_rows = select_top1(all_rows)
    best_of_k_all_rows = select_best_of_k(all_rows, topk=topk)

    auto_summary_by_method = bool(summary_by_method)
    if summary_by_method is None:
        auto_summary_by_method = "method" in all_rows.columns and all_rows["method"].fillna("").astype(str).str.strip().ne("").any()

    summary: dict[str, object] = {
        "include_invalid_rows": bool(include_invalid_rows),
        "overall": {
            "per_sample": benchmark_summary(filtered_all_rows, source_rows=all_rows),
            "top1": benchmark_summary(top1_df, source_rows=top1_all_rows),
            "best_of_k": benchmark_summary(best_of_k_df, source_rows=best_of_k_all_rows),
        },
        "by_target_id": _group_summaries(filtered_all_rows, "target_id"),
    }
    if summary_by_confidence:
        summary["by_confidence"] = {
            "per_sample": _group_summaries(filtered_all_rows, "mapping_confidence_label"),
            "top1": _group_summaries(top1_df, "mapping_confidence_label"),
            "best_of_k": _group_summaries(best_of_k_df, "mapping_confidence_label"),
        }
    if auto_summary_by_method:
        summary["by_method"] = {
            "per_sample": _group_summaries(filtered_all_rows, "method"),
            "top1": _group_summaries(top1_df, "method"),
            "best_of_k": _group_summaries(best_of_k_df, "method"),
        }
    return summary


def benchmark_summary(metrics_df: pd.DataFrame, source_rows: pd.DataFrame | None = None) -> dict[str, object]:
    """Return a benchmark-oriented summary with explainability statistics."""

    source = source_rows if source_rows is not None else metrics_df
    return {
        "count": int(len(metrics_df)),
        "success_rate_dockq_ge_0_23": _rate(metrics_df, "dockq", lambda series: series >= 0.23),
        "success_rate_dockq_ge_0_49": _rate(metrics_df, "dockq", lambda series: series >= 0.49),
        "success_rate_dockq_ge_0_80": _rate(metrics_df, "dockq", lambda series: series >= 0.80),
        "mean_mapping_confidence_score": _mean(metrics_df, "mapping_confidence_score"),
        "fraction_low_confidence_mapping": _status_fraction(source, "low_confidence_mapping"),
        "fraction_samples_using_sequence_fallback": _bool_fraction(source, "used_sequence_fallback"),
        "fraction_samples_using_ca_fallback_irmsd": _bool_fraction(source, "used_ca_fallback_for_irmsd"),
        "fraction_samples_using_ca_fallback_lrmsd": _bool_fraction(source, "used_ca_fallback_for_lrmsd"),
        "mean_interface_precision": _mean(metrics_df, "interface_precision"),
        "mean_interface_recall": _mean(metrics_df, "interface_recall"),
        "mean_interface_f1": _mean(metrics_df, "interface_f1"),
        "mean_dockq": _mean(metrics_df, "dockq"),
        "mean_pairwise_dockq_mean": _mean(metrics_df, "pairwise_dockq_mean"),
        "mean_fnat": _mean(metrics_df, "fnat"),
        "mean_pairwise_fnat_mean": _mean(metrics_df, "pairwise_fnat_mean"),
        "status_counts": (
            source["status"].fillna("success").value_counts().sort_index().to_dict()
            if "status" in source.columns
            else {}
        ),
    }


def _group_summaries(metrics_df: pd.DataFrame, column: str) -> dict[str, object]:
    """Return grouped benchmark summaries."""

    if metrics_df.empty or column not in metrics_df.columns:
        return {}
    values = metrics_df[column].fillna("").astype(str).str.strip()
    grouped = metrics_df.assign(_group_value=values)
    grouped = grouped[grouped["_group_value"] != ""]
    if grouped.empty:
        return {}
    return {
        group_value: benchmark_summary(group_df.drop(columns=["_group_value"], errors="ignore"))
        for group_value, group_df in grouped.groupby("_group_value")
    }


def _bool_fraction(metrics_df: pd.DataFrame, column: str) -> float | None:
    """Return the fraction of rows where a boolean-like column is true."""

    if metrics_df.empty or column not in metrics_df.columns:
        return None
    series = metrics_df[column].fillna(False).astype(bool)
    return float(series.mean())


def _status_fraction(metrics_df: pd.DataFrame, status: str) -> float | None:
    """Return the fraction of rows with a particular status."""

    if metrics_df.empty or "status" not in metrics_df.columns:
        return None
    return float(metrics_df["status"].fillna("success").eq(status).mean())
