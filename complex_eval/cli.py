"""Command-line interface for protein complex evaluation."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .aggregate import write_aggregate_outputs
from .diagnostics import strip_explainability_fields, write_per_sample_diagnostics_jsonl
from .evaluate import EvaluationConfig, safe_evaluate_record, validate_manifest_columns
from .validation import write_validation_outputs
from .visualize import write_visualization_outputs

LOGGER = logging.getLogger("complex_eval")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV manifest describing prediction/native pairs.")
    parser.add_argument("--out_dir", required=True, help="Output directory for metrics and summaries.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k ranks to consider for best-of-k summaries.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument(
        "--contact_cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom cutoff in Angstrom for native residue-residue contacts.",
    )
    parser.add_argument(
        "--interface_cutoff",
        type=float,
        default=10.0,
        help="Heavy-atom cutoff in Angstrom for native interface residue detection.",
    )
    parser.add_argument(
        "--include_all_atom_rmsd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable all-heavy-atom RMSD calculation.",
    )
    parser.add_argument(
        "--include_lddt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CA-based lDDT calculation.",
    )
    parser.add_argument(
        "--include_clashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable approximate steric clash counting.",
    )
    parser.add_argument(
        "--ignore_hydrogens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore hydrogens during structure parsing and atom matching.",
    )
    parser.add_argument(
        "--sequence_fallback",
        action="store_true",
        help="Use sequence-based residue matching when numbering-based matching is poor.",
    )
    parser.add_argument(
        "--strict_mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat low-confidence residue/chain mappings as invalid rows for strict research workflows.",
    )
    parser.add_argument(
        "--min_matched_residue_fraction",
        type=float,
        default=0.7,
        help="Minimum per-side matched residue fraction before a row is flagged as low-confidence.",
    )
    parser.add_argument(
        "--min_sequence_identity",
        type=float,
        default=0.5,
        help="Minimum per-chain sequence identity before a row is flagged as low-confidence.",
    )
    parser.add_argument(
        "--max_chain_length_difference",
        type=int,
        default=10,
        help="Maximum allowed absolute per-chain length difference before a row is flagged as low-confidence.",
    )
    parser.add_argument(
        "--include_failed_rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include failed rows in per_sample_metrics.csv with status/error metadata.",
    )
    parser.add_argument(
        "--include_invalid_rows_in_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include low-confidence and failed rows in summary statistics instead of filtering to status=success.",
    )
    parser.add_argument(
        "--validation_mode",
        choices=("none", "dockq"),
        default="none",
        help="Optional post-hoc validation against an external reference implementation.",
    )
    parser.add_argument(
        "--dockq_executable",
        default=None,
        help="Optional explicit DockQ executable path for --validation_mode dockq.",
    )
    parser.add_argument(
        "--write_diagnostics_jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write results/per_sample_diagnostics.jsonl with structured per-sample diagnostics.",
    )
    parser.add_argument(
        "--include_explainability_fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include explainability fields in written CSV/JSON outputs.",
    )
    parser.add_argument(
        "--mapping_confidence_mode",
        choices=("heuristic",),
        default="heuristic",
        help="Mapping confidence scoring mode.",
    )
    parser.add_argument(
        "--summary_by_method",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write method-stratified diagnostic summaries. Defaults to auto when a method column exists.",
    )
    parser.add_argument(
        "--summary_by_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write confidence-stratified diagnostic summaries.",
    )
    parser.add_argument(
        "--write_visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write lightweight HTML/SVG visualization outputs for diagnostic review.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest_dir = manifest_path.parent

    LOGGER.info("Reading manifest from %s", manifest_path)
    manifest_df = pd.read_csv(manifest_path)
    validate_manifest_columns(manifest_df.columns)

    config = EvaluationConfig(
        contact_cutoff=args.contact_cutoff,
        interface_cutoff=args.interface_cutoff,
        include_all_atom_rmsd=args.include_all_atom_rmsd,
        include_lddt=args.include_lddt,
        include_clashes=args.include_clashes,
        ignore_hydrogens=args.ignore_hydrogens,
        sequence_fallback=args.sequence_fallback,
        strict_mapping=args.strict_mapping,
        min_matched_residue_fraction=args.min_matched_residue_fraction,
        min_sequence_identity=args.min_sequence_identity,
        max_chain_length_difference=args.max_chain_length_difference,
        mapping_confidence_mode=args.mapping_confidence_mode,
    )

    records = manifest_df.to_dict(orient="records")
    LOGGER.info("Evaluating %d samples with %d worker(s)", len(records), args.workers)
    successes, failures, all_rows = _run_evaluation(
        records,
        config=config,
        manifest_dir=manifest_dir,
        workers=args.workers,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = list(successes)
    if args.include_failed_rows:
        metrics_rows.extend(failures)
    if not args.include_explainability_fields:
        metrics_rows = [strip_explainability_fields(row) for row in metrics_rows]
    metrics_df = pd.DataFrame(metrics_rows)
    all_rows_for_reports = [row.copy() for row in all_rows]
    diagnostics_rows = (
        all_rows_for_reports
        if args.include_explainability_fields
        else [strip_explainability_fields(row) for row in all_rows_for_reports]
    )
    all_rows_df = pd.DataFrame(all_rows_for_reports)
    failures_df = pd.DataFrame(
        failures,
        columns=[
            "sample_id",
            "target_id",
            "rank",
            "method",
            "pred_path",
            "gt_path",
            "status",
            "failure_category",
            "error_type",
            "error_message",
            "mapping_low_confidence_reasons",
            "config",
        ],
    )

    if metrics_df.empty:
        LOGGER.warning("No samples were evaluated successfully.")
    summary_diagnostics = write_aggregate_outputs(
        metrics_df=metrics_df,
        out_dir=out_dir,
        topk=args.topk,
        include_invalid_rows_in_summary=args.include_invalid_rows_in_summary,
        all_rows_df=all_rows_df,
        summary_by_method=args.summary_by_method,
        summary_by_confidence=args.summary_by_confidence,
    )
    failures_df.to_csv(out_dir / "failures.csv", index=False)
    if args.write_diagnostics_jsonl:
        write_per_sample_diagnostics_jsonl(
            rows=diagnostics_rows,
            path=out_dir / "per_sample_diagnostics.jsonl",
            include_explainability_fields=args.include_explainability_fields,
        )
    if args.write_visualizations:
        write_visualization_outputs(
            all_rows_df=all_rows_df,
            summary_diagnostics=summary_diagnostics,
            out_dir=out_dir,
        )
    if args.validation_mode != "none" and not metrics_df.empty:
        validation_summary = write_validation_outputs(
            metrics_df=metrics_df,
            out_dir=out_dir,
            mode=args.validation_mode,
            dockq_executable=args.dockq_executable,
            include_invalid_rows=args.include_invalid_rows_in_summary,
        )
        LOGGER.info("Validation summary status: %s", validation_summary.get("status", "unknown"))

    low_confidence_count = int(
        pd.Series([row.get("status", "success") for row in successes]).eq("low_confidence_mapping").sum()
    )
    success_count = int(pd.Series([row.get("status", "success") for row in successes]).eq("success").sum())

    LOGGER.info(
        "Completed: %d success, %d low-confidence, %d failed",
        success_count,
        low_confidence_count,
        len(failures),
    )
    if success_count == 0:
        return 1
    if args.strict_mapping and low_confidence_count > 0:
        return 2
    return 0


def _run_evaluation(
    records: list[dict[str, object]],
    config: EvaluationConfig,
    manifest_dir: Path,
    workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run evaluation serially or with multiprocessing."""

    successes: list[tuple[int, dict[str, object]]] = []
    failures: list[tuple[int, dict[str, object]]] = []

    if workers <= 1:
        iterator = (
            (index, safe_evaluate_record(record, config=config, manifest_dir=manifest_dir))
            for index, record in enumerate(records)
        )
        for index, outcome in tqdm(iterator, total=len(records), desc="Evaluating", unit="sample"):
            if outcome.ok:
                successes.append((index, outcome.metrics or {}))
            else:
                failures.append((index, outcome.failure or {}))
        return _sorted_rows(successes), _sorted_rows(failures), _sorted_rows([*successes, *failures])

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {
            executor.submit(safe_evaluate_record, record, config, manifest_dir): index
            for index, record in enumerate(records)
        }
        for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample), desc="Evaluating", unit="sample"):
            index = future_to_sample[future]
            outcome = future.result()
            if outcome.ok:
                successes.append((index, outcome.metrics or {}))
            else:
                failures.append((index, outcome.failure or {}))

    return _sorted_rows(successes), _sorted_rows(failures), _sorted_rows([*successes, *failures])


def _sorted_rows(rows: list[tuple[int, dict[str, object]]]) -> list[dict[str, object]]:
    """Return deterministic row ordering by manifest order."""

    return [row for _, row in sorted(rows, key=lambda item: item[0])]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
