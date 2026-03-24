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
from .evaluate import EvaluationConfig, REQUIRED_MANIFEST_COLUMNS, safe_evaluate_record

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
    _validate_manifest_columns(manifest_df)

    config = EvaluationConfig(
        contact_cutoff=args.contact_cutoff,
        interface_cutoff=args.interface_cutoff,
        include_all_atom_rmsd=args.include_all_atom_rmsd,
        include_lddt=args.include_lddt,
        include_clashes=args.include_clashes,
        ignore_hydrogens=args.ignore_hydrogens,
        sequence_fallback=args.sequence_fallback,
    )

    records = manifest_df.to_dict(orient="records")
    LOGGER.info("Evaluating %d samples with %d worker(s)", len(records), args.workers)
    successes, failures = _run_evaluation(records, config=config, manifest_dir=manifest_dir, workers=args.workers)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(successes)
    failures_df = pd.DataFrame(
        failures,
        columns=["sample_id", "target_id", "rank", "pred_path", "gt_path", "error", "config"],
    )

    if metrics_df.empty:
        LOGGER.warning("No samples were evaluated successfully.")
    write_aggregate_outputs(metrics_df=metrics_df, out_dir=out_dir, topk=args.topk)
    failures_df.to_csv(out_dir / "failures.csv", index=False)

    LOGGER.info("Completed: %d succeeded, %d failed", len(successes), len(failures))
    return 0 if successes else 1


def _run_evaluation(
    records: list[dict[str, object]],
    config: EvaluationConfig,
    manifest_dir: Path,
    workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run evaluation serially or with multiprocessing."""

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    if workers <= 1:
        iterator = (safe_evaluate_record(record, config=config, manifest_dir=manifest_dir) for record in records)
        for outcome in tqdm(iterator, total=len(records), desc="Evaluating", unit="sample"):
            if outcome.ok:
                successes.append(outcome.metrics or {})
            else:
                failures.append(outcome.failure or {})
        return successes, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {
            executor.submit(safe_evaluate_record, record, config, manifest_dir): record.get("sample_id", "")
            for record in records
        }
        for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample), desc="Evaluating", unit="sample"):
            outcome = future.result()
            if outcome.ok:
                successes.append(outcome.metrics or {})
            else:
                failures.append(outcome.failure or {})

    return successes, failures


def _validate_manifest_columns(manifest_df: pd.DataFrame) -> None:
    """Ensure the manifest contains the required columns."""

    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest_df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
