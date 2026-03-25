# Engineering Report

## Summary of Changes

- Added explicit chain and residue mapping diagnostics to each evaluated row.
- Added structured row status fields: `success`, `low_confidence_mapping`, and `failed`.
- Added configurable mapping confidence thresholds and strict mapping mode.
- Made summaries exclude invalid rows by default.
- Added deterministic ordering for parallel evaluation outputs.
- Added optional external validation against DockQ with graceful fallback when the tool is unavailable.
- Added edge-case tests and deterministic end-to-end regression fixtures.
- Constrained dependency versions for more reproducible installations.

## Risk Assessment

### Before

- Chain name mismatches could fall back to positional mapping with limited visibility.
- Sequence fallback had little quantitative quality reporting.
- Batch failures and low-confidence mappings were easy to confuse with valid metric rows.
- Multiprocessing output ordering was not explicitly stabilized.
- There was no stored end-to-end regression fixture set and no external-reference validation path.

### After

- Every evaluated row now carries mapping diagnostics and an explicit `status`.
- Low-confidence mappings are surfaced with concrete reasons and excluded from summaries by default.
- Strict mode turns low-confidence mappings into visible non-zero-exit outcomes for batch workflows.
- Multiprocessing results are emitted in deterministic manifest order.
- DockQ comparison can now be generated when the external tool is available.
- Regression and edge-case coverage now covers mapping, parsing, fallback behavior, malformed manifests, and worker consistency.

## Unresolved Risks

- Multi-chain DockQ validation remains limited. The wrapper currently validates only one-chain-per-side examples.
- Symmetric complexes may still require domain-specific chain assignment logic beyond generic positional safeguards.
- Internal metrics are still approximations unless compared to external references such as DockQ.
- Real-world mmCIF/PDB pathology is broad; more curated fixture coverage is still advisable.

## Suggested Next Validation Steps

1. Run `--validation_mode dockq` on a representative benchmark subset and inspect the diff summary.
2. Add curated real-world fixtures covering biologically realistic edge cases from your target benchmark.
3. Compare summary outputs against an independently curated baseline before any paper submission.
4. Extend external validation to additional trusted tools if your benchmark depends on them.

## Exact Files Changed

- `README.md`
- `requirements.txt`
- `complex_eval/aggregate.py`
- `complex_eval/cli.py`
- `complex_eval/evaluate.py`
- `complex_eval/metrics.py`
- `complex_eval/validation.py`
- `tests/test_safety.py`
- `tests/test_regression.py`
- `tests/test_validation.py`
- `tests/fixtures/benchmark/native.pdb`
- `tests/fixtures/benchmark/pred_identical.pdb`
- `tests/fixtures/benchmark/pred_transformed.pdb`
- `tests/fixtures/benchmark/manifest.csv`
- `tests/fixtures/benchmark/expected_per_sample.json`
- `tests/fixtures/benchmark/expected_summary_top1.json`
