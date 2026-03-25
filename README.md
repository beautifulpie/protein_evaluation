# complex_eval

`complex_eval` is a robust, diagnosable protein-complex evaluation toolkit.

The repository is intentionally positioned as more than a plain metric calculator. It focuses on:

- robust evaluation under messy real-world mapping conditions
- explainable outputs that show why a score is low
- benchmark-friendly summaries for large evaluation runs

It supports binary receptor/ligand evaluation, multimer evaluation with explicit chain-group input, PDB/mmCIF parsing, structured failure reporting, mapping-confidence scoring, diagnostic tags, and optional validation against the official DockQ executable.

## Why Use This Instead Of A Plain Metric Calculator?

Many structure-evaluation tools return a score without telling you whether the mapping was trustworthy. That is risky in scientific workflows, especially when chain names drift, residue numbering shifts, atoms are missing, or partial structures are evaluated at scale.

`complex_eval` differentiates itself by making those conditions visible:

- mapping confidence is scored heuristically on `[0, 1]`
- each sample gets a confidence label: `high`, `medium`, or `low`
- diagnostic tags explain common failure modes such as `used_sequence_fallback`, `ambiguous_chain_mapping`, or `sparse_atom_coverage`
- interface contact errors are decomposed into recovered, missing, and false contacts
- JSONL diagnostics preserve the context needed to audit benchmark rows later

The goal is not to overclaim metric authority. The goal is to reduce the chance of silent misuse.

## Current Scope

Recommended use cases:

- binary docking or binder-design benchmarks with a clear receptor/ligand split
- multimer benchmarks where ordered chain groups are explicitly provided
- large batch evaluation pipelines where rows with `status != success` must be filtered safely
- regression-tested internal evaluation for research iteration

Not yet fully validated for:

- automatic global chain permutation search in highly symmetric multimers
- ambiguous chain remapping when no reliable group order is available
- strict equivalence to every external implementation on every edge case
- use as the sole source of truth for a paper without auditing mapping diagnostics

## Installation

Python 3.10+ is required.

Standard install:

```bash
cd protein_evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Pinned reproducible install:

```bash
pip install -r requirements-pinned.txt
pip install -e .
```

Entry points:

- `python -m complex_eval.cli`
- `complex-eval`

Dependency policy:

- `requirements.txt`: supported version ranges
- `requirements-pinned.txt`: exact tested versions
- `pyproject.toml`: package metadata and console entry point

## Package Layout

```text
complex_eval/
  __init__.py
  aggregate.py
  align.py
  cli.py
  diagnostics.py
  evaluate.py
  io_utils.py
  metrics.py
  multimer.py
  validation.py
  visualize.py
tests/
  fixtures/
  test_diagnostics.py
  test_metrics.py
  test_multimer.py
  test_regression.py
  test_safety.py
  test_validation.py
  test_visualize.py
examples/
  benchmark_visualization/
    README.md
    report.html
    plots/
```

## Manifest Formats

All manifests require:

- `sample_id`
- `target_id`
- `rank`
- `pred_path`
- `gt_path`

### Binary Manifest

Use these columns:

- `pred_receptor_chains`
- `pred_ligand_chains`
- `gt_receptor_chains`
- `gt_ligand_chains`

Example:

```csv
sample_id,target_id,rank,pred_path,gt_path,pred_receptor_chains,pred_ligand_chains,gt_receptor_chains,gt_ligand_chains,method
T001_r1,T001,1,preds/T001_rank1.pdb,gts/T001_native.cif,A,B,A,B,baseline
T001_r2,T001,2,preds/T001_rank2.pdb,gts/T001_native.cif,A,B,A,B,new_model
```

### Explicit Multimer Group Manifest

Use these columns:

- `pred_chain_groups`
- `gt_chain_groups`

Format:

- groups separated by `|`
- chain IDs inside each group separated by `,`

Example:

```csv
sample_id,target_id,rank,pred_path,gt_path,pred_chain_groups,gt_chain_groups
M001_r1,M001,1,preds/M001_rank1.pdb,gts/M001_native.cif,"A,B|C|D,E","A,B|C|D,E"
```

Important:

- group order is part of the scientific input
- this repository does not attempt symmetry-aware multimer permutation search
- if exactly two groups are supplied, the evaluator uses the binary metric path

## Quick Start

Basic strict evaluation:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --topk 5 \
  --workers 8
```

Equivalent module invocation:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results
```

Enable residue sequence fallback:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --sequence_fallback
```

Relax strict mapping to inspect low-confidence rows without a non-zero exit code:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --no-strict_mapping
```

Run external DockQ validation when available:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --validation_mode dockq \
  --dockq_executable DockQ
```

Hide explainability fields from per-sample outputs while preserving the core evaluator:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --no-include_explainability_fields
```

Disable visualization outputs when you only want machine-readable tables:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --no-write_visualizations
```

## Key CLI Safety Flags

- `--strict_mapping` / `--no-strict_mapping`
- `--min_matched_residue_fraction`
- `--min_sequence_identity`
- `--max_chain_length_difference`
- `--write_diagnostics_jsonl` / `--no-write_diagnostics_jsonl`
- `--include_explainability_fields` / `--no-include_explainability_fields`
- `--mapping_confidence_mode heuristic`
- `--summary_by_method` / `--no-summary_by_method`
- `--summary_by_confidence` / `--no-summary_by_confidence`
- `--write_visualizations` / `--no-write_visualizations`

## Core Outputs

The CLI writes:

- `results/per_sample_metrics.csv`
- `results/per_sample_diagnostics.jsonl`
- `results/per_target_best_of_k.csv`
- `results/summary_top1.json`
- `results/summary_best_of_k.json`
- `results/summary_diagnostics.json`
- `results/report.html`
- `results/plots/*.svg`
- `results/failures.csv`
- validation outputs when `--validation_mode dockq` is enabled

### `per_sample_metrics.csv`

This is the flat table intended for downstream analysis. It retains the existing scalar metrics and adds explainability-oriented columns such as:

- `mapping_confidence_score`
- `mapping_confidence_label`
- `diagnostic_tags`
- `dockq_decomposition_fnat_term`
- `dockq_decomposition_irmsd_term`
- `dockq_decomposition_lrmsd_term`
- `interface_native_contact_count`
- `interface_pred_contact_count`
- `interface_recovered_contact_count`
- `interface_missing_contact_count`
- `interface_false_contact_count`
- `interface_precision`
- `interface_recall`
- `interface_f1`
- `failure_category`

### `per_sample_diagnostics.jsonl`

This is the richer, explainable audit trail. Each JSON line contains:

- identifiers and paths
- evaluation mode and status
- core metrics
- mapping diagnostics
- interface diagnostics
- explainability fields
- warnings and failure reasons

### Visualization Outputs

The evaluator also writes a lightweight static visualization bundle without adding a plotting dependency:

- `results/report.html`
- `results/plots/status_counts.svg`
- `results/plots/confidence_label_counts.svg`
- `results/plots/diagnostic_tag_counts.svg`
- `results/plots/performance_snapshot.svg`
- `results/plots/method_mean_dockq.svg`
- `results/plots/mapping_confidence_vs_dockq.svg`
- `results/plots/interface_precision_vs_recall.svg`
- `results/plots/metric_*.svg` for core structural, interface, and clash metrics

These outputs are designed for benchmark review rather than publication figures. They help surface:

- top-1 and best-of-k prediction performance at a glance
- method-level mean DockQ when a `method` column is present
- failure-heavy or low-confidence-heavy runs
- disagreement between mapping confidence and DockQ
- interface recall/precision tradeoffs
- recurring diagnostic tags across a batch
- per-sample distributions for `ca_rmsd`, `all_atom_rmsd`, `irmsd`, `lrmsd`, `fnat`, `dockq`, `lddt_ca`, clash metrics, and interface precision/recall/F1

### Bundled Example Visualization

A checked-in example visualization bundle is included here:

- [examples/benchmark_visualization/report.html](/home/jung/protein_evaluation/examples/benchmark_visualization/report.html)
- [examples/benchmark_visualization/plots/status_counts.svg](/home/jung/protein_evaluation/examples/benchmark_visualization/plots/status_counts.svg)
- [examples/benchmark_visualization/plots/confidence_label_counts.svg](/home/jung/protein_evaluation/examples/benchmark_visualization/plots/confidence_label_counts.svg)
- [examples/benchmark_visualization/source/manifest.csv](/home/jung/protein_evaluation/examples/benchmark_visualization/source/manifest.csv)

This bundled example is generated from 12 deterministic toy protein-complex fixtures under `examples/benchmark_visualization/source/`. It is intended as a stable reference for reviewers and repository visitors, not as a substitute for a real external benchmark.

### `summary_diagnostics.json`

This is the benchmark-facing summary report. It adds:

- `success_rate_dockq_ge_0_23`
- `success_rate_dockq_ge_0_49`
- `success_rate_dockq_ge_0_80`
- `mean_mapping_confidence_score`
- `fraction_low_confidence_mapping`
- `fraction_samples_using_sequence_fallback`
- `fraction_samples_using_ca_fallback_irmsd`
- `fraction_samples_using_ca_fallback_lrmsd`
- `mean_interface_precision`
- `mean_interface_recall`
- `mean_interface_f1`

It also includes:

- overall summaries for per-sample, top-1, and best-of-k views
- optional confidence-stratified summaries
- per-target summaries
- optional method-stratified summaries when a `method` column exists

## Mapping Confidence

`mapping_confidence_score` is a deterministic heuristic in `[0, 1]`.

It is based on:

- matched residue coverage
- sequence identity of mapped chains
- matched atom coverage
- chain length agreement

It is then penalized for:

- sequence fallback use
- ambiguous positional chain mapping
- sparse atom coverage
- non-fallback parse warnings
- CA fallback use in interface metrics

Labels:

- `high`: `>= 0.85`
- `medium`: `>= 0.60` and `< 0.85`
- `low`: `< 0.60`

This score is intentionally heuristic and interpretable. It is not a calibrated probability of correctness.

## Diagnostic Tags

Current normalized tags include:

- `clean_mapping`
- `used_sequence_fallback`
- `low_matched_residue_fraction`
- `low_sequence_identity`
- `chain_length_mismatch`
- `ambiguous_chain_mapping`
- `sparse_atom_coverage`
- `used_ca_fallback_irmsd`
- `used_ca_fallback_lrmsd`
- `parse_warning_present`
- `partial_interface_observed`
- `false_interface_contacts_present`
- `likely_chain_swap`
- `likely_numbering_offset`

These tags are attached even when `status == success`.

Normalized failure categories include:

- `parse_error`
- `mapping_error`
- `metric_computation_error`
- `invalid_input`
- `unknown_error`

## Explainability Fields

Additional per-sample explainability fields include:

- `dockq_decomposition_fnat_term`
- `dockq_decomposition_irmsd_term`
- `dockq_decomposition_lrmsd_term`
- `interface_native_contact_count`
- `interface_pred_contact_count`
- `interface_recovered_contact_count`
- `interface_missing_contact_count`
- `interface_false_contact_count`
- `interface_precision`
- `interface_recall`
- `interface_f1`

These complement `fnat` and `dockq` by making the failure mode clearer.

## Realistic Diagnostic JSON Example

Example from the deterministic benchmark fixture:

```json
{
  "sample_id": "identical",
  "target_id": "T001",
  "rank": 1,
  "evaluation_mode": "binary",
  "status": "success",
  "core_metrics": {
    "dockq": 1.0,
    "fnat": 1.0,
    "irmsd": 0.0,
    "lrmsd": 0.0,
    "matched_residue_fraction": 1.0
  },
  "mapping_diagnostics": {
    "mapping_confidence_score": 1.0,
    "mapping_confidence_label": "high",
    "receptor_matched_residue_fraction": 1.0,
    "ligand_matched_residue_fraction": 1.0
  },
  "interface_diagnostics": {
    "interface_native_contact_count": 5,
    "interface_pred_contact_count": 5,
    "interface_recovered_contact_count": 5,
    "interface_missing_contact_count": 0,
    "interface_false_contact_count": 0,
    "interface_precision": 1.0,
    "interface_recall": 1.0,
    "interface_f1": 1.0
  },
  "explainability": {
    "diagnostic_tags": ["clean_mapping"],
    "dockq_decomposition_fnat_term": 1.0,
    "dockq_decomposition_irmsd_term": 1.0,
    "dockq_decomposition_lrmsd_term": 1.0
  }
}
```

## Benchmark Summary Example

Representative `summary_diagnostics.json` fragment:

```json
{
  "overall": {
    "best_of_k": {
      "count": 2,
      "success_rate_dockq_ge_0_23": 1.0,
      "mean_mapping_confidence_score": 1.0,
      "fraction_low_confidence_mapping": 0.0,
      "mean_interface_precision": 1.0,
      "mean_interface_recall": 1.0,
      "mean_interface_f1": 1.0
    }
  },
  "by_confidence": {
    "best_of_k": {
      "high": {
        "count": 2,
        "mean_mapping_confidence_score": 1.0
      }
    }
  }
}
```

## Recommended Filtering For Paper Tables

At minimum, check:

- `status == success`
- `mapping_low_confidence_reasons` is empty
- `mapping_confidence_label != low`
- `mapping_confidence_score >= 0.85` for strict benchmark tables
- `parse_warning` is empty or explicitly reviewed

Recommended binary filter:

- `status == success`
- `mapping_confidence_score >= 0.85`
- `receptor_min_chain_sequence_identity >= 0.70`
- `ligand_min_chain_sequence_identity >= 0.70`
- no multi-chain positional remapping unless manually reviewed

Recommended multimer filter:

- `status == success`
- `mapping_confidence_score >= 0.85`
- `group_min_matched_residue_fraction >= 0.80`
- `group_min_chain_sequence_identity >= 0.70`
- `group_used_positional_chain_mapping_any == False`

## Metric Notes

The repository still computes the existing scalar metrics:

- `ca_rmsd`
- `all_atom_rmsd`
- `irmsd`
- `lrmsd`
- `fnat`
- `dockq`
- `lddt_ca`
- internal clash metrics

For true multimer evaluation:

- full-complex global metrics are still reported
- interface metrics are computed pairwise across explicit chain groups
- `dockq`, `irmsd`, and `lrmsd` are intentionally left unset at the top level for `evaluation_mode=multimer`
- use `pairwise_*` fields instead

## External Validation

`--validation_mode dockq` compares internal metrics against the official DockQ executable when available.

It writes:

- `validation_dockq_per_sample.csv`
- `validation_dockq_summary.json`

Current validation limitations:

- binary rows only
- one chain per side only

If DockQ is unavailable, the repository degrades gracefully and records validation status as `unavailable`.

## Limitations And Assumptions

- chain-group order is trusted input
- alternate locations are reduced to the highest-occupancy altloc
- hydrogens are ignored by default
- sequence fallback helps with numbering offsets but can hide bad manifests if overused
- clash metrics are approximate and are not MolProbity clashscores
- true symmetry-aware multimer matching remains out of scope

## Development And Testing

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

Run the deterministic benchmark smoke test:

```bash
python -m complex_eval.cli \
  --manifest tests/fixtures/benchmark/manifest.csv \
  --out_dir smoke_results \
  --workers 1
```

The GitHub Actions workflow installs the pinned dependency set, installs the package through `pyproject.toml`, and runs the full CPU test suite.

## License

MIT. See `LICENSE`.
