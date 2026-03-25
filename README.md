# complex_eval

`complex_eval` is a safety-oriented Python toolkit for evaluating predicted protein complexes against reference structures. It is designed for research batch workflows where visible failure is preferable to a silent wrong score.

The repository supports:

- binary complexes with explicit receptor/ligand chain splits
- multimer complexes when the manifest defines ordered chain groups
- PDB and mmCIF inputs
- strict mapping diagnostics, structured row status fields, and optional external DockQ validation

## What This Evaluator Is Safe For

Recommended use cases:

- binary docking or binder-design evaluation with a clear receptor/ligand split
- multimer evaluation where chain-group correspondence is known and provided explicitly in the manifest
- batch experiments where rows with `status != success` are filtered out before reporting metrics
- regression testing and large-scale sweeps where deterministic failure visibility matters

Not yet fully validated for:

- automatic permutation resolution in highly symmetric multimers
- ambiguous chain remapping without user-provided group order
- exact equivalence to every external metric implementation on every edge case
- paper-ready reporting without checking mapping diagnostics and, ideally, external validation on a benchmark subset

## Installation

Python 3.10+ is required.

Standard installation:

```bash
cd protein_evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Reproducible installation using the tested dependency set:

```bash
pip install -r requirements-pinned.txt
pip install -e .
```

Package entry points:

- module form: `python -m complex_eval.cli`
- console script form: `complex-eval`

Dependency policy:

- `requirements.txt`: supported version ranges
- `requirements-pinned.txt`: exact versions used for regression-tested reruns
- `pyproject.toml`: package metadata and console entry point

## Package Layout

```text
complex_eval/
  __init__.py
  aggregate.py
  align.py
  cli.py
  evaluate.py
  io_utils.py
  metrics.py
  multimer.py
  validation.py
tests/
  fixtures/
  test_metrics.py
  test_multimer.py
  test_regression.py
  test_safety.py
  test_validation.py
```

## Manifest Formats

All manifests require:

- `sample_id`
- `target_id`
- `rank`
- `pred_path`
- `gt_path`

### Legacy Binary Manifest

Use these columns for receptor/ligand evaluation:

- `pred_receptor_chains`
- `pred_ligand_chains`
- `gt_receptor_chains`
- `gt_ligand_chains`

Example:

```csv
sample_id,target_id,rank,pred_path,gt_path,pred_receptor_chains,pred_ligand_chains,gt_receptor_chains,gt_ligand_chains
T001_r1,T001,1,preds/T001_rank1.pdb,gts/T001_native.cif,"A","B","A","B"
T002_r3,T002,3,preds/T002_rank3.cif,gts/T002_native.pdb,"A,B","C,D","X,Y","Z,W"
```

### Multimer Group Manifest

Use these columns when the complex contains more than two manifest-defined partners:

- `pred_chain_groups`
- `gt_chain_groups`

Format:

- groups are separated by `|`
- chain IDs inside a group are separated by `,`

Example:

```csv
sample_id,target_id,rank,pred_path,gt_path,pred_chain_groups,gt_chain_groups
M001_r1,M001,1,preds/M001_rank1.pdb,gts/M001_native.cif,"A,B|C|D,E","A,B|C|D,E"
```

Interpretation:

- group 1 is `A,B`
- group 2 is `C`
- group 3 is `D,E`

Important:

- group order is trusted input
- `complex_eval` does not attempt global multimer chain permutation search
- if exactly two groups are provided, the evaluator uses the binary metric path

## Usage

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

Enable sequence-based residue matching fallback:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --sequence_fallback
```

Inspect low-confidence rows without failing the run:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --no-strict_mapping
```

Tune mapping thresholds:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --min_matched_residue_fraction 0.85 \
  --min_sequence_identity 0.70 \
  --max_chain_length_difference 5
```

Include failed rows in `per_sample_metrics.csv`:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --include_failed_rows
```

Include invalid rows in summaries:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --include_invalid_rows_in_summary
```

Validate against the official DockQ executable when it is installed:

```bash
complex-eval \
  --manifest manifest.csv \
  --out_dir results \
  --validation_mode dockq \
  --dockq_executable DockQ
```

Notes on validation mode:

- the repository degrades gracefully when DockQ is unavailable
- current external DockQ validation is limited to binary one-chain-per-side rows
- validation outputs are written separately and never silently replace internal metrics

## Outputs

The CLI writes:

- `results/per_sample_metrics.csv`
- `results/per_target_best_of_k.csv`
- `results/summary_top1.json`
- `results/summary_best_of_k.json`
- `results/failures.csv`
- `results/validation_dockq_per_sample.csv` when `--validation_mode dockq`
- `results/validation_dockq_summary.json` when `--validation_mode dockq`

### Fields You Must Check Before Using Results In A Paper

At minimum, check:

- `status == success`
- `mapping_low_confidence_reasons` is empty
- `matched_residue_fraction` is high enough for the benchmark
- `mapping_confidence_score` is not suspiciously low
- `parse_warning` is empty or explicitly reviewed

Recommended filtering criteria for binary studies:

- `status == success`
- `matched_residue_fraction >= 0.8`
- `receptor_min_chain_sequence_identity >= 0.7`
- `ligand_min_chain_sequence_identity >= 0.7`
- no positional multi-chain mapping unless manually reviewed

Recommended filtering criteria for multimer studies:

- `status == success`
- `group_min_matched_residue_fraction >= 0.8`
- `group_min_chain_sequence_identity >= 0.7`
- `group_used_positional_chain_mapping_any == False`
- inspect `pairwise_interface_metrics_json` for the benchmark subset used in reporting

### Key Per-sample Columns

Common fields:

- `sample_id`
- `target_id`
- `rank`
- `evaluation_mode`
- `status`
- `error_type`
- `error_message`
- `mapping_low_confidence_reasons`
- `ca_rmsd`
- `all_atom_rmsd`
- `fnat`
- `lddt_ca`
- `clash_count`
- `clashes_per_1000_atoms`
- `matched_residue_fraction`
- `matched_atom_fraction`
- `num_matched_residues`
- `num_matched_atoms`
- `mapping_confidence_score`
- `used_sequence_fallback`
- `parse_warning`

Binary-specific mapping diagnostics include:

- `receptor_num_matched_residues`
- `ligand_num_matched_residues`
- `receptor_num_unmatched_gt_residues`
- `ligand_num_unmatched_gt_residues`
- `receptor_min_chain_sequence_identity`
- `ligand_min_chain_sequence_identity`
- `receptor_max_chain_length_difference`
- `ligand_max_chain_length_difference`
- `receptor_chain_mapping_strategy`
- `ligand_chain_mapping_strategy`
- `receptor_used_positional_chain_mapping`
- `ligand_used_positional_chain_mapping`
- `dockq`
- `irmsd`
- `lrmsd`

True multimer-specific fields include:

- `num_chain_groups`
- `pairwise_interface_count`
- `pairwise_interfaces_with_native_contacts`
- `pairwise_fnat_mean`
- `pairwise_irmsd_mean`
- `pairwise_lrmsd_mean`
- `pairwise_dockq_mean`
- `pairwise_dockq_min`
- `pairwise_dockq_acceptable_fraction`
- `pairwise_interface_metrics_json`
- `group_min_matched_residue_fraction`
- `group_min_chain_sequence_identity`
- `group_max_chain_length_difference`
- `group_used_positional_chain_mapping_any`
- `group_mapping_summary`

Important multimer safety behavior:

- for `evaluation_mode=multimer`, `dockq`, `irmsd`, and `lrmsd` are intentionally left as `NaN`
- use `pairwise_*` fields instead

## Metric Definitions

### Global Metrics

- `ca_rmsd`: global Cα RMSD after optimal rigid superposition over matched residues
- `all_atom_rmsd`: all-heavy-atom RMSD using matched residues and atom names
- `lddt_ca`: Cα-based lDDT using native pairs within `15.0 Å`
- `clash_count`: internal approximate heavy-atom clash count
- `clashes_per_1000_atoms`: `1000 * clash_count / num_heavy_atoms`

### Binary Interface Metrics

- native interface residues: residues with any heavy atom within `10.0 Å` of the opposite partner
- `irmsd`: interface backbone RMSD using `N`, `CA`, `C`, `O`, with `CA` fallback if needed
- `lrmsd`: ligand RMSD after receptor superposition
- `fnat`: fraction of native residue-residue contacts recovered, with a heavy-atom cutoff of `5.0 Å`
- `dockq`: internal DockQ implementation

```text
DockQ = (Fnat + 1 / (1 + (iRMSD / 1.5)^2) + 1 / (1 + (LRMSD / 8.5)^2)) / 3
```

### Multimer Interface Metrics

For `evaluation_mode=multimer`:

- full-complex metrics are computed across all manifest-defined groups together
- interface metrics are computed pairwise for every unique group pair
- `fnat` is reported as the recovered fraction across all pairwise native contacts
- `dockq`, `irmsd`, and `lrmsd` are not collapsed into a single scalar because that behavior is not validated here

## Aggregation

- `summary_top1.json` summarizes the top-ranked row per target
- `summary_best_of_k.json` summarizes the best row among `rank <= topk`
- by default, summaries exclude rows whose `status` is not `success`

Selection behavior:

- binary rows are ranked by `dockq`, then `fnat`, then `irmsd`, then `rank`
- multimer rows fall back to `pairwise_dockq_mean`, then `pairwise_fnat_mean`, then `pairwise_irmsd_mean`, then `rank`

## External Validation

When `--validation_mode dockq` is enabled and the DockQ executable is available:

- internal vs external metrics are compared per sample
- per-sample diffs are written to CSV
- summary JSON includes mean/max absolute diffs
- Pearson and Spearman correlations are reported when enough rows exist
- explicit pass/fail thresholds are written to the summary

If DockQ is unavailable:

- validation summary status is set to `unavailable`
- evaluation results are still written
- the repository does not silently invent reference values

## Assumptions And Limitations

- chain-group order is part of the scientific input and must be provided correctly
- alternate locations are reduced to the highest-occupancy altloc per atom name
- hydrogens are ignored by default
- sequence fallback can improve coverage for numbering offsets but may hide bad manifests if used indiscriminately
- internal clash metrics are approximations and are not MolProbity clashscores
- external validation currently targets binary one-chain-per-side DockQ comparisons only
- true symmetry-aware multimer chain permutation search is out of scope for this repository

## Development And Testing

Run the full CPU test suite:

```bash
python -m unittest discover -s tests -v
```

Build and install the package locally:

```bash
python -m pip install --upgrade pip
pip install -r requirements-pinned.txt
pip install -e .
```

The GitHub Actions workflow installs the pinned dependency set and then installs the package through `pyproject.toml` before running tests.

## License

This project is released under the MIT License. See `LICENSE`.
