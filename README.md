# Protein_complex_eval

`complex_eval` is a small Python toolkit for evaluating predicted binary protein complexes against ground-truth structures. It supports PDB and mmCIF input, multi-chain receptor/ligand partners, per-sample metrics, and top-1 / best-of-k aggregation.

## Features

- Binary complex evaluation with a receptor/ligand split, where each side may contain one or more chains.
- PDB and mmCIF parsing through `gemmi`.
- Pure-Python implementations of Kabsch alignment, RMSD metrics, interface residue detection, Fnat, DockQ, lDDT-Cα, and an approximate internal clash metric.
- Optional sequence-based residue matching fallback when residue numbering does not line up well.
- Parallel evaluation over a manifest CSV with structured failure reporting.
- Explicit mapping diagnostics and row status fields so low-confidence mappings are visible instead of silently accepted.
- Optional validation output against the official DockQ executable when it is available.

## Research Safety

This evaluator is safest for:

- binary complexes with a clear receptor/ligand split
- structures with consistent chain partitioning between prediction and native reference
- cases where residue numbering is mostly consistent or sequence fallback is known to be appropriate
- batch evaluation workflows where rows with `status != success` are filtered out

This evaluator is not yet fully validated for:

- ambiguous multi-chain partner remapping beyond the manifest-provided binary split
- highly symmetric complexes where multiple chain assignments are equally plausible
- exact equivalence with every external metric implementation in all edge cases
- use as the only source of truth in a paper without checking mapping diagnostics and, ideally, external validation

## Installation

```bash
cd protein_evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required.

For reproducible experiments, prefer the exact tested dependency set:

```bash
pip install -r requirements-pinned.txt
```

Repository policy:

- `requirements.txt`: supported version ranges for general installation
- `requirements-pinned.txt`: exact versions used for deterministic experiment reruns and regression reproduction

## Package Layout

```text
complex_eval/
  __init__.py
  io_utils.py
  align.py
  metrics.py
  evaluate.py
  aggregate.py
  cli.py
tests/
  test_metrics.py
requirements.txt
README.md
```

## Manifest Format

The input manifest must be a CSV with these columns:

- `sample_id`
- `target_id`
- `rank`
- `pred_path`
- `gt_path`
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

Chain columns may contain comma-separated IDs. Relative structure paths are resolved relative to the manifest directory.

## Usage

Basic evaluation:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --topk 5 \
  --workers 8
```

The CLI now defaults to strict mapping mode. Rows with low-confidence residue or chain mappings are marked as `status=low_confidence_mapping`, excluded from summaries by default, and cause a non-zero exit code in strict mode.

Enable sequence-based residue matching fallback explicitly:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --workers 4 \
  --sequence_fallback
```

Disable strict mapping if you want to inspect low-confidence rows without failing the run:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --sequence_fallback \
  --no-strict_mapping
```

Include failed rows in `per_sample_metrics.csv` and opt in to summaries that include invalid rows:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --include_failed_rows \
  --include_invalid_rows_in_summary
```

Run external validation against DockQ when the executable is installed:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --validation_mode dockq \
  --dockq_executable DockQ
```

Disable optional expensive metrics if needed:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --no-include_all_atom_rmsd \
  --no-include_lddt \
  --no-include_clashes
```

## Outputs

The CLI writes:

- `results/per_sample_metrics.csv`
- `results/per_target_best_of_k.csv`
- `results/summary_top1.json`
- `results/summary_best_of_k.json`
- `results/failures.csv`
- `results/validation_dockq_per_sample.csv` when `--validation_mode dockq`
- `results/validation_dockq_summary.json` when `--validation_mode dockq`

### Per-sample CSV columns

Representative columns include:

- `sample_id`
- `target_id`
- `rank`
- `status`
- `error_type`
- `error_message`
- `mapping_low_confidence_reasons`
- `ca_rmsd`
- `all_atom_rmsd`
- `irmsd`
- `lrmsd`
- `fnat`
- `dockq`
- `lddt_ca`
- `clash_count`
- `clashes_per_1000_atoms`
- `matched_residue_fraction`
- `matched_atom_fraction`
- `num_matched_residues`
- `num_matched_atoms`
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
- `used_sequence_fallback`
- `used_ca_fallback_for_irmsd`
- `used_ca_fallback_for_lrmsd`
- `parse_warning`

## Metric Definitions

- `ca_rmsd`: global Cα RMSD after optimal rigid superposition over matched residues across the whole complex.
- `all_atom_rmsd`: all-heavy-atom RMSD using matched residue/atom names. Hydrogens are ignored by default.
- Native interface residues: residues in the ground-truth receptor or ligand that have any heavy atom within `10.0 Å` of the opposite partner.
- `irmsd`: interface RMSD on interface backbone atoms (`N`, `CA`, `C`, `O`). Falls back to `CA` when backbone coverage is too poor.
- `lrmsd`: ligand RMSD after superposing the predicted receptor onto the native receptor using receptor backbone atoms, with `CA` fallback if needed.
- `fnat`: fraction of native residue-residue contacts recovered by the prediction, where a contact is any heavy-atom pair within `5.0 Å`.
- `dockq`: internal DockQ implementation:

  ```text
  DockQ = (Fnat + 1 / (1 + (iRMSD / 1.5)^2) + 1 / (1 + (LRMSD / 8.5)^2)) / 3
  ```

- `lddt_ca`: Cα-based lDDT using native Cα pairs within `15.0 Å` and thresholds `0.5`, `1.0`, `2.0`, and `4.0 Å`.
- `clash_count`: approximate heavy-atom clash count using element-based van der Waals radii and simple bonded-neighbor exclusions.
- `clashes_per_1000_atoms`: `1000 * clash_count / num_heavy_atoms`.

## Aggregation

- `summary_top1.json` summarizes the top-ranked sample per target.
- `summary_best_of_k.json` summarizes the best sample per target among rows with `rank <= topk`.
- `per_target_best_of_k.csv` stores the selected best-of-k row per target.
- By default, summaries exclude rows whose `status` is not `success`.

Best-of-k selection uses:

1. highest `dockq`
2. highest `fnat`
3. lowest `irmsd`
4. lowest `rank`

Each JSON summary includes:

- `count`
- per-metric `mean`, `median`, `std`, `min`, `max`
- DockQ success rates for `>= 0.23`, `>= 0.49`, `>= 0.80`
- iRMSD success rate for `<= 2.0`
- average `lddt_ca`
- average `clash_count`
- average `clashes_per_1000_atoms`

## Assumptions and Limitations

- This first version supports binary complex evaluation only: receptor vs ligand.
- Each side may contain multiple chains, but true multimer interface assignment beyond a binary split is out of scope.
- If predicted and native chain identifiers differ, chains are matched by manifest order unless the chain ID sets are identical. Multi-chain positional remapping is flagged as low-confidence.
- Residue matching first uses `(chain mapping, residue number, insertion code, residue name)`. Sequence fallback is optional and may reduce effective coverage if the sequences are not cleanly alignable.
- Alternate locations are reduced to the highest-occupancy altloc per atom name.
- Hydrogens are ignored by default.
- Missing atoms and unmatched residues reduce coverage metrics and may trigger `CA` fallback in backbone-based metrics.
- The internal clash metric is an approximation. It is not an exact MolProbity clashscore and should not be interpreted as one.
- DockQ validation is optional and currently limited to one-chain-per-side comparisons unless you extend the wrapper for more complex chain-group mappings.
- External tools such as DockQ or MolProbity are not required and are not used by default.

## Recommended Filtering For Papers

Before using any row in a figure, table, or benchmark summary, check at minimum:

- `status == success`
- `matched_residue_fraction` is high enough for your benchmark
- `receptor_matched_residue_fraction` and `ligand_matched_residue_fraction` are both acceptable
- `receptor_min_chain_sequence_identity` and `ligand_min_chain_sequence_identity` are acceptable when sequence fallback was used
- `parse_warning` and `mapping_low_confidence_reasons` are empty

A conservative starting filter is:

```text
status == success
matched_residue_fraction >= 0.90
receptor_matched_residue_fraction >= 0.90
ligand_matched_residue_fraction >= 0.90
```

If DockQ is available, also inspect `validation_dockq_summary.json` and the per-sample diff file before using the evaluator as a benchmark source in a paper.

## Additional Documentation

- `docs/engineering_report.md`

## Running Tests

```bash
cd protein_evaluation
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License. See `LICENSE`.
