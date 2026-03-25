# Protein_complex_eval

`complex_eval` is a small Python toolkit for evaluating predicted binary protein complexes against ground-truth structures. It supports PDB and mmCIF input, multi-chain receptor/ligand partners, per-sample metrics, and top-1 / best-of-k aggregation.

## Features

- Binary complex evaluation with a receptor/ligand split, where each side may contain one or more chains.
- PDB and mmCIF parsing through `gemmi`.
- Pure-Python implementations of Kabsch alignment, RMSD metrics, interface residue detection, Fnat, DockQ, lDDT-Cα, and an approximate internal clash metric.
- Optional sequence-based residue matching fallback when residue numbering does not line up well.
- Parallel evaluation over a manifest CSV with structured failure reporting.

## Installation

```bash
cd protein_evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required.

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

Enable sequence-based residue matching fallback explicitly:

```bash
python -m complex_eval.cli \
  --manifest manifest.csv \
  --out_dir results \
  --workers 4 \
  --sequence_fallback
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

### Per-sample CSV columns

Representative columns include:

- `sample_id`
- `target_id`
- `rank`
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
- If predicted and native chain identifiers differ, chains are matched by manifest order unless the chain ID sets are identical.
- Residue matching first uses `(chain mapping, residue number, insertion code, residue name)`. Sequence fallback is optional and may reduce effective coverage if the sequences are not cleanly alignable.
- Alternate locations are reduced to the highest-occupancy altloc per atom name.
- Hydrogens are ignored by default.
- Missing atoms and unmatched residues reduce coverage metrics and may trigger `CA` fallback in backbone-based metrics.
- The internal clash metric is an approximation. It is not an exact MolProbity clashscore and should not be interpreted as one.
- External tools such as DockQ or MolProbity are not required and are not used by default.

## Running Tests

```bash
cd protein_evaluation
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License. See `LICENSE`.
