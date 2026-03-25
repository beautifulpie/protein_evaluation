# Benchmark Visualization Example

This directory contains a checked-in example of the lightweight visualization bundle produced by `complex-eval`.

The current bundle was generated from 12 deterministic toy protein-complex fixtures. They are intentionally small and synthetic so that the example stays lightweight and reproducible inside the repository.

Files:

- `report.html`
- `per_sample_metrics.csv`
- `per_sample_diagnostics.jsonl`
- `summary_diagnostics.json`
- `plots/performance_snapshot.svg`
- `plots/method_mean_dockq.svg`
- `plots/status_counts.svg`
- `plots/confidence_label_counts.svg`
- `plots/diagnostic_tag_counts.svg`
- `plots/mapping_confidence_vs_dockq.svg`
- `plots/interface_precision_vs_recall.svg`
- `source/manifest.csv`
- `source/data/*.pdb`

The example input set contains:

- 12 target complexes
- a mix of perfect, rigid-transform, interface-shift, numbering-offset, chain-name-mismatch, sparse-atom, and combined-diagnostic cases
- method labels to exercise method-stratified summaries

This example is for repository demonstration and reviewer inspection. It is not intended to represent a validated external benchmark.

To regenerate a fresh local copy:

```bash
complex-eval \
  --manifest examples/benchmark_visualization/source/manifest.csv \
  --out_dir examples/benchmark_visualization \
  --sequence_fallback \
  --no-strict_mapping
```
