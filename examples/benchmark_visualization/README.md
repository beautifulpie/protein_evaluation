# Benchmark Visualization Example

This directory contains a checked-in example of the lightweight visualization bundle produced by `complex-eval`.

Files:

- `report.html`
- `plots/status_counts.svg`
- `plots/confidence_label_counts.svg`
- `plots/diagnostic_tag_counts.svg`
- `plots/mapping_confidence_vs_dockq.svg`
- `plots/interface_precision_vs_recall.svg`

The example was generated from the deterministic benchmark fixture under `tests/fixtures/benchmark/`.

To regenerate a fresh local copy:

```bash
complex-eval \
  --manifest tests/fixtures/benchmark/manifest.csv \
  --out_dir results
```
