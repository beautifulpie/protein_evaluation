"""Lightweight deterministic visualization outputs for evaluation results."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

METRIC_PLOT_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    ("ca_rmsd", "Global C-alpha RMSD", "Whole-complex C-alpha alignment error; lower is better.", False),
    ("all_atom_rmsd", "All-heavy-atom RMSD", "Matched heavy-atom RMSD after rigid alignment; lower is better.", False),
    ("irmsd", "iRMSD", "Interface backbone RMSD on native interface residues; lower is better.", False),
    ("lrmsd", "LRMSD", "Ligand RMSD after receptor superposition; lower is better.", False),
    ("fnat", "Fnat", "Recovered native contact fraction; higher is better.", True),
    ("dockq", "DockQ", "Internal DockQ implementation; higher is better.", True),
    ("lddt_ca", "lDDT-Ca", "C-alpha local distance difference test; higher is better.", True),
    ("clash_count", "Clash Count", "Approximate heavy-atom steric clashes; lower is better.", False),
    (
        "clashes_per_1000_atoms",
        "Clashes Per 1000 Atoms",
        "Approximate clash density normalized by atom count; lower is better.",
        False,
    ),
    ("interface_precision", "Interface Precision", "Fraction of predicted contacts that are native; higher is better.", True),
    ("interface_recall", "Interface Recall", "Fraction of native contacts recovered; higher is better.", True),
    ("interface_f1", "Interface F1", "Harmonic mean of interface precision and recall; higher is better.", True),
)


def write_visualization_outputs(
    all_rows_df: pd.DataFrame,
    summary_diagnostics: dict[str, object],
    out_dir: str | Path,
) -> None:
    """Write a lightweight HTML report and SVG plots.

    The visualization layer intentionally avoids heavyweight plotting
    dependencies. Outputs are static, deterministic, and reviewable in git
    artifacts or benchmark folders.
    """

    out_path = Path(out_dir)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    status_counts = _value_counts(all_rows_df, "status", default_value="success")
    confidence_counts = _value_counts(all_rows_df, "mapping_confidence_label")
    tag_counts = _tag_counts(all_rows_df)
    performance_snapshot = _performance_snapshot(summary_diagnostics)
    method_mean_dockq = _method_metric_map(summary_diagnostics, metric="mean_dockq")

    (plots_dir / "status_counts.svg").write_text(
        _bar_chart_svg(
            status_counts,
            title="Row Status Counts",
            subtitle="All evaluated rows",
            color="#2f5aa8",
        ),
        encoding="utf-8",
    )
    (plots_dir / "confidence_label_counts.svg").write_text(
        _bar_chart_svg(
            confidence_counts,
            title="Mapping Confidence Labels",
            subtitle="All evaluated rows",
            color="#3f8f5f",
        ),
        encoding="utf-8",
    )
    (plots_dir / "diagnostic_tag_counts.svg").write_text(
        _bar_chart_svg(
            tag_counts.head(10).to_dict(),
            title="Top Diagnostic Tags",
            subtitle="Most frequent explainability tags",
            color="#a85f2f",
        ),
        encoding="utf-8",
    )
    (plots_dir / "performance_snapshot.svg").write_text(
        _bar_chart_svg(
            performance_snapshot,
            title="Prediction Performance Snapshot",
            subtitle="Overall benchmark summary",
            color="#6b46c1",
        ),
        encoding="utf-8",
    )
    (plots_dir / "method_mean_dockq.svg").write_text(
        _bar_chart_svg(
            method_mean_dockq,
            title="Method Mean DockQ",
            subtitle="Per-sample method summary" if method_mean_dockq else "No method-stratified summary available.",
            color="#0f766e",
        ),
        encoding="utf-8",
    )
    (plots_dir / "mapping_confidence_vs_dockq.svg").write_text(
        _scatter_svg(
            frame=all_rows_df,
            x_column="mapping_confidence_score",
            y_column="dockq",
            title="Mapping Confidence vs DockQ",
            x_label="mapping_confidence_score",
            y_label="dockq",
            color_column="status",
        ),
        encoding="utf-8",
    )
    (plots_dir / "interface_precision_vs_recall.svg").write_text(
        _scatter_svg(
            frame=all_rows_df,
            x_column="interface_recall",
            y_column="interface_precision",
            title="Interface Recall vs Precision",
            x_label="interface_recall",
            y_label="interface_precision",
            color_column="mapping_confidence_label",
        ),
        encoding="utf-8",
    )
    for metric_column, title, subtitle, higher_is_better in METRIC_PLOT_SPECS:
        (plots_dir / f"metric_{metric_column}.svg").write_text(
            _metric_distribution_svg(
                frame=all_rows_df,
                metric_column=metric_column,
                title=title,
                subtitle=subtitle,
                higher_is_better=higher_is_better,
                color_column="status",
            ),
            encoding="utf-8",
        )

    report_html = _build_html_report(all_rows_df, summary_diagnostics)
    (out_path / "report.html").write_text(report_html, encoding="utf-8")


def _build_html_report(all_rows_df: pd.DataFrame, summary_diagnostics: dict[str, object]) -> str:
    """Build a static HTML visualization report."""

    overall = summary_diagnostics.get("overall", {})
    best_of_k = overall.get("best_of_k", {}) if isinstance(overall, dict) else {}
    top1 = overall.get("top1", {}) if isinstance(overall, dict) else {}
    per_sample = overall.get("per_sample", {}) if isinstance(overall, dict) else {}
    low_confidence_rows = _low_confidence_table(all_rows_df)
    top_targets = _top_target_table(summary_diagnostics)
    performance_table = _performance_overview_table(summary_diagnostics)
    method_table = _method_performance_table(summary_diagnostics)
    top_samples = _top_sample_table(all_rows_df)
    metric_gallery = _metric_gallery_html()

    cards = [
        _metric_card("Per-sample mean DockQ", _fmt(per_sample.get("mean_dockq"))),
        _metric_card("Top-1 mean DockQ", _fmt(top1.get("mean_dockq"))),
        _metric_card("Best-of-k mean DockQ", _fmt(best_of_k.get("mean_dockq"))),
        _metric_card("Top-1 DockQ>=0.23", _fmt(top1.get("success_rate_dockq_ge_0_23"))),
        _metric_card("Best-of-k DockQ>=0.23", _fmt(best_of_k.get("success_rate_dockq_ge_0_23"))),
        _metric_card("Best-of-k mean Fnat", _fmt(_summary_metric(best_of_k, "mean_fnat", "mean_pairwise_fnat_mean"))),
        _metric_card("Best-of-k mean interface F1", _fmt(best_of_k.get("mean_interface_f1"))),
        _metric_card("Best-of-k mean confidence", _fmt(best_of_k.get("mean_mapping_confidence_score"))),
        _metric_card("Low-confidence fraction", _fmt(best_of_k.get("fraction_low_confidence_mapping"))),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>complex_eval report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #18202a;
      background: #f7f8fb;
    }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    p {{ margin: 8px 0 16px 0; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 24px 0;
    }}
    .card {{
      background: white;
      border: 1px solid #d6dbe4;
      border-radius: 10px;
      padding: 14px;
    }}
    .card-title {{
      font-size: 12px;
      color: #566577;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      margin: 18px 0 24px 0;
    }}
    .panel {{
      background: white;
      border: 1px solid #d6dbe4;
      border-radius: 10px;
      padding: 14px;
    }}
    img {{
      width: 100%;
      border: 1px solid #eef1f5;
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #e5e8ee;
      vertical-align: top;
    }}
    th {{
      background: #f2f4f8;
    }}
    code {{
      background: #eef2f8;
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>complex_eval diagnostic report</h1>
  <p>This report is a lightweight visual audit layer over the evaluator outputs. It is intended for benchmark triage, mapping-quality review, result explainability, and prediction-performance review.</p>

  <div class="cards">
    {''.join(cards)}
  </div>

  <div class="grid">
    <div class="panel">
      <h2>Prediction performance overview</h2>
      <img src="plots/performance_snapshot.svg" alt="Prediction performance overview" />
    </div>
    <div class="panel">
      <h2>Method mean DockQ</h2>
      <img src="plots/method_mean_dockq.svg" alt="Method mean DockQ" />
    </div>
    <div class="panel">
      <h2>Status overview</h2>
      <img src="plots/status_counts.svg" alt="Status counts" />
    </div>
    <div class="panel">
      <h2>Confidence labels</h2>
      <img src="plots/confidence_label_counts.svg" alt="Confidence labels" />
    </div>
    <div class="panel">
      <h2>Mapping confidence vs DockQ</h2>
      <img src="plots/mapping_confidence_vs_dockq.svg" alt="Mapping confidence vs DockQ" />
    </div>
    <div class="panel">
      <h2>Interface recall vs precision</h2>
      <img src="plots/interface_precision_vs_recall.svg" alt="Interface recall vs precision" />
    </div>
    <div class="panel">
      <h2>Diagnostic tag frequency</h2>
      <img src="plots/diagnostic_tag_counts.svg" alt="Diagnostic tag counts" />
    </div>
  </div>

  <div class="panel">
    <h2>Benchmark performance summary</h2>
    {performance_table}
  </div>

  <div class="panel" style="margin-top:18px;">
    <h2>Top predicted samples</h2>
    {top_samples}
  </div>

  <div class="panel" style="margin-top:18px;">
    <h2>Method performance snapshot</h2>
    {method_table}
  </div>

  <div class="panel">
    <h2>Low-confidence or failed rows</h2>
    {low_confidence_rows}
  </div>

  <div class="panel" style="margin-top:18px;">
    <h2>Per-target summary snapshot</h2>
    {top_targets}
  </div>

  <div class="panel" style="margin-top:18px;">
    <h2>Detailed metric gallery</h2>
    <p>These plots expose the per-sample distribution of each core evaluation metric, including structural RMSDs, contact recovery, lDDT-Ca, clash diagnostics, and interface precision/recall/F1.</p>
    <div class="grid">
      {metric_gallery}
    </div>
  </div>
</body>
</html>
"""


def _metric_card(title: str, value: str) -> str:
    """Return a simple HTML metric card."""

    return (
        f'<div class="card"><div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value">{html.escape(value)}</div></div>'
    )


def _low_confidence_table(all_rows_df: pd.DataFrame) -> str:
    """Return an HTML table of the most important problematic rows."""

    if all_rows_df.empty:
        return "<p>No rows available.</p>"
    frame = all_rows_df.copy()
    status = frame["status"].fillna("success") if "status" in frame.columns else pd.Series("success", index=frame.index)
    confidence = pd.to_numeric(frame.get("mapping_confidence_score"), errors="coerce")
    mask = status.ne("success") | confidence.lt(0.85)
    filtered = frame[mask].copy()
    if filtered.empty:
        return "<p>No low-confidence or failed rows.</p>"
    columns = [
        column
        for column in [
            "sample_id",
            "target_id",
            "method",
            "status",
            "mapping_confidence_label",
            "mapping_confidence_score",
            "dockq",
            "fnat",
            "diagnostic_tags",
            "error_message",
        ]
        if column in filtered.columns
    ]
    display = filtered[columns].sort_values(
        by=[column for column in ["status", "mapping_confidence_score", "sample_id"] if column in filtered.columns],
        ascending=[True, True, True][: len([column for column in ["status", "mapping_confidence_score", "sample_id"] if column in filtered.columns])],
    ).head(25)
    return display.to_html(index=False, classes="table", border=0, escape=True)


def _top_target_table(summary_diagnostics: dict[str, object]) -> str:
    """Return a compact per-target summary table."""

    by_target = summary_diagnostics.get("by_target_id", {})
    if not isinstance(by_target, dict) or not by_target:
        return "<p>No target summary available.</p>"
    rows: list[dict[str, object]] = []
    for target_id, metrics in by_target.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "target_id": target_id,
                "count": metrics.get("count"),
                "mean_mapping_confidence_score": metrics.get("mean_mapping_confidence_score"),
                "mean_interface_f1": metrics.get("mean_interface_f1"),
                "mean_dockq": metrics.get("mean_dockq"),
                "success_rate_dockq_ge_0_23": metrics.get("success_rate_dockq_ge_0_23"),
            }
        )
    if not rows:
        return "<p>No target summary available.</p>"
    frame = pd.DataFrame(rows).sort_values(
        by=["mean_dockq", "target_id"],
        ascending=[False, True],
        na_position="last",
    ).head(20)
    return frame.to_html(index=False, classes="table", border=0, escape=True)


def _performance_overview_table(summary_diagnostics: dict[str, object]) -> str:
    """Return a compact benchmark performance table."""

    overall = summary_diagnostics.get("overall", {})
    if not isinstance(overall, dict) or not overall:
        return "<p>No performance summary available.</p>"

    rows: list[dict[str, object]] = []
    for view_name in ("per_sample", "top1", "best_of_k"):
        metrics = overall.get(view_name)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "view": view_name,
                "count": metrics.get("count"),
                "mean_dockq": metrics.get("mean_dockq"),
                "mean_fnat": _summary_metric(metrics, "mean_fnat", "mean_pairwise_fnat_mean"),
                "mean_interface_f1": metrics.get("mean_interface_f1"),
                "mean_mapping_confidence_score": metrics.get("mean_mapping_confidence_score"),
                "success_rate_dockq_ge_0_23": metrics.get("success_rate_dockq_ge_0_23"),
                "success_rate_dockq_ge_0_49": metrics.get("success_rate_dockq_ge_0_49"),
                "success_rate_dockq_ge_0_80": metrics.get("success_rate_dockq_ge_0_80"),
            }
        )
    if not rows:
        return "<p>No performance summary available.</p>"
    return pd.DataFrame(rows).to_html(index=False, classes="table", border=0, escape=True)


def _method_performance_table(summary_diagnostics: dict[str, object]) -> str:
    """Return a compact method-level performance table when available."""

    by_method = summary_diagnostics.get("by_method", {})
    per_sample = by_method.get("per_sample", {}) if isinstance(by_method, dict) else {}
    if not isinstance(per_sample, dict) or not per_sample:
        return "<p>No method-stratified summary available.</p>"

    rows: list[dict[str, object]] = []
    for method_name, metrics in per_sample.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "method": method_name,
                "count": metrics.get("count"),
                "mean_dockq": metrics.get("mean_dockq"),
                "mean_fnat": _summary_metric(metrics, "mean_fnat", "mean_pairwise_fnat_mean"),
                "mean_interface_f1": metrics.get("mean_interface_f1"),
                "mean_mapping_confidence_score": metrics.get("mean_mapping_confidence_score"),
                "success_rate_dockq_ge_0_23": metrics.get("success_rate_dockq_ge_0_23"),
            }
        )
    if not rows:
        return "<p>No method-stratified summary available.</p>"
    frame = pd.DataFrame(rows).sort_values(
        by=["mean_dockq", "method"],
        ascending=[False, True],
        na_position="last",
    )
    return frame.to_html(index=False, classes="table", border=0, escape=True)


def _top_sample_table(all_rows_df: pd.DataFrame) -> str:
    """Return a leaderboard-style sample table by performance."""

    if all_rows_df.empty:
        return "<p>No rows available.</p>"
    frame = all_rows_df.copy()
    for column in ("dockq", "fnat", "interface_f1", "mapping_confidence_score"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    columns = [
        column
        for column in [
            "sample_id",
            "target_id",
            "method",
            "status",
            "dockq",
            "fnat",
            "interface_f1",
            "mapping_confidence_label",
            "mapping_confidence_score",
        ]
        if column in frame.columns
    ]
    sort_columns = [column for column in ["dockq", "fnat", "mapping_confidence_score", "sample_id"] if column in frame.columns]
    ascending = [False, False, False, True][: len(sort_columns)]
    display = frame[columns].sort_values(by=sort_columns, ascending=ascending, na_position="last").head(20)
    return display.to_html(index=False, classes="table", border=0, escape=True)


def _metric_gallery_html() -> str:
    """Return HTML panels for the per-metric SVG gallery."""

    panels: list[str] = []
    for metric_column, title, _, _ in METRIC_PLOT_SPECS:
        filename = f"plots/metric_{metric_column}.svg"
        panels.append(
            '<div class="panel">'
            f"<h2>{html.escape(title)}</h2>"
            f'<img src="{html.escape(filename)}" alt="{html.escape(title)}" />'
            "</div>"
        )
    return "".join(panels)


def _value_counts(frame: pd.DataFrame, column: str, default_value: str = "") -> dict[str, int]:
    """Return stable value counts for a string column."""

    if column not in frame.columns or frame.empty:
        return {}
    values = frame[column].fillna(default_value).astype(str).str.strip()
    if default_value:
        values = values.replace("", default_value)
    counts = values.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items() if str(key) != ""}


def _tag_counts(frame: pd.DataFrame) -> pd.Series:
    """Return counts of semicolon-delimited diagnostic tags."""

    if "diagnostic_tags" not in frame.columns or frame.empty:
        return pd.Series(dtype="int64")
    counts: dict[str, int] = {}
    for text in frame["diagnostic_tags"].fillna("").astype(str):
        for tag in [item.strip() for item in text.split(";") if item.strip()]:
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return pd.Series(dtype="int64")
    return pd.Series(counts).sort_values(ascending=False)


def _bar_chart_svg(data: dict[str, float | int], title: str, subtitle: str, color: str) -> str:
    """Render a simple horizontal bar chart as SVG."""

    width = 720
    bar_height = 28
    top_margin = 70
    left_margin = 170
    chart_width = width - left_margin - 40
    items = list(data.items())
    height = max(180, top_margin + 30 + bar_height * max(1, len(items)))

    if not items:
        return _empty_svg(title, subtitle, "No data available.")

    max_value = max(float(value) for _, value in items) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="22" font-family="sans-serif" fill="#1f2937">{html.escape(title)}</text>',
        f'<text x="24" y="52" font-size="12" font-family="sans-serif" fill="#6b7280">{html.escape(subtitle)}</text>',
    ]

    for index, (label, value) in enumerate(items):
        y = top_margin + index * bar_height
        numeric_value = float(value)
        bar_width = 0 if max_value == 0 else int(chart_width * (numeric_value / max_value))
        parts.append(
            f'<text x="24" y="{y + 18}" font-size="12" font-family="sans-serif" fill="#111827">{html.escape(label)}</text>'
        )
        parts.append(
            f'<rect x="{left_margin}" y="{y + 5}" width="{bar_width}" height="16" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left_margin + bar_width + 8}" y="{y + 18}" font-size="12" font-family="sans-serif" fill="#111827">{html.escape(_fmt(value))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _scatter_svg(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    color_column: str,
) -> str:
    """Render a simple scatter plot as SVG."""

    width = 720
    height = 420
    left = 70
    right = 30
    top = 60
    bottom = 55
    plot_width = width - left - right
    plot_height = height - top - bottom

    if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
        return _empty_svg(title, f"{x_label} vs {y_label}", "No data available.")

    x = pd.to_numeric(frame[x_column], errors="coerce")
    y = pd.to_numeric(frame[y_column], errors="coerce")
    valid = frame.assign(_x=x, _y=y).dropna(subset=["_x", "_y"]).copy()
    if valid.empty:
        return _empty_svg(title, f"{x_label} vs {y_label}", "No plottable points.")

    x_values = valid["_x"].astype(float)
    y_values = valid["_y"].astype(float)
    x_min, x_max = _axis_bounds(x_values.tolist())
    y_min, y_max = _axis_bounds(y_values.tolist())

    palette = {
        "success": "#2563eb",
        "low_confidence_mapping": "#d97706",
        "failed": "#dc2626",
        "high": "#2563eb",
        "medium": "#d97706",
        "low": "#dc2626",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="22" font-family="sans-serif" fill="#1f2937">{html.escape(title)}</text>',
        f'<text x="24" y="50" font-size="12" font-family="sans-serif" fill="#6b7280">{html.escape(x_label)} vs {html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#475569">{html.escape(x_label)}</text>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#475569">{html.escape(y_label)}</text>',
    ]

    for _, row in valid.iterrows():
        x_value = float(row["_x"])
        y_value = float(row["_y"])
        x_pos = left + _scale(x_value, x_min, x_max, plot_width)
        y_pos = top + plot_height - _scale(y_value, y_min, y_max, plot_height)
        color_key = str(row.get(color_column, "")).strip()
        color = palette.get(color_key, "#4b5563")
        label = html.escape(str(row.get("sample_id", "")))
        parts.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="4" fill="{color}" fill-opacity="0.85">'
            f'<title>{label}: {x_label}={x_value:.3f}, {y_label}={y_value:.3f}, {color_column}={color_key}</title>'
            "</circle>"
        )

    parts.append(
        f'<text x="{left}" y="{top + plot_height + 20}" font-size="11" font-family="sans-serif" fill="#64748b">{x_min:.2f}</text>'
    )
    parts.append(
        f'<text x="{left + plot_width}" y="{top + plot_height + 20}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">{x_max:.2f}</text>'
    )
    parts.append(
        f'<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">{y_min:.2f}</text>'
    )
    parts.append(
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">{y_max:.2f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _metric_distribution_svg(
    frame: pd.DataFrame,
    metric_column: str,
    title: str,
    subtitle: str,
    higher_is_better: bool,
    color_column: str,
) -> str:
    """Render a sorted per-sample metric distribution plot as SVG."""

    width = 720
    height = 320
    left = 70
    right = 30
    top = 60
    bottom = 55
    plot_width = width - left - right
    plot_height = height - top - bottom

    if frame.empty or metric_column not in frame.columns:
        return _empty_svg(title, subtitle, "No data available.")

    values = pd.to_numeric(frame[metric_column], errors="coerce")
    valid = frame.assign(_metric_value=values).dropna(subset=["_metric_value"]).copy()
    if valid.empty:
        return _empty_svg(title, subtitle, "No plottable points.")

    valid = valid.sort_values(
        by=["_metric_value", "sample_id"],
        ascending=[not higher_is_better, True],
        na_position="last",
    ).reset_index(drop=True)
    metric_values = valid["_metric_value"].astype(float).tolist()
    y_min, y_max = _axis_bounds(metric_values)
    mean_value = float(pd.Series(metric_values).mean())
    median_value = float(pd.Series(metric_values).median())

    palette = {
        "success": "#2563eb",
        "low_confidence_mapping": "#d97706",
        "failed": "#dc2626",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="22" font-family="sans-serif" fill="#1f2937">{html.escape(title)}</text>',
        f'<text x="24" y="50" font-size="12" font-family="sans-serif" fill="#6b7280">{html.escape(subtitle)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#475569">samples sorted by metric</text>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#475569">{html.escape(metric_column)}</text>',
    ]

    for line_value, label, color in (
        (mean_value, "mean", "#7c3aed"),
        (median_value, "median", "#0f766e"),
    ):
        y_pos = top + plot_height - _scale(line_value, y_min, y_max, plot_height)
        parts.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_width}" y2="{y_pos:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{left + plot_width - 4}" y="{y_pos - 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="{color}">{label}: {line_value:.3f}</text>'
        )

    count = len(valid)
    for index, row in valid.iterrows():
        if count == 1:
            x_pos = left + plot_width / 2.0
        else:
            x_pos = left + (index / (count - 1)) * plot_width
        value = float(row["_metric_value"])
        y_pos = top + plot_height - _scale(value, y_min, y_max, plot_height)
        color_key = str(row.get(color_column, "")).strip()
        color = palette.get(color_key, "#4b5563")
        label = html.escape(str(row.get("sample_id", "")))
        target = html.escape(str(row.get("target_id", "")))
        parts.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="4" fill="{color}" fill-opacity="0.88">'
            f'<title>{label} ({target}): {metric_column}={value:.3f}, {color_column}={color_key}</title>'
            "</circle>"
        )

    parts.append(
        f'<text x="{left}" y="{top + plot_height + 20}" font-size="11" font-family="sans-serif" fill="#64748b">best-ranked sample</text>'
    )
    parts.append(
        f'<text x="{left + plot_width}" y="{top + plot_height + 20}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">worst-ranked sample</text>'
    )
    parts.append(
        f'<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">{y_min:.2f}</text>'
    )
    parts.append(
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#64748b">{y_max:.2f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _empty_svg(title: str, subtitle: str, message: str) -> str:
    """Return an empty-state SVG."""

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="24" y="30" font-size="22" font-family="sans-serif" fill="#1f2937">{html.escape(title)}</text>'
        f'<text x="24" y="52" font-size="12" font-family="sans-serif" fill="#6b7280">{html.escape(subtitle)}</text>'
        f'<text x="24" y="120" font-size="14" font-family="sans-serif" fill="#6b7280">{html.escape(message)}</text>'
        "</svg>"
    )


def _axis_bounds(values: list[float]) -> tuple[float, float]:
    """Return non-degenerate axis bounds."""

    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        padding = 0.5 if math.isclose(minimum, 0.0) else max(abs(minimum) * 0.1, 0.1)
        return minimum - padding, maximum + padding
    return minimum, maximum


def _scale(value: float, low: float, high: float, span: float) -> float:
    """Scale a numeric value into a screen span."""

    if math.isclose(low, high):
        return span / 2.0
    return (value - low) / (high - low) * span


def _fmt(value: object) -> str:
    """Format a summary scalar for HTML cards."""

    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def _performance_snapshot(summary_diagnostics: dict[str, object]) -> dict[str, float]:
    """Return compact overall performance values for the summary plot."""

    overall = summary_diagnostics.get("overall", {})
    if not isinstance(overall, dict):
        return {}
    best_of_k = overall.get("best_of_k", {}) if isinstance(overall.get("best_of_k"), dict) else {}
    top1 = overall.get("top1", {}) if isinstance(overall.get("top1"), dict) else {}
    per_sample = overall.get("per_sample", {}) if isinstance(overall.get("per_sample"), dict) else {}
    return {
        "per_sample mean DockQ": _finite_or_zero(per_sample.get("mean_dockq")),
        "top1 mean DockQ": _finite_or_zero(top1.get("mean_dockq")),
        "best_of_k mean DockQ": _finite_or_zero(best_of_k.get("mean_dockq")),
        "best_of_k DockQ>=0.23": _finite_or_zero(best_of_k.get("success_rate_dockq_ge_0_23")),
        "best_of_k mean Fnat": _finite_or_zero(_summary_metric(best_of_k, "mean_fnat", "mean_pairwise_fnat_mean")),
        "best_of_k mean interface F1": _finite_or_zero(best_of_k.get("mean_interface_f1")),
    }


def _method_metric_map(summary_diagnostics: dict[str, object], metric: str) -> dict[str, float]:
    """Return a method-to-metric mapping for plotting."""

    by_method = summary_diagnostics.get("by_method", {})
    per_sample = by_method.get("per_sample", {}) if isinstance(by_method, dict) else {}
    if not isinstance(per_sample, dict):
        return {}
    items: list[tuple[str, float]] = []
    for method_name, metrics in per_sample.items():
        if not isinstance(metrics, dict):
            continue
        value = _safe_numeric(metrics.get(metric))
        if math.isnan(value):
            continue
        items.append((str(method_name), value))
    items.sort(key=lambda item: (-item[1], item[0]))
    return {name: value for name, value in items}


def _summary_metric(summary: dict[str, object], primary_key: str, fallback_key: str) -> object:
    """Return a summary metric with an explicit fallback."""

    primary_value = summary.get(primary_key)
    primary_numeric = _safe_numeric(primary_value)
    if not math.isnan(primary_numeric):
        return primary_value
    return summary.get(fallback_key)


def _safe_numeric(value: object) -> float:
    """Return a float or NaN from a summary-like scalar."""

    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(numeric):
        return float("nan")
    return numeric


def _finite_or_zero(value: object) -> float:
    """Return a finite float or zero for plotting."""

    numeric = _safe_numeric(value)
    if math.isnan(numeric):
        return 0.0
    return numeric
