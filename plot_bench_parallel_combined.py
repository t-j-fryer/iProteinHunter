#!/usr/bin/env python3
"""Generate combined normalized benchmark SVGs across predictors.

This script recreates:
- parallel_vs_time_per_run_normalized_combined.svg
- parallel_vs_time_per_cycle_normalized_combined.svg

It also supports optional horizontal dashed baseline lines.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


def _load_run_means(summary_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = int(row["parallel"])
            mean_sec = float(row["mean_sec"])
            rows.append({"parallel": float(p), "mean_per_parallel": mean_sec / p})
    rows.sort(key=lambda r: r["parallel"])
    return rows


def _load_cycle_means(cycles_csv: Path) -> list[dict[str, float]]:
    by_parallel: dict[int, list[float]] = defaultdict(list)
    with cycles_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_parallel[int(row["parallel"])].append(float(row["duration_sec"]))

    rows: list[dict[str, float]] = []
    for p in sorted(by_parallel):
        vals = by_parallel[p]
        mean_sec = sum(vals) / len(vals)
        rows.append({"parallel": float(p), "mean_per_parallel": mean_sec / p})
    return rows


def _nice_tick_step(y_max: float, n_ticks: int = 5) -> float:
    raw = max(1e-9, y_max / n_ticks)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1:
        nice = 1
    elif norm <= 2:
        nice = 2
    elif norm <= 5:
        nice = 5
    else:
        nice = 10
    return nice * mag


def write_svg(
    series_rows: list[tuple[str, str, list[dict[str, float]]]],
    out_path: Path,
    title: str,
    y_label: str,
    baseline_y: float | None = None,
    baseline_label: str | None = None,
) -> None:
    width, height = 900, 540
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    x_vals: list[float] = []
    y_vals: list[float] = []
    for _, _, rows in series_rows:
        x_vals.extend(r["parallel"] for r in rows)
        y_vals.extend(r["mean_per_parallel"] for r in rows)
    if baseline_y is not None:
        y_vals.append(baseline_y)

    x_min, x_max = min(x_vals), max(x_vals)
    y_min = 0.0
    y_max = max(y_vals) * 1.1
    x_span = max(1.0, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    def x_to_px(x: float) -> float:
        return margin + (x - x_min) / x_span * plot_w

    def y_to_px(y: float) -> float:
        return margin + plot_h - (y - y_min) / y_span * plot_h

    x_ticks = [int(x) for x in range(int(x_min), int(x_max) + 1)]
    y_step = _nice_tick_step(y_max)
    y_ticks = [0.0]
    y = y_step
    while y <= y_max + 1e-9:
        y_ticks.append(y)
        y += y_step

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append("<style>")
    parts.append("  .axis { stroke: #111; stroke-width: 2; }")
    parts.append("  .label { font-family: Helvetica, Arial, sans-serif; font-size: 14px; fill: #111; }")
    parts.append("  .title { font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 600; fill: #111; }")
    parts.append("  .line { fill: none; stroke-width: 3; }")
    parts.append("  .point { stroke-width: 1; }")
    parts.append("  .baseline { fill: none; stroke: #444; stroke-width: 2.5; stroke-dasharray: 10,8; }")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    # Axes
    parts.append(f'<line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" />')
    parts.append(
        f'<line class="axis" x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" />'
    )

    # Ticks
    for y_tick in y_ticks:
        y_px = y_to_px(y_tick)
        parts.append(f'<text class="label" x="{margin - 10}" y="{y_px + 5}" text-anchor="end">{y_tick:g}</text>')
        parts.append(f'<line class="axis" x1="{margin - 5}" y1="{y_px}" x2="{margin}" y2="{y_px}" />')
    for x_tick in x_ticks:
        x_px = x_to_px(float(x_tick))
        parts.append(f'<text class="label" x="{x_px}" y="{height - margin + 22}" text-anchor="middle">{x_tick}</text>')
        parts.append(f'<line class="axis" x1="{x_px}" y1="{height - margin}" x2="{x_px}" y2="{height - margin + 5}" />')

    # Optional baseline
    if baseline_y is not None:
        y_px = y_to_px(baseline_y)
        parts.append(f'<path class="baseline" d="M {margin} {y_px:.2f} L {width - margin} {y_px:.2f}" />')

    # Series
    for label, color, rows in series_rows:
        path_bits = []
        for i, row in enumerate(rows):
            x_px = x_to_px(row["parallel"])
            y_px = y_to_px(row["mean_per_parallel"])
            path_bits.append(f'{"M" if i == 0 else "L"} {x_px:.2f} {y_px:.2f}')
        parts.append(f'<path class="line" d="{" ".join(path_bits)}" stroke="{color}" />')
        for row in rows:
            x_px = x_to_px(row["parallel"])
            y_px = y_to_px(row["mean_per_parallel"])
            parts.append(f'<circle class="point" cx="{x_px:.2f}" cy="{y_px:.2f}" r="5" fill="{color}" stroke="#0d3d66" />')

    # Legend
    legend_x = 620
    legend_y = 60
    legend_step = 20
    for i, (label, color, _) in enumerate(series_rows):
        y = legend_y + i * legend_step
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}" />')
        parts.append(f'<text class="label" x="{legend_x + 20}" y="{y + 12}" text-anchor="start">{label}</text>')

    if baseline_y is not None and baseline_label:
        y = legend_y + len(series_rows) * legend_step + 2
        parts.append(
            f'<path class="baseline" d="M {legend_x} {y + 7:.2f} L {legend_x + 14} {y + 7:.2f}" />'
        )
        parts.append(f'<text class="label" x="{legend_x + 20}" y="{y + 12}" text-anchor="start">{baseline_label}</text>')

    parts.append(f'<text class="title" x="{width / 2}" y="40" text-anchor="middle">{title}</text>')
    parts.append(f'<text class="label" x="{width / 2}" y="520" text-anchor="middle">Parallel Runs</text>')
    parts.append(f'<text class="label" transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle">{y_label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def write_svg_logy(
    series_rows: list[tuple[str, str, list[dict[str, float]]]],
    out_path: Path,
    title: str,
    y_label: str,
    log_base: float,
    baseline_y: float | None = None,
    baseline_label: str | None = None,
) -> None:
    width, height = 900, 540
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    x_vals: list[float] = []
    y_vals: list[float] = []
    for _, _, rows in series_rows:
        x_vals.extend(r["parallel"] for r in rows)
        y_vals.extend(r["mean_per_parallel"] for r in rows)
    if baseline_y is not None:
        y_vals.append(baseline_y)
    if any(y <= 0 for y in y_vals):
        raise ValueError("Log-scale plotting requires positive y-values.")

    x_min, x_max = min(x_vals), max(x_vals)
    x_span = max(1.0, x_max - x_min)

    y_logs = [math.log(y, log_base) for y in y_vals]
    y_min_log = min(y_logs)
    y_max_log = max(y_logs)
    y_min_exp = math.floor(y_min_log)
    y_max_exp = math.ceil(y_max_log)
    y_span_log = max(1e-9, y_max_exp - y_min_exp)

    def x_to_px(x: float) -> float:
        return margin + (x - x_min) / x_span * plot_w

    def y_to_px(y: float) -> float:
        y_log = math.log(y, log_base)
        return margin + plot_h - (y_log - y_min_exp) / y_span_log * plot_h

    x_ticks = [int(x) for x in range(int(x_min), int(x_max) + 1)]
    y_tick_vals = [log_base**e for e in range(y_min_exp, y_max_exp + 1)]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append("<style>")
    parts.append("  .axis { stroke: #111; stroke-width: 2; }")
    parts.append("  .label { font-family: Helvetica, Arial, sans-serif; font-size: 14px; fill: #111; }")
    parts.append("  .title { font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 600; fill: #111; }")
    parts.append("  .line { fill: none; stroke-width: 3; }")
    parts.append("  .point { stroke-width: 1; }")
    parts.append("  .baseline { fill: none; stroke: #444; stroke-width: 2.5; stroke-dasharray: 10,8; }")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    parts.append(f'<line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" />')
    parts.append(
        f'<line class="axis" x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" />'
    )

    for y_tick in y_tick_vals:
        y_px = y_to_px(y_tick)
        parts.append(f'<text class="label" x="{margin - 10}" y="{y_px + 5}" text-anchor="end">{y_tick:g}</text>')
        parts.append(f'<line class="axis" x1="{margin - 5}" y1="{y_px}" x2="{margin}" y2="{y_px}" />')
    for x_tick in x_ticks:
        x_px = x_to_px(float(x_tick))
        parts.append(f'<text class="label" x="{x_px}" y="{height - margin + 22}" text-anchor="middle">{x_tick}</text>')
        parts.append(f'<line class="axis" x1="{x_px}" y1="{height - margin}" x2="{x_px}" y2="{height - margin + 5}" />')

    if baseline_y is not None:
        y_px = y_to_px(baseline_y)
        parts.append(f'<path class="baseline" d="M {margin} {y_px:.2f} L {width - margin} {y_px:.2f}" />')

    for label, color, rows in series_rows:
        path_bits = []
        for i, row in enumerate(rows):
            x_px = x_to_px(row["parallel"])
            y_px = y_to_px(row["mean_per_parallel"])
            path_bits.append(f'{"M" if i == 0 else "L"} {x_px:.2f} {y_px:.2f}')
        parts.append(f'<path class="line" d="{" ".join(path_bits)}" stroke="{color}" />')
        for row in rows:
            x_px = x_to_px(row["parallel"])
            y_px = y_to_px(row["mean_per_parallel"])
            parts.append(f'<circle class="point" cx="{x_px:.2f}" cy="{y_px:.2f}" r="5" fill="{color}" stroke="#0d3d66" />')

    legend_x = 620
    legend_y = 60
    legend_step = 20
    for i, (label, color, _) in enumerate(series_rows):
        y = legend_y + i * legend_step
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}" />')
        parts.append(f'<text class="label" x="{legend_x + 20}" y="{y + 12}" text-anchor="start">{label}</text>')

    if baseline_y is not None and baseline_label:
        y = legend_y + len(series_rows) * legend_step + 2
        parts.append(
            f'<path class="baseline" d="M {legend_x} {y + 7:.2f} L {legend_x + 14} {y + 7:.2f}" />'
        )
        parts.append(f'<text class="label" x="{legend_x + 20}" y="{y + 12}" text-anchor="start">{baseline_label}</text>')

    parts.append(f'<text class="title" x="{width / 2}" y="40" text-anchor="middle">{title}</text>')
    parts.append(f'<text class="label" x="{width / 2}" y="520" text-anchor="middle">Parallel Runs</text>')
    parts.append(f'<text class="label" transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle">{y_label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main() -> None:
    root = Path("/Users/thomasfryer/iProteinHunter/output")
    out_dir = root / "bench_parallel_combined_20260217_090819"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (
            "OpenFold3",
            "#2ca02c",
            root / "bench_parallel_openfold3_varruns_20260216_114714",
        ),
        (
            "IntelliFold",
            "#1f77b4",
            root / "bench_parallel_intellifold_varruns_20260216_170338",
        ),
        (
            "Boltz",
            "#d62728",
            root / "bench_parallel_boltz_varruns_20260216_192453",
        ),
        (
            "Boltz (CPU)",
            "#ff7f0e",
            root / "bench_parallel_boltz_cpu_varruns_20260224_190204",
        ),
    ]

    run_series: list[tuple[str, str, list[dict[str, float]]]] = []
    cycle_series: list[tuple[str, str, list[dict[str, float]]]] = []
    for label, color, ds_dir in datasets:
        run_rows = _load_run_means(ds_dir / "timing_summary.csv")
        cycle_rows = _load_cycle_means(ds_dir / "timing_cycles_long.csv")
        run_series.append((label, color, run_rows))
        cycle_series.append((label, color, cycle_rows))

    write_svg(
        series_rows=run_series,
        out_path=out_dir / "parallel_vs_time_per_run_normalized_combined.svg",
        title="Normalized Mean Time per Run (All Predictors)",
        y_label="Mean Time per Run / Parallel (sec)",
        baseline_y=43.0,
        baseline_label="Boltz-CUDA-5090 (43 s/run)",
    )
    write_svg(
        series_rows=cycle_series,
        out_path=out_dir / "parallel_vs_time_per_cycle_normalized_combined.svg",
        title="Normalized Mean Time per Cycle (All Predictors)",
        y_label="Mean Time per Cycle / Parallel (sec)",
        baseline_y=7.1667,
        baseline_label="Boltz-CUDA-5090 (7.1667 s/cycle)",
    )

    write_svg_logy(
        series_rows=run_series,
        out_path=out_dir / "parallel_vs_time_per_run_normalized_combined_log10.svg",
        title="Normalized Mean Time per Run (All Predictors, log10 y-axis)",
        y_label="Mean Time per Run / Parallel (sec, log10 scale)",
        log_base=10.0,
        baseline_y=43.0,
        baseline_label="Boltz-CUDA-5090 (43 s/run)",
    )
    write_svg_logy(
        series_rows=cycle_series,
        out_path=out_dir / "parallel_vs_time_per_cycle_normalized_combined_log10.svg",
        title="Normalized Mean Time per Cycle (All Predictors, log10 y-axis)",
        y_label="Mean Time per Cycle / Parallel (sec, log10 scale)",
        log_base=10.0,
        baseline_y=7.1667,
        baseline_label="Boltz-CUDA-5090 (7.1667 s/cycle)",
    )
    write_svg_logy(
        series_rows=run_series,
        out_path=out_dir / "parallel_vs_time_per_run_normalized_combined_log2.svg",
        title="Normalized Mean Time per Run (All Predictors, log2 y-axis)",
        y_label="Mean Time per Run / Parallel (sec, log2 scale)",
        log_base=2.0,
        baseline_y=43.0,
        baseline_label="Boltz-CUDA-5090 (43 s/run)",
    )
    write_svg_logy(
        series_rows=cycle_series,
        out_path=out_dir / "parallel_vs_time_per_cycle_normalized_combined_log2.svg",
        title="Normalized Mean Time per Cycle (All Predictors, log2 y-axis)",
        y_label="Mean Time per Cycle / Parallel (sec, log2 scale)",
        log_base=2.0,
        baseline_y=7.1667,
        baseline_label="Boltz-CUDA-5090 (7.1667 s/cycle)",
    )

    print(f"Wrote: {out_dir / 'parallel_vs_time_per_run_normalized_combined.svg'}")
    print(f"Wrote: {out_dir / 'parallel_vs_time_per_cycle_normalized_combined.svg'}")
    print(f"Wrote: {out_dir / 'parallel_vs_time_per_run_normalized_combined_log10.svg'}")
    print(f"Wrote: {out_dir / 'parallel_vs_time_per_cycle_normalized_combined_log10.svg'}")
    print(f"Wrote: {out_dir / 'parallel_vs_time_per_run_normalized_combined_log2.svg'}")
    print(f"Wrote: {out_dir / 'parallel_vs_time_per_cycle_normalized_combined_log2.svg'}")


if __name__ == "__main__":
    main()
