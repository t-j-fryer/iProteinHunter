#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory


DEFAULT_PREDICTORS = ["boltz", "intellifold", "openfold3"]
DEFAULT_COLORS = ["#efe4dc", "#dfb8b0", "#d98eaa", "#d0619e", "#af2a98", "#7a0c9f"]
PREDICTOR_DISPLAY = {
    "boltz": "Boltz-2",
    "intellifold": "IntelliFold",
    "openfold3": "OpenFold-3 (preview)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot unconditional helix-kill runs as cycle-colored boxplots grouped by "
            "helix-kill condition and design model."
        )
    )
    p.add_argument(
        "--root",
        required=True,
        help=(
            "Campaign root with predictor folders, e.g. "
            "output/test_helixkill_6cond_12runs_allpredictors_YYYYmmdd_HHMMSS"
        ),
    )
    p.add_argument(
        "--metric",
        default="complex_plddt",
        help="Metric column to plot from comparison_scores_long.csv (default: complex_plddt).",
    )
    p.add_argument(
        "--predictors",
        default=",".join(DEFAULT_PREDICTORS),
        help="Comma-separated predictor order (default: boltz,intellifold,openfold3).",
    )
    p.add_argument(
        "--stage",
        default="design",
        help="Stage filter from comparison_scores_long.csv (default: design).",
    )
    p.add_argument(
        "--out-prefix",
        default="unconditional_helixkill_model_cycle",
        help="Output filename prefix (default: unconditional_helixkill_model_cycle).",
    )
    return p.parse_args()


def parse_case(case_name: str) -> tuple[int, float]:
    if case_name == "unconditional_nohelixkill":
        return 0, 0.0
    m = re.search(r"nhc(\d+)$", case_name)
    if not m:
        return 1, math.nan
    return 1, int(m.group(1)) / 100.0


def case_sort_key(case_name: str) -> tuple[float, str]:
    hk, nhc = parse_case(case_name)
    if hk == 0:
        return (-1.0, case_name)
    return (nhc, case_name)


def case_label(case_name: str) -> str:
    hk, nhc = parse_case(case_name)
    if hk == 0:
        return "No Helix Kill"
    if not math.isfinite(nhc):
        return case_name
    return f"HKC {nhc:.1f}"


def parse_cycle(value: str) -> int | None:
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None


def parse_float(value: str) -> float | None:
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def discover_cases(root: Path, predictors: list[str]) -> list[str]:
    seen: set[str] = set()
    for pred in predictors:
        pred_dir = root / pred
        if not pred_dir.exists():
            continue
        for p in pred_dir.iterdir():
            if p.is_dir() and p.name.startswith("unconditional_"):
                seen.add(p.name)
    return sorted(seen, key=case_sort_key)


def load_metric_rows(
    root: Path,
    predictors: list[str],
    cases: list[str],
    stage: str,
    metric_col: str,
) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    cycles_seen: set[int] = set()

    for pred in predictors:
        for case in cases:
            csv_path = root / pred / case / "comparison_scores_long.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("stage", "") != stage:
                        continue
                    cyc = parse_cycle(row.get("cycle", ""))
                    if cyc is None:
                        continue
                    val = parse_float(row.get(metric_col, ""))
                    if val is None:
                        continue
                    # OpenFold stores pLDDT on a 0-100 scale while other predictors here use 0-1.
                    if pred == "openfold3" and metric_col in {"complex_plddt", "plddt"}:
                        val = val / 100.0
                    run_i = parse_cycle(row.get("run", ""))
                    rows.append(
                        {
                            "case": case,
                            "predictor": pred,
                            "run": run_i if run_i is not None else -1,
                            "cycle": cyc,
                            "metric": val,
                        }
                    )
                    cycles_seen.add(cyc)
    return rows, sorted(cycles_seen)


def build_plot_positions(cases: list[str], predictors: list[str], cycles: list[int]) -> tuple[dict, dict, dict]:
    pos: dict[tuple[str, str, int], float] = {}
    case_centers: dict[str, float] = {}
    model_centers: dict[tuple[str, str], float] = {}

    cycle_step = 0.24
    pred_gap = 0.50
    case_gap = 1.25

    x = 0.0
    for case in cases:
        case_start = x
        for pred in predictors:
            pred_start = x
            for i, cyc in enumerate(cycles):
                pos[(case, pred, cyc)] = pred_start + i * cycle_step
            model_centers[(case, pred)] = pred_start + ((len(cycles) - 1) * cycle_step) / 2.0
            x = pred_start + len(cycles) * cycle_step + pred_gap
        case_end = x - pred_gap
        case_centers[case] = (case_start + case_end) / 2.0
        x = case_end + case_gap

    return pos, case_centers, model_centers


def write_long_csv(rows: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "predictor", "run", "cycle", "metric"])
        w.writeheader()
        for r in sorted(rows, key=lambda z: (case_sort_key(z["case"]), z["predictor"], z["cycle"], z["run"])):
            w.writerow(r)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    predictors = [x.strip() for x in args.predictors.split(",") if x.strip()]
    if not predictors:
        raise SystemExit("No predictors specified.")

    cases = discover_cases(root, predictors)
    if not cases:
        raise SystemExit(f"No unconditional_* case directories found under {root}")

    rows, cycles = load_metric_rows(root, predictors, cases, args.stage, args.metric)
    if not rows:
        raise SystemExit(
            f"No usable rows found for stage={args.stage!r}, metric={args.metric!r} in comparison_scores_long.csv files."
        )

    pos_map, case_centers, model_centers = build_plot_positions(cases, predictors, cycles)
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["case"], r["predictor"], r["cycle"])].append(r["metric"])

    out_prefix = root / args.out_prefix
    out_long_csv = root / f"{args.out_prefix}_long.csv"
    write_long_csv(rows, out_long_csv)

    colors = DEFAULT_COLORS[: len(cycles)]
    if len(colors) < len(cycles):
        cmap = plt.get_cmap("magma")
        colors = [cmap(i / max(1, len(cycles) - 1)) for i in range(len(cycles))]

    fig_w = max(18.0, 2.8 * len(cases))
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))

    for case in cases:
        for pred in predictors:
            for i, cyc in enumerate(cycles):
                vals = grouped.get((case, pred, cyc), [])
                if not vals:
                    continue
                x = pos_map[(case, pred, cyc)]
                bp = ax.boxplot(
                    vals,
                    positions=[x],
                    widths=0.18,
                    patch_artist=True,
                    showfliers=False,
                    zorder=2,
                )
                for b in bp["boxes"]:
                    b.set(facecolor=colors[i], edgecolor="black", linewidth=1.2)
                for key in ("whiskers", "caps", "medians"):
                    for item in bp[key]:
                        item.set(color="black", linewidth=1.2)

    # Condition separators.
    cond_centers = [case_centers[c] for c in cases]
    for i in range(len(cond_centers) - 1):
        mid = 0.5 * (cond_centers[i] + cond_centers[i + 1])
        ax.axvline(mid, color="#bdbdbd", linewidth=1.0, zorder=1)

    ax.set_xticks(cond_centers)
    ax.set_xticklabels([case_label(c) for c in cases], rotation=0)
    ax.set_xlabel("Helix-kill condition")
    if args.metric in {"complex_plddt", "plddt"}:
        ax.set_ylabel("pLDDT")
    else:
        ax.set_ylabel(args.metric.replace("_", " "))
    ax.set_title(f"{args.metric.replace('_', ' ')} by helix-kill condition, model, and cycle")
    ax.grid(False)
    ax.set_axisbelow(True)

    # Model labels repeated within each condition block, angled and placed below x-axis.
    xaxis_text = blended_transform_factory(ax.transData, ax.transAxes)
    for case in cases:
        for pred in predictors:
            ax.text(
                model_centers[(case, pred)],
                -0.16,
                PREDICTOR_DISPLAY.get(pred, pred),
                transform=xaxis_text,
                rotation=35,
                ha="right",
                va="top",
                fontsize=8,
            )

    legend_handles = [Patch(facecolor=colors[i], edgecolor="black", label=f"Cycle {cyc}") for i, cyc in enumerate(cycles)]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1,
    )

    fig.subplots_adjust(bottom=0.26, right=0.83)
    out_svg = root / f"{args.out_prefix}.svg"
    out_png = root / f"{args.out_prefix}.png"
    fig.savefig(out_svg, format="svg", dpi=300)
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)

    print(f"Wrote: {out_svg}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_long_csv}")


if __name__ == "__main__":
    main()
