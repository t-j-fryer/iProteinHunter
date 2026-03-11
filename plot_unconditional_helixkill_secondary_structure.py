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
PREDICTOR_DISPLAY = {
    "boltz": "Boltz-2",
    "intellifold": "IntelliFold",
    "openfold3": "OpenFold-3 (preview)",
}
SS_TYPES = ["helix", "sheet", "loop"]
SS_DISPLAY = {"helix": "Helix", "sheet": "Sheet", "loop": "Loop"}
# Match cycle 1, cycle 3, cycle 5 colors from the pLDDT cycle plot.
SS_COLORS = {"helix": "#dfb8b0", "sheet": "#d0619e", "loop": "#7a0c9f"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot DSSP secondary-structure distributions grouped by helix-kill condition "
            "and design model."
        )
    )
    p.add_argument(
        "--root",
        required=True,
        help="Campaign root with boltz/intellifold/openfold3 subfolders.",
    )
    p.add_argument(
        "--predictors",
        default=",".join(DEFAULT_PREDICTORS),
        help="Comma-separated predictor order (default: boltz,intellifold,openfold3).",
    )
    p.add_argument(
        "--scope",
        choices=["chain", "all"],
        default="chain",
        help="Use DSSP fractions from binder chain or all residues (default: chain).",
    )
    p.add_argument(
        "--cycles",
        choices=["all", "final"],
        default="all",
        help="Use all cycles or only final cycle=5 (default: all).",
    )
    p.add_argument(
        "--out-prefix",
        default="unconditional_helixkill_secondary_structure_model",
        help="Output filename prefix.",
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


def parse_float(v: str) -> float | None:
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(v: str) -> int | None:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None


def discover_cases(root: Path, predictors: list[str]) -> list[str]:
    seen: set[str] = set()
    for pred in predictors:
        dssp_csv = root / pred / "dssp_secondary_structure_per_structure.csv"
        if not dssp_csv.exists():
            continue
        with dssp_csv.open() as f:
            for row in csv.DictReader(f):
                case = str(row.get("case", "")).strip()
                if case.startswith("unconditional_"):
                    seen.add(case)
    return sorted(seen, key=case_sort_key)


def load_rows(
    root: Path,
    predictors: list[str],
    cases: list[str],
    scope: str,
    cycles_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    suffix = "chain" if scope == "chain" else "all"
    cols = {
        "helix": f"helix_frac_{suffix}",
        "sheet": f"sheet_frac_{suffix}",
        "loop": f"loop_frac_{suffix}",
    }

    for pred in predictors:
        dssp_csv = root / pred / "dssp_secondary_structure_per_structure.csv"
        if not dssp_csv.exists():
            continue
        with dssp_csv.open() as f:
            for row in csv.DictReader(f):
                case = str(row.get("case", "")).strip()
                if case not in cases:
                    continue
                cyc = parse_int(row.get("cycle", ""))
                if cyc is None:
                    continue
                if cycles_mode == "final" and cyc != 5:
                    continue
                run_i = parse_int(row.get("run", ""))
                for ss in SS_TYPES:
                    val = parse_float(row.get(cols[ss], ""))
                    if val is None:
                        continue
                    rows.append(
                        {
                            "case": case,
                            "predictor": pred,
                            "run": run_i if run_i is not None else -1,
                            "cycle": cyc,
                            "structure_type": ss,
                            "value": val,
                        }
                    )
    return rows


def build_positions(cases: list[str], predictors: list[str]) -> tuple[dict, dict, dict]:
    pos: dict[tuple[str, str, str], float] = {}
    case_centers: dict[str, float] = {}
    model_centers: dict[tuple[str, str], float] = {}

    ss_step = 0.24
    pred_gap = 0.50
    case_gap = 1.25

    x = 0.0
    for case in cases:
        case_start = x
        for pred in predictors:
            pred_start = x
            for i, ss in enumerate(SS_TYPES):
                pos[(case, pred, ss)] = pred_start + i * ss_step
            model_centers[(case, pred)] = pred_start + ((len(SS_TYPES) - 1) * ss_step) / 2.0
            x = pred_start + len(SS_TYPES) * ss_step + pred_gap
        case_end = x - pred_gap
        case_centers[case] = (case_start + case_end) / 2.0
        x = case_end + case_gap
    return pos, case_centers, model_centers


def write_long_csv(rows: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "predictor", "run", "cycle", "structure_type", "value"])
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda z: (
                case_sort_key(z["case"]),
                z["predictor"],
                z["structure_type"],
                z["run"],
                z["cycle"],
            ),
        ):
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
        raise SystemExit("No unconditional_* cases found in DSSP CSV files.")

    rows = load_rows(root, predictors, cases, args.scope, args.cycles)
    if not rows:
        raise SystemExit("No rows loaded from DSSP CSV files with current filters.")

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["case"], r["predictor"], r["structure_type"])].append(r["value"])

    pos_map, case_centers, model_centers = build_positions(cases, predictors)
    fig_w = max(18.0, 2.8 * len(cases))
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))

    for case in cases:
        for pred in predictors:
            for ss in SS_TYPES:
                vals = grouped.get((case, pred, ss), [])
                if not vals:
                    continue
                x = pos_map[(case, pred, ss)]
                bp = ax.boxplot(
                    vals,
                    positions=[x],
                    widths=0.18,
                    patch_artist=True,
                    showfliers=False,
                    zorder=2,
                )
                for b in bp["boxes"]:
                    b.set(facecolor=SS_COLORS[ss], edgecolor="black", linewidth=1.2)
                for key in ("whiskers", "caps", "medians"):
                    for item in bp[key]:
                        item.set(color="black", linewidth=1.2)

    cond_centers = [case_centers[c] for c in cases]
    for i in range(len(cond_centers) - 1):
        mid = 0.5 * (cond_centers[i] + cond_centers[i + 1])
        ax.axvline(mid, color="#bdbdbd", linewidth=1.0, zorder=1)

    ax.set_xticks(cond_centers)
    ax.set_xticklabels([case_label(c) for c in cases], rotation=0)
    ax.set_xlabel("Helix-kill condition")
    ax.set_ylabel("Secondary-structure fraction (DSSP)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.set_axisbelow(True)

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

    legend_handles = [Patch(facecolor=SS_COLORS[ss], edgecolor="black", label=SS_DISPLAY[ss]) for ss in SS_TYPES]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    out_svg = root / f"{args.out_prefix}.svg"
    out_png = root / f"{args.out_prefix}.png"
    out_long_csv = root / f"{args.out_prefix}_long.csv"
    write_long_csv(rows, out_long_csv)

    fig.subplots_adjust(bottom=0.26, right=0.83)
    fig.savefig(out_svg, format="svg", dpi=300)
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)

    print(f"Wrote: {out_svg}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_long_csv}")


if __name__ == "__main__":
    main()
