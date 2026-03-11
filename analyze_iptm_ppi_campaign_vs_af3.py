from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================a
# USER SETTINGS
# =========================
CAMPAIGN_ROOT = Path(
    "/Users/thomasfryer/iProteinHunter/output/PPI_HelixKill_AllDesign_SolMPNN"
).resolve()

# Folder containing AF3 output subfolders named like:
#   boltz_ppi_boltz_hk_0p2_run_001_cycle_00/
# with files:
#   *_summary_confidences.json
AF3_OUTPUT_ROOT = Path(
    "/Users/thomasfryer/iProteinHunter/output/PPI_HelixKill_AllDesign_SolMPNN/AF3_Outputs"
).resolve()

OUT_DIR = (CAMPAIGN_ROOT / "ipTM_analysis_af3").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

WHITE_BG = True
EXCLUDE_CYCLE0 = True

DARK_FIG_BG = "#111111"
LIGHT_FIG_BG = "#ffffff"
LIGHT_TEXT = "#111111"
DARK_TEXT = "#ffffff"

PREDICTOR_ORDER = ["boltz", "intellifold", "openfold-3-mlx"]
PREDICTOR_LABEL = {
    "boltz": "Boltz-2",
    "intellifold": "IntelliFold",
    "openfold-3-mlx": "OpenFold-3 (preview)",
}

CYCLE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
]

HIST_BINS = 30
KDE_GRID_N = 400

CASE_RE = re.compile(r"^ppi_(boltz|intellifold|openfold-3-mlx)_(.+)$", re.IGNORECASE)
AF3_SUMMARY_RE = re.compile(r"(.+)_summary_confidences\.json$", re.IGNORECASE)


# =========================
# Utils
# =========================
def safe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def setup_axes_style(ax):
    if WHITE_BG:
        ax.figure.set_facecolor(LIGHT_FIG_BG)
        ax.set_facecolor(LIGHT_FIG_BG)
        ax.tick_params(colors=LIGHT_TEXT)
        ax.xaxis.label.set_color(LIGHT_TEXT)
        ax.yaxis.label.set_color(LIGHT_TEXT)
        ax.title.set_color(LIGHT_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#333333")
    else:
        ax.figure.set_facecolor(DARK_FIG_BG)
        ax.set_facecolor(DARK_FIG_BG)
        ax.tick_params(colors=DARK_TEXT)
        ax.xaxis.label.set_color(DARK_TEXT)
        ax.yaxis.label.set_color(DARK_TEXT)
        ax.title.set_color(DARK_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")


def gaussian_kde_simple(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(grid)
    std = np.std(x, ddof=1)
    if std <= 1e-12:
        std = 1e-3
    h = std * (n ** (-1 / 5))
    if h <= 1e-12:
        h = 1e-3
    z = (grid[:, None] - x[None, :]) / h
    dens = np.mean(np.exp(-0.5 * z * z), axis=1) / (h * np.sqrt(2 * np.pi))
    return dens


def parse_condition_label(cond: str) -> str:
    c = cond.lower()
    if c == "nohelix":
        return "No Helix Kill"
    m = re.match(r"hk_(\d+)p(\d+)", c)
    if m:
        return f"HKC {m.group(1)}.{m.group(2)}"
    return cond


def condition_sort_key(cond: str) -> tuple:
    c = cond.lower()
    if c == "nohelix":
        return (0, 0.0)
    m = re.match(r"hk_(\d+)p(\d+)", c)
    if m:
        return (1, float(f"{m.group(1)}.{m.group(2)}"))
    return (9, 0.0)


def predictor_sort_key(p: str) -> int:
    p = p.lower()
    if p in PREDICTOR_ORDER:
        return PREDICTOR_ORDER.index(p)
    return 99


def extract_iptm_from_json(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    # Common key variants in AF3 summary outputs
    for k in (
        "iptm", "i_ptm", "iptm_score", "iptm+ptm", "ranking_score", "iptm_global"
    ):
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v

    if isinstance(d.get("scores"), dict):
        v = safe_float(d["scores"].get("iptm"))
        if v is not None:
            return v

    return None


@dataclass(frozen=True)
class RowKey:
    predictor: str
    case_name: str
    run: int
    cycle: int

    @property
    def af3_name(self) -> str:
        return f"{self.predictor}_{self.case_name}_run_{self.run:03d}_cycle_{self.cycle:02d}"


# =========================
# 1) Ingest design summary rows
# =========================
rows_design = []
for case_dir in sorted(CAMPAIGN_ROOT.glob("ppi_*/summary_all_runs.csv")):
    case_name = case_dir.parent.name
    m = CASE_RE.match(case_name)
    if not m:
        continue
    predictor = m.group(1).lower()
    condition = m.group(2)
    condition_label = parse_condition_label(condition)

    with case_dir.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            run_i = int(r["run"])
            cyc_i = int(r["cycle"])
            if EXCLUDE_CYCLE0 and cyc_i == 0:
                continue
            iptm = safe_float(r.get("iptm"))
            if iptm is None:
                continue

            key = RowKey(
                predictor=predictor,
                case_name=case_name,
                run=run_i,
                cycle=cyc_i,
            )
            rows_design.append(
                dict(
                    key=key,
                    predictor=predictor,
                    predictor_label=PREDICTOR_LABEL.get(predictor, predictor),
                    condition=condition,
                    condition_label=condition_label,
                    case_name=case_name,
                    run=run_i,
                    cycle=cyc_i,
                    design_iptm=iptm,
                    binder_sequence=r.get("binder_sequence", ""),
                    design_confidence_json=r.get("confidence_json", ""),
                    design_structure_path=r.get("structure_path", ""),
                )
            )

print(f"Design rows parsed: {len(rows_design)}")
if not rows_design:
    raise SystemExit("No design rows parsed from summary_all_runs.csv")


# =========================
# 2) Ingest AF3 summary confidences
# =========================
af3_iptm_by_name = {}
if AF3_OUTPUT_ROOT.exists():
    for p in sorted(AF3_OUTPUT_ROOT.rglob("*_summary_confidences.json")):
        m = AF3_SUMMARY_RE.match(p.name)
        if not m:
            continue
        name = m.group(1)
        iptm = extract_iptm_from_json(p)
        if iptm is None:
            continue
        af3_iptm_by_name[name] = (iptm, str(p))
else:
    print(f"[WARN] AF3_OUTPUT_ROOT does not exist: {AF3_OUTPUT_ROOT}")

print(f"AF3 summary files parsed: {len(af3_iptm_by_name)}")


# =========================
# 3) Join design rows to AF3 rows by canonical name
# =========================
rows = []
matched = 0
for r in rows_design:
    af3_name = r["key"].af3_name
    af3_hit = af3_iptm_by_name.get(af3_name)
    af3_iptm = af3_hit[0] if af3_hit else None
    af3_json = af3_hit[1] if af3_hit else ""
    if af3_hit:
        matched += 1

    rows.append(
        dict(
            predictor=r["predictor"],
            predictor_label=r["predictor_label"],
            condition=r["condition"],
            condition_label=r["condition_label"],
            case_name=r["case_name"],
            run=r["run"],
            cycle=r["cycle"],
            af3_name=af3_name,
            design_iptm=r["design_iptm"],
            af3_iptm=af3_iptm,
            delta_af3_minus_design=(af3_iptm - r["design_iptm"]) if af3_iptm is not None else None,
            design_confidence_json=r["design_confidence_json"],
            design_structure_path=r["design_structure_path"],
            af3_summary_conf_json=af3_json,
            binder_sequence=r["binder_sequence"],
        )
    )

print(f"Joined rows: {len(rows)} | matched AF3: {matched}")


# =========================
# 4) Export long CSV
# =========================
out_csv = OUT_DIR / "iptm_design_vs_af3_long.csv"
fieldnames = [
    "predictor", "predictor_label", "condition", "condition_label", "case_name",
    "run", "cycle", "af3_name", "design_iptm", "af3_iptm", "delta_af3_minus_design",
    "design_confidence_json", "design_structure_path", "af3_summary_conf_json",
    "binder_sequence",
]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in sorted(
        rows,
        key=lambda z: (
            condition_sort_key(z["condition"]),
            predictor_sort_key(z["predictor"]),
            z["cycle"],
            z["run"],
        ),
    ):
        w.writerow(r)
print("Wrote:", out_csv)


# =========================
# Prepare plotting groups
# =========================
conditions = sorted({r["condition"] for r in rows}, key=condition_sort_key)
predictors = sorted({r["predictor"] for r in rows}, key=predictor_sort_key)
cycles = sorted({r["cycle"] for r in rows})

if EXCLUDE_CYCLE0:
    cycles = [c for c in cycles if c != 0]
if not cycles:
    raise SystemExit("No cycles available after EXCLUDE_CYCLE0 filtering")


def build_group_positions(condition_keys, predictor_keys, cycle_keys):
    # x-axis groups are (condition, predictor); within each group, split by cycle
    pos = {}
    centers = {}
    cycle_step = 0.18
    pred_gap = 0.45
    cond_gap = 0.80

    x = 0.0
    for cond in condition_keys:
        cond_start = x
        for pred in predictor_keys:
            g_start = x
            for i, cyc in enumerate(cycle_keys):
                pos[(cond, pred, cyc)] = x + i * cycle_step
            g_end = x + (len(cycle_keys) - 1) * cycle_step
            centers[(cond, pred)] = 0.5 * (g_start + g_end)
            x = g_end + pred_gap
        # increase gap between conditions
        x += cond_gap

    return pos, centers


def grouped_vals(metric: str, require_af3: bool = False):
    g = defaultdict(list)
    for r in rows:
        if require_af3 and r.get("af3_iptm") is None:
            continue
        v = r.get(metric)
        if v is None:
            continue
        g[(r["condition"], r["predictor"], r["cycle"])].append(v)
    return g


pos_map, centers = build_group_positions(conditions, predictors, cycles)
edge = "#111111" if WHITE_BG else "#dddddd"
colors = CYCLE_COLORS[: len(cycles)]
if len(colors) < len(cycles):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(1, len(cycles) - 1)) for i in range(len(cycles))]


# =========================
# 5) Box plot: DESIGN ipTM
# =========================
fig_w = max(16.0, 1.5 * len(conditions) * len(predictors))
fig, ax = plt.subplots(figsize=(fig_w, 7.2))
setup_axes_style(ax)

g_design = grouped_vals("design_iptm", require_af3=False)
for cond in conditions:
    for pred in predictors:
        for i, cyc in enumerate(cycles):
            vals = g_design.get((cond, pred, cyc), [])
            if not vals:
                continue
            x = pos_map[(cond, pred, cyc)]
            bp = ax.boxplot(vals, positions=[x], widths=0.14, patch_artist=True, showfliers=False, zorder=2)
            for b in bp["boxes"]:
                b.set(facecolor=colors[i], edgecolor=edge, linewidth=1.1)
            for key in ("whiskers", "caps", "medians"):
                for item in bp[key]:
                    item.set(color=edge, linewidth=1.1)

xticks = []
xticklabels = []
for cond in conditions:
    for pred in predictors:
        xticks.append(centers[(cond, pred)])
        xticklabels.append(f"{parse_condition_label(cond)}\n{PREDICTOR_LABEL.get(pred, pred)}")

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=28, ha="right", fontsize=8)
ax.set_ylabel("ipTM")
ax.set_ylim(0, 1.0)
ax.set_title("Design ipTM by condition, design model, and cycle")
ax.grid(False)

legend_handles = [Patch(facecolor=colors[i], edgecolor=edge, label=f"Cycle {cyc}") for i, cyc in enumerate(cycles)]
ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

fig.subplots_adjust(bottom=0.35, right=0.82)
out_png = OUT_DIR / "design_iptm_condition_model_cycle.png"
out_svg = OUT_DIR / "design_iptm_condition_model_cycle.svg"
fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
fig.savefig(out_svg, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print("Wrote:", out_png)
print("Wrote:", out_svg)


# =========================
# 6) Box plot: AF3 ipTM
# =========================
fig, ax = plt.subplots(figsize=(fig_w, 7.2))
setup_axes_style(ax)

g_af3 = grouped_vals("af3_iptm", require_af3=True)
for cond in conditions:
    for pred in predictors:
        for i, cyc in enumerate(cycles):
            vals = g_af3.get((cond, pred, cyc), [])
            if not vals:
                continue
            x = pos_map[(cond, pred, cyc)]
            bp = ax.boxplot(vals, positions=[x], widths=0.14, patch_artist=True, showfliers=False, zorder=2)
            for b in bp["boxes"]:
                b.set(facecolor=colors[i], edgecolor=edge, linewidth=1.1)
            for key in ("whiskers", "caps", "medians"):
                for item in bp[key]:
                    item.set(color=edge, linewidth=1.1)

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=28, ha="right", fontsize=8)
ax.set_ylabel("ipTM")
ax.set_ylim(0, 1.0)
ax.set_title("AF3 ipTM by condition, design model, and cycle")
ax.grid(False)
ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

fig.subplots_adjust(bottom=0.35, right=0.82)
out_png = OUT_DIR / "af3_iptm_condition_model_cycle.png"
out_svg = OUT_DIR / "af3_iptm_condition_model_cycle.svg"
fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
fig.savefig(out_svg, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print("Wrote:", out_png)
print("Wrote:", out_svg)


# =========================
# 7) Scatter: Design ipTM vs AF3 ipTM (matched rows)
# =========================
matched_rows = [r for r in rows if r["af3_iptm"] is not None]
if matched_rows:
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    setup_axes_style(ax)

    for pred in predictors:
        vals = [(r["design_iptm"], r["af3_iptm"]) for r in matched_rows if r["predictor"] == pred]
        if not vals:
            continue
        x = np.array([v[0] for v in vals], dtype=float)
        y = np.array([v[1] for v in vals], dtype=float)
        ax.scatter(x, y, s=16, alpha=0.65, label=PREDICTOR_LABEL.get(pred, pred))

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#666666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Design ipTM")
    ax.set_ylabel("AF3 ipTM")
    ax.set_title("Design vs AF3 ipTM (matched run/cycle)")
    ax.grid(False)
    ax.legend(frameon=False)

    out_png = OUT_DIR / "design_vs_af3_iptm_scatter.png"
    out_svg = OUT_DIR / "design_vs_af3_iptm_scatter.svg"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(out_svg, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Wrote:", out_png)
    print("Wrote:", out_svg)


# =========================
# 8) Delta distribution (AF3 - Design)
# =========================
if matched_rows:
    vals = np.array([r["delta_af3_minus_design"] for r in matched_rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        xmin, xmax = float(np.min(vals)), float(np.max(vals))
        pad = 0.02
        xmin -= pad
        xmax += pad

        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        setup_axes_style(ax)

        for pred in predictors:
            pvals = np.array(
                [r["delta_af3_minus_design"] for r in matched_rows if r["predictor"] == pred],
                dtype=float,
            )
            pvals = pvals[np.isfinite(pvals)]
            if pvals.size < 3:
                continue
            grid = np.linspace(xmin, xmax, KDE_GRID_N)
            dens = gaussian_kde_simple(pvals, grid)
            ax.plot(grid, dens, linewidth=1.9, label=PREDICTOR_LABEL.get(pred, pred))

        ax.axvline(0.0, linestyle="--", linewidth=1.2, color="#666666")
        ax.set_xlabel("AF3 ipTM - Design ipTM")
        ax.set_ylabel("Density (KDE)")
        ax.set_title("Delta ipTM distribution by design model")
        ax.grid(False)
        ax.legend(frameon=False)

        out_png = OUT_DIR / "delta_af3_minus_design_kde.png"
        out_svg = OUT_DIR / "delta_af3_minus_design_kde.svg"
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(out_svg, dpi=300, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Wrote:", out_png)
        print("Wrote:", out_svg)


print("\nDone. Output folder:", OUT_DIR)
print(f"Rows total: {len(rows)} | AF3 matched: {matched}")

# quick console breakdown
by_pred = defaultdict(int)
for r in rows:
    by_pred[r["predictor"]] += 1
print("Rows by predictor:", dict(sorted(by_pred.items(), key=lambda kv: predictor_sort_key(kv[0]))))
