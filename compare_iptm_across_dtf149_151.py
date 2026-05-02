#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, pearsonr


@dataclass(frozen=True)
class CampaignSpec:
    run_name: str
    design_predictor: str
    post_predictors: Tuple[str, str]


CAMPAIGNS: Tuple[CampaignSpec, ...] = (
    CampaignSpec("dTF149", "boltz", ("intellifold", "openfold3")),
    CampaignSpec("dTF150", "intellifold", ("boltz", "openfold3")),
    CampaignSpec("dTF151", "openfold3", ("boltz", "intellifold")),
)


def load_iptm_csv(path: Path, iptm_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")
    df[iptm_col_name] = pd.to_numeric(df["iptm"], errors="coerce")
    out = df[["run", "cycle", iptm_col_name]].dropna(subset=["run", "cycle"]).copy()
    out["run"] = out["run"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    return out


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(pearsonr(x, y).statistic)


def plot_scatter_per_campaign(
    design_df: pd.DataFrame,
    merged_by_post: Dict[str, pd.DataFrame],
    campaign: CampaignSpec,
    out_path: Path,
) -> List[Dict[str, object]]:
    fig, axes = plt.subplots(1, len(campaign.post_predictors), figsize=(12, 5), constrained_layout=True)
    if len(campaign.post_predictors) == 1:
        axes = [axes]

    stats_rows: List[Dict[str, object]] = []
    for ax, post in zip(axes, campaign.post_predictors):
        merged = merged_by_post[post]
        valid = merged.dropna(subset=["design_iptm", "post_iptm"])
        n_all = int(merged.shape[0])
        n_valid = int(valid.shape[0])
        r = np.nan

        if n_valid > 0:
            x = valid["design_iptm"].to_numpy()
            y = valid["post_iptm"].to_numpy()
            r = pearson_safe(x, y)
            ax.scatter(x, y, s=12, alpha=0.35)
            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            r_text = "nan" if np.isnan(r) else f"{r:.3f}"
            ax.text(
                0.03,
                0.97,
                f"n_valid={n_valid}/{n_all}\nr={r_text}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No valid iPTM pairs\n(n_valid=0/{n_all})",
                transform=ax.transAxes,
                va="center",
                ha="center",
                fontsize=10,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        ax.set_title(f"{campaign.design_predictor} design vs {post} post")
        ax.set_xlabel("Design iPTM")
        ax.set_ylabel("Post iPTM")

        stats_rows.append(
            {
                "campaign": campaign.run_name,
                "design_predictor": campaign.design_predictor,
                "post_predictor": post,
                "n_pairs": n_all,
                "n_valid_pairs": n_valid,
                "pearson_r": r,
            }
        )

    fig.suptitle(f"{campaign.run_name}: Design vs Post iPTM Correlation", fontsize=13)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return stats_rows


def kde_line(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return
    if vals.size < 2 or np.allclose(vals, vals[0]):
        ax.axvline(float(vals[0]), linestyle="--", linewidth=1.4, label=f"{label} (single)")
        return
    xs = np.linspace(0.0, 1.0, 400)
    ys = gaussian_kde(vals)(xs)
    ax.plot(xs, ys, linewidth=2, label=label)


def plot_kde_per_campaign(
    design_df: pd.DataFrame,
    post_df_by_name: Dict[str, pd.DataFrame],
    campaign: CampaignSpec,
    out_path: Path,
) -> List[Dict[str, object]]:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    dist_rows: List[Dict[str, object]] = []

    design_vals = design_df["design_iptm"].to_numpy()
    kde_line(ax, design_vals, f"design ({campaign.design_predictor})")
    dist_rows.append(
        {
            "campaign": campaign.run_name,
            "predictor_role": "design",
            "predictor": campaign.design_predictor,
            "n_values": int(np.isfinite(design_vals).sum()),
            "mean_iptm": float(np.nanmean(design_vals)) if np.isfinite(design_vals).any() else np.nan,
            "median_iptm": float(np.nanmedian(design_vals)) if np.isfinite(design_vals).any() else np.nan,
        }
    )

    for post in campaign.post_predictors:
        vals = post_df_by_name[post]["post_iptm"].to_numpy()
        kde_line(ax, vals, f"post ({post})")
        dist_rows.append(
            {
                "campaign": campaign.run_name,
                "predictor_role": "post",
                "predictor": post,
                "n_values": int(np.isfinite(vals).sum()),
                "mean_iptm": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                "median_iptm": float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan,
            }
        )

    ax.set_title(f"{campaign.run_name}: iPTM KDE Distributions")
    ax.set_xlabel("iPTM")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return dist_rows


def main() -> None:
    repo_root = Path("/Users/thomasfryer/iProteinHunter")
    out_root = repo_root / "output" / "iptm_compare_dTF149_151"
    out_root.mkdir(parents=True, exist_ok=True)

    corr_rows: List[Dict[str, object]] = []
    dist_rows: List[Dict[str, object]] = []

    for campaign in CAMPAIGNS:
        campaign_root = repo_root / "output" / campaign.run_name
        design_path = campaign_root / "summary_all_runs.csv"
        if not design_path.exists():
            print(f"Skipping {campaign.run_name}: missing {design_path}")
            continue

        design_df = load_iptm_csv(design_path, "design_iptm")
        post_df_by_name: Dict[str, pd.DataFrame] = {}
        merged_by_post: Dict[str, pd.DataFrame] = {}

        for post in campaign.post_predictors:
            post_path = campaign_root / f"summary_post_{post}.csv"
            if not post_path.exists():
                print(f"Missing post summary for {campaign.run_name}: {post_path}")
                post_df = pd.DataFrame(columns=["run", "cycle", "post_iptm"])
            else:
                post_df = load_iptm_csv(post_path, "post_iptm")
            post_df_by_name[post] = post_df
            merged_by_post[post] = design_df.merge(post_df, on=["run", "cycle"], how="inner")

        scatter_out = out_root / f"scatter_{campaign.run_name}_{campaign.design_predictor}.png"
        kde_out = out_root / f"kde_{campaign.run_name}_{campaign.design_predictor}.png"

        corr_rows.extend(plot_scatter_per_campaign(design_df, merged_by_post, campaign, scatter_out))
        dist_rows.extend(plot_kde_per_campaign(design_df, post_df_by_name, campaign, kde_out))

    corr_df = pd.DataFrame(corr_rows)
    dist_df = pd.DataFrame(dist_rows)
    corr_df.to_csv(out_root / "iptm_correlation_stats.csv", index=False)
    dist_df.to_csv(out_root / "iptm_distribution_stats.csv", index=False)

    print(f"Wrote plots and stats to: {out_root}")


if __name__ == "__main__":
    main()
