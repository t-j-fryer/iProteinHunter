#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr


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


def load_iptm(path: Path, predictor: str, role: str, campaign: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["iptm"] = pd.to_numeric(df["iptm"], errors="coerce")
    out = df[["run", "cycle", "iptm"]].dropna().copy()
    out["run"] = out["run"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    out["predictor"] = predictor
    out["role"] = role
    out["campaign"] = campaign
    return out


def safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(func(x, y).statistic)


def add_zscores(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = long_df.groupby("predictor", as_index=False)["iptm"].agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(columns={"count": "n_values", "mean": "mean_iptm", "std": "std_iptm"})
    stat_map = stats.set_index("predictor")[["mean_iptm", "std_iptm"]]

    z = []
    for _, row in long_df.iterrows():
        mean_ = float(stat_map.loc[row["predictor"], "mean_iptm"])
        std_ = float(stat_map.loc[row["predictor"], "std_iptm"])
        if np.isfinite(std_) and std_ > 0:
            z.append((float(row["iptm"]) - mean_) / std_)
        else:
            z.append(float("nan"))
    out = long_df.copy()
    out["z_iptm"] = z
    return out, stats


def summarize_pairs(z_df: pd.DataFrame, campaign: CampaignSpec) -> Tuple[pd.DataFrame, List[dict]]:
    design = z_df[(z_df["campaign"] == campaign.run_name) & (z_df["role"] == "design")][
        ["run", "cycle", "iptm", "z_iptm"]
    ].rename(columns={"iptm": "design_iptm", "z_iptm": "design_z"})

    rows: List[dict] = []
    per_pair = {}

    for post in campaign.post_predictors:
        post_df = z_df[
            (z_df["campaign"] == campaign.run_name)
            & (z_df["role"] == "post")
            & (z_df["predictor"] == post)
        ][["run", "cycle", "iptm", "z_iptm"]].rename(columns={"iptm": "post_iptm", "z_iptm": "post_z"})
        merged = design.merge(post_df, on=["run", "cycle"], how="inner")
        per_pair[post] = merged

        x_raw = merged["design_iptm"].to_numpy()
        y_raw = merged["post_iptm"].to_numpy()
        x_z = merged["design_z"].to_numpy()
        y_z = merged["post_z"].to_numpy()
        rows.append(
            {
                "campaign": campaign.run_name,
                "design_predictor": campaign.design_predictor,
                "post_predictor": post,
                "n_pairs": int(merged.shape[0]),
                "pearson_raw": safe_corr(pearsonr, x_raw, y_raw),
                "spearman_raw": safe_corr(spearmanr, x_raw, y_raw),
                "pearson_z": safe_corr(pearsonr, x_z, y_z),
                "spearman_z": safe_corr(spearmanr, x_z, y_z),
                "design_raw_mean": float(np.nanmean(x_raw)) if x_raw.size else float("nan"),
                "post_raw_mean": float(np.nanmean(y_raw)) if y_raw.size else float("nan"),
                "design_z_mean": float(np.nanmean(x_z)) if x_z.size else float("nan"),
                "post_z_mean": float(np.nanmean(y_z)) if y_z.size else float("nan"),
            }
        )

    p1, p2 = campaign.post_predictors
    wide = design.rename(columns={"design_iptm": f"{campaign.design_predictor}_iptm", "design_z": f"{campaign.design_predictor}_z"})
    for post in (p1, p2):
        post_df = z_df[
            (z_df["campaign"] == campaign.run_name)
            & (z_df["role"] == "post")
            & (z_df["predictor"] == post)
        ][["run", "cycle", "iptm", "z_iptm"]].rename(columns={"iptm": f"{post}_iptm", "z_iptm": f"{post}_z"})
        wide = wide.merge(post_df, on=["run", "cycle"], how="inner")

    z_cols = [f"{campaign.design_predictor}_z", f"{p1}_z", f"{p2}_z"]
    wide["consensus_z_mean"] = wide[z_cols].mean(axis=1)
    wide["consensus_z_min"] = wide[z_cols].min(axis=1)
    wide["consensus_z_std"] = wide[z_cols].std(axis=1)
    wide = wide.sort_values(["consensus_z_mean", "consensus_z_min"], ascending=[False, False]).reset_index(drop=True)
    wide["rank_consensus_z_mean"] = np.arange(1, len(wide) + 1)
    wide.insert(0, "campaign", campaign.run_name)
    return wide, rows


def plot_consensus_scatter(rank_df: pd.DataFrame, campaign: CampaignSpec, out_path: Path) -> None:
    p1, p2 = campaign.post_predictors
    dz = f"{campaign.design_predictor}_z"
    p1z = f"{p1}_z"
    p2z = f"{p2}_z"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].scatter(rank_df[dz], rank_df[p1z], s=12, alpha=0.35)
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[0].axvline(0.0, linestyle="--", linewidth=1)
    axes[0].set_title(f"{campaign.design_predictor} (design) vs {p1} (post)")
    axes[0].set_xlabel(f"{campaign.design_predictor} z-iPTM")
    axes[0].set_ylabel(f"{p1} z-iPTM")

    axes[1].scatter(rank_df[dz], rank_df[p2z], s=12, alpha=0.35)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].axvline(0.0, linestyle="--", linewidth=1)
    axes[1].set_title(f"{campaign.design_predictor} (design) vs {p2} (post)")
    axes[1].set_xlabel(f"{campaign.design_predictor} z-iPTM")
    axes[1].set_ylabel(f"{p2} z-iPTM")

    fig.suptitle(f"{campaign.run_name}: z-scored iPTM comparisons", fontsize=13)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path("/Users/thomasfryer/iProteinHunter")
    out_root = repo_root / "output" / "iptm_compare_dTF149_151_zscore"
    out_root.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for c in CAMPAIGNS:
        croot = repo_root / "output" / c.run_name
        design_path = croot / "summary_all_runs.csv"
        if design_path.exists():
            long_rows.append(load_iptm(design_path, c.design_predictor, "design", c.run_name))
        for post in c.post_predictors:
            post_path = croot / f"summary_post_{post}.csv"
            if post_path.exists():
                long_rows.append(load_iptm(post_path, post, "post", c.run_name))

    if not long_rows:
        raise SystemExit("No input summaries found.")

    long_df = pd.concat(long_rows, ignore_index=True)
    z_df, predictor_stats = add_zscores(long_df)
    z_df.to_csv(out_root / "zscore_long_all.csv", index=False)
    predictor_stats.to_csv(out_root / "zscore_predictor_reference_stats.csv", index=False)

    all_rank_rows = []
    pair_rows = []
    for c in CAMPAIGNS:
        rank_df, pair_stats = summarize_pairs(z_df, c)
        all_rank_rows.append(rank_df)
        pair_rows.extend(pair_stats)
        rank_df.to_csv(out_root / f"zscore_rankings_{c.run_name}.csv", index=False)
        rank_df.head(50).to_csv(out_root / f"zscore_rankings_top50_{c.run_name}.csv", index=False)
        plot_consensus_scatter(rank_df, c, out_root / f"scatter_z_{c.run_name}_{c.design_predictor}.png")

    pd.concat(all_rank_rows, ignore_index=True).to_csv(out_root / "zscore_rankings_all_campaigns.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_root / "zscore_pairwise_stats.csv", index=False)
    print(f"Wrote z-score comparisons to: {out_root}")


if __name__ == "__main__":
    main()
