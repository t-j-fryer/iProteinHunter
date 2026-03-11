#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import gemmi
from Bio.PDB import MMCIFParser, PDBIO
from scipy import stats


HELIX_CODES = {"H", "G", "I"}
SHEET_CODES = {"E", "B"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DSSP-based secondary-structure analysis for helix-kill benchmark outputs."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Campaign root (e.g. output/test_unconditional_helixkill_YYYYmmdd_HHMMSS)",
    )
    p.add_argument(
        "--chain-id",
        default="A",
        help="Binder chain ID to analyze separately (default: A)",
    )
    return p.parse_args()


def parse_case(case_name: str) -> tuple[int, float]:
    if case_name == "unconditional_nohelixkill":
        return 0, 0.0
    m = re.search(r"nhc(\d+)$", case_name)
    if m:
        return 1, int(m.group(1)) / 100.0
    return 0, math.nan


def case_sort_key(case_name: str) -> tuple[float, str]:
    hk, nhc = parse_case(case_name)
    if hk == 0:
        return (-1.0, case_name)
    return (nhc, case_name)


def to_three_state_counts(ss_codes: list[str]) -> tuple[int, int, int]:
    h = sum(1 for c in ss_codes if c in HELIX_CODES)
    s = sum(1 for c in ss_codes if c in SHEET_CODES)
    l = len(ss_codes) - h - s
    return h, s, l


def _cif_to_pdb_with_gemmi(path: Path) -> Path:
    def has_atoms(structure) -> bool:
        for model in structure:
            for chain in model:
                for res in chain:
                    for _ in res:
                        return True
        return False

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        out_pdb = Path(tmp.name)

    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        st = None

    if st is not None and len(st) > 0 and has_atoms(st):
        st.write_pdb(str(out_pdb))
        return out_pdb

    # Fallback for CIFs that don't parse via read_structure (e.g. missing occupancy).
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    tab = block.find_mmcif_category("_atom_site.")
    if len(tab) == 0:
        raise ValueError(f"No _atom_site category found in {path}")

    tags = [str(t) for t in tab.tags]
    idx_full = {t: i for i, t in enumerate(tags)}
    idx_short = {}
    for t, i in idx_full.items():
        key = t.split(".", 1)[1] if "." in t else t
        idx_short[key] = i

    def get(row, name: str, default: str = "") -> str:
        if name in idx_short:
            return row[idx_short[name]]
        full = f"_atom_site.{name}"
        if full in idx_full:
            return row[idx_full[full]]
        return default

    def as_float(v: str, default: float = 0.0) -> float:
        try:
            if v in ("", ".", "?"):
                return default
            return float(v)
        except Exception:
            return default

    def as_int(v: str, default: int = 0) -> int:
        try:
            if v in ("", ".", "?"):
                return default
            return int(float(v))
        except Exception:
            return default

    lines = []
    serial = 1
    for row in tab:
        rec = (get(row, "group_PDB", "ATOM") or "ATOM").strip().upper()
        if rec not in ("ATOM", "HETATM"):
            continue
        atom_name = (get(row, "auth_atom_id") or get(row, "label_atom_id") or "X").strip()
        alt = (get(row, "label_alt_id", " ") or " ").strip()
        if alt in (".", "?", ""):
            alt = " "
        resname = (get(row, "auth_comp_id") or get(row, "label_comp_id") or "UNK").strip()[:3]
        chain = (get(row, "auth_asym_id") or get(row, "label_asym_id") or "A").strip()
        chain = chain[0] if chain else "A"
        resseq = as_int(get(row, "auth_seq_id") or get(row, "label_seq_id"), 1)
        icode = (get(row, "pdbx_PDB_ins_code", " ") or " ").strip()
        if icode in (".", "?", ""):
            icode = " "
        x = as_float(get(row, "Cartn_x"))
        y = as_float(get(row, "Cartn_y"))
        z = as_float(get(row, "Cartn_z"))
        occ = as_float(get(row, "occupancy"), 1.0)
        b = as_float(get(row, "B_iso_or_equiv"), 0.0)
        elem = (get(row, "type_symbol") or atom_name[:1] or "X").strip().upper()[:2]
        charge = (get(row, "pdbx_formal_charge", "") or "").strip()
        if charge in (".", "?"):
            charge = ""
        if charge and len(charge) == 1 and charge in "+-":
            charge = f"1{charge}"
        charge = charge[:2]
        atom_name_fmt = atom_name[:4].rjust(4)
        line = (
            f"{rec:<6}{serial:>5} {atom_name_fmt}{alt:1}{resname:>3} {chain:1}"
            f"{resseq:>4}{icode:1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{b:>6.2f}"
            f"          {elem:>2}{charge:>2}"
        )
        lines.append(line)
        serial += 1

    if not lines:
        raise ValueError(f"Failed to convert CIF atoms for {path}")

    with out_pdb.open("w") as fh:
        for ln in lines:
            fh.write(ln + "\n")
        fh.write("END\n")
    return out_pdb


def dssp_codes_from_structure(path: Path) -> tuple[list[str], list[str]]:
    if path.suffix.lower() == ".pdb":
        traj = md.load(str(path))
    else:
        tmp_path = None
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("x", str(path))
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(tmp_path))
        except Exception:
            tmp_path = _cif_to_pdb_with_gemmi(path)
        try:
            traj = md.load(str(tmp_path))
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    dssp = md.compute_dssp(traj, simplified=False)[0].tolist()
    chain_ids = [getattr(res.chain, "chain_id", "") for res in traj.topology.residues]
    return dssp, chain_ids


def load_rows(root: Path, chain_id: str) -> list[dict]:
    rows: list[dict] = []
    case_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("unconditional_")]
    for case_dir in case_dirs:
        case = case_dir.name
        hk, nhc = parse_case(case)

        for model_path in sorted(case_dir.glob("run_*/cycle_*/pred_min/model_0.cif")):
            run_tag = model_path.parts[-4]
            cycle_tag = model_path.parts[-3]
            run_i = int(run_tag.split("_")[1])
            cyc_i = int(cycle_tag.split("_")[1])
            try:
                dssp_codes, chain_ids = dssp_codes_from_structure(model_path)
            except Exception:
                pdb_path = model_path.with_suffix(".pdb")
                if not pdb_path.exists():
                    continue
                dssp_codes, chain_ids = dssp_codes_from_structure(pdb_path)

            all_h, all_s, all_l = to_three_state_counts(dssp_codes)
            n_all = len(dssp_codes)

            a_codes = [c for c, cid in zip(dssp_codes, chain_ids) if cid == chain_id]
            a_h, a_s, a_l = to_three_state_counts(a_codes) if a_codes else (0, 0, 0)
            n_a = len(a_codes)

            rows.append(
                {
                    "case": case,
                    "helix_kill": hk,
                    "negative_helix_constant": nhc,
                    "run": run_i,
                    "cycle": cyc_i,
                    "model_path": str(model_path),
                    "n_all": n_all,
                    "helix_all": all_h,
                    "sheet_all": all_s,
                    "loop_all": all_l,
                    "helix_frac_all": all_h / n_all if n_all else 0.0,
                    "sheet_frac_all": all_s / n_all if n_all else 0.0,
                    "loop_frac_all": all_l / n_all if n_all else 0.0,
                    "n_chain": n_a,
                    "helix_chain": a_h,
                    "sheet_chain": a_s,
                    "loop_chain": a_l,
                    "helix_frac_chain": a_h / n_a if n_a else 0.0,
                    "sheet_frac_chain": a_s / n_a if n_a else 0.0,
                    "loop_frac_chain": a_l / n_a if n_a else 0.0,
                }
            )
    return rows


def write_per_structure(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "case",
        "helix_kill",
        "negative_helix_constant",
        "run",
        "cycle",
        "model_path",
        "n_all",
        "helix_all",
        "sheet_all",
        "loop_all",
        "helix_frac_all",
        "sheet_frac_all",
        "loop_frac_all",
        "n_chain",
        "helix_chain",
        "sheet_chain",
        "loop_chain",
        "helix_frac_chain",
        "sheet_frac_chain",
        "loop_frac_chain",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["negative_helix_constant"] = f"{r['negative_helix_constant']:.2f}"
            w.writerow(out)


def summarize(rows: list[dict], subset_name: str, out_csv: Path) -> list[dict]:
    summary: list[dict] = []
    cases = sorted({r["case"] for r in rows}, key=case_sort_key)
    for case in cases:
        rs = [r for r in rows if r["case"] == case]
        for scope in ("chain", "all"):
            hf = [r[f"helix_frac_{scope}"] for r in rs]
            sf = [r[f"sheet_frac_{scope}"] for r in rs]
            lf = [r[f"loop_frac_{scope}"] for r in rs]
            hk, nhc = parse_case(case)
            summary.append(
                {
                    "subset": subset_name,
                    "case": case,
                    "helix_kill": hk,
                    "negative_helix_constant": nhc,
                    "scope": scope,
                    "n_structures": len(rs),
                    "helix_mean_frac": float(np.mean(hf)),
                    "helix_sd_frac": float(np.std(hf, ddof=1)) if len(hf) > 1 else 0.0,
                    "sheet_mean_frac": float(np.mean(sf)),
                    "sheet_sd_frac": float(np.std(sf, ddof=1)) if len(sf) > 1 else 0.0,
                    "loop_mean_frac": float(np.mean(lf)),
                    "loop_sd_frac": float(np.std(lf, ddof=1)) if len(lf) > 1 else 0.0,
                }
            )

    fields = [
        "subset",
        "case",
        "helix_kill",
        "negative_helix_constant",
        "scope",
        "n_structures",
        "helix_mean_frac",
        "helix_sd_frac",
        "sheet_mean_frac",
        "sheet_sd_frac",
        "loop_mean_frac",
        "loop_sd_frac",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            out = dict(r)
            out["negative_helix_constant"] = f"{r['negative_helix_constant']:.2f}"
            w.writerow(out)
    return summary


def _holm_correct(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def run_stats(rows: list[dict], subset_name: str, out_csv: Path) -> None:
    cases = sorted({r["case"] for r in rows}, key=case_sort_key)
    groups = {c: [r["helix_frac_chain"] for r in rows if r["case"] == c] for c in cases}

    out_rows: list[dict] = []
    if len(cases) >= 2:
        h, p = stats.kruskal(*[groups[c] for c in cases])
        out_rows.append(
            {
                "subset": subset_name,
                "test": "kruskal_helix_frac_chain",
                "group_a": ",".join(cases),
                "group_b": "",
                "statistic": h,
                "p_value": p,
                "p_value_holm": "",
            }
        )

    base = "unconditional_nohelixkill"
    comps = [c for c in cases if c != base]
    raw_pvals = []
    mw_rows = []
    for c in comps:
        u, p = stats.mannwhitneyu(groups[base], groups[c], alternative="two-sided")
        raw_pvals.append(p)
        mw_rows.append(
            {
                "subset": subset_name,
                "test": "mannwhitney_helix_frac_chain",
                "group_a": base,
                "group_b": c,
                "statistic": u,
                "p_value": p,
            }
        )
    adj = _holm_correct(raw_pvals) if raw_pvals else []
    for row, p_adj in zip(mw_rows, adj):
        row["p_value_holm"] = p_adj
        out_rows.append(row)

    hk_rows = [r for r in rows if r["helix_kill"] == 1]
    if hk_rows:
        x = np.array([r["negative_helix_constant"] for r in hk_rows], dtype=float)
        y = np.array([r["helix_frac_chain"] for r in hk_rows], dtype=float)
        rho, p = stats.spearmanr(x, y)
        out_rows.append(
            {
                "subset": subset_name,
                "test": "spearman_nhc_vs_helix_frac_chain",
                "group_a": "negative_helix_constant",
                "group_b": "helix_frac_chain",
                "statistic": rho,
                "p_value": p,
                "p_value_holm": "",
            }
        )

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "test",
                "group_a",
                "group_b",
                "statistic",
                "p_value",
                "p_value_holm",
            ],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)


def plot_effect(rows: list[dict], subset_name: str, out_svg: Path) -> None:
    cases = sorted({r["case"] for r in rows}, key=case_sort_key)
    x_labels = []
    x = []
    h, s, l = [], [], []
    for i, case in enumerate(cases):
        rs = [r for r in rows if r["case"] == case]
        hk, nhc = parse_case(case)
        x.append(i)
        x_labels.append("off" if hk == 0 else f"{nhc:.2f}")
        h.append(np.mean([r["helix_frac_chain"] for r in rs]))
        s.append(np.mean([r["sheet_frac_chain"] for r in rs]))
        l.append(np.mean([r["loop_frac_chain"] for r in rs]))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, h, marker="o", color="#d62728", label="Helix")
    ax.plot(x, s, marker="o", color="#1f77b4", label="Sheet")
    ax.plot(x, l, marker="o", color="#2ca02c", label="Loop")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("negative-helix-constant (off = no helix kill)")
    ax.set_ylabel("Mean fraction (chain A)")
    ax.set_title(f"DSSP secondary structure vs helix-kill strength ({subset_name})")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def plot_box_helix(rows: list[dict], subset_name: str, out_svg: Path) -> None:
    cases = sorted({r["case"] for r in rows}, key=case_sort_key)
    data = [[r["helix_frac_chain"] for r in rows if r["case"] == c] for c in cases]
    labels = []
    for c in cases:
        hk, nhc = parse_case(c)
        labels.append("off" if hk == 0 else f"{nhc:.2f}")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
    for box in bp["boxes"]:
        box.set(facecolor="#f2f2f2", edgecolor="#333333")
    for med in bp["medians"]:
        med.set(color="#d62728", linewidth=2)
    ax.set_xlabel("negative-helix-constant (off = no helix kill)")
    ax.set_ylabel("Helix fraction (chain A)")
    ax.set_title(f"DSSP helix fraction distribution ({subset_name})")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    rows = load_rows(root, args.chain_id)
    if not rows:
        raise SystemExit("No structures parsed for DSSP analysis.")

    out_per = root / "dssp_secondary_structure_per_structure.csv"
    write_per_structure(rows, out_per)

    rows_all = list(rows)
    rows_final = [r for r in rows if r["cycle"] == 5]

    out_summary_all = root / "dssp_secondary_structure_summary_by_case_all_cycles.csv"
    out_summary_final = root / "dssp_secondary_structure_summary_by_case_final_cycle.csv"
    summarize(rows_all, "all_cycles", out_summary_all)
    summarize(rows_final, "final_cycle", out_summary_final)

    out_stats_all = root / "dssp_stats_chainA_all_cycles.csv"
    out_stats_final = root / "dssp_stats_chainA_final_cycle.csv"
    run_stats(rows_all, "all_cycles", out_stats_all)
    run_stats(rows_final, "final_cycle", out_stats_final)

    plot_effect(rows_all, "all cycles", root / "dssp_effect_chainA_all_cycles.svg")
    plot_effect(rows_final, "final cycle", root / "dssp_effect_chainA_final_cycle.svg")
    plot_box_helix(rows_all, "all cycles", root / "dssp_boxplot_helix_chainA_all_cycles.svg")
    plot_box_helix(rows_final, "final cycle", root / "dssp_boxplot_helix_chainA_final_cycle.svg")

    print(f"Wrote: {out_per}")
    print(f"Wrote: {out_summary_all}")
    print(f"Wrote: {out_summary_final}")
    print(f"Wrote: {out_stats_all}")
    print(f"Wrote: {out_stats_final}")
    print(f"Wrote: {root / 'dssp_effect_chainA_all_cycles.svg'}")
    print(f"Wrote: {root / 'dssp_effect_chainA_final_cycle.svg'}")
    print(f"Wrote: {root / 'dssp_boxplot_helix_chainA_all_cycles.svg'}")
    print(f"Wrote: {root / 'dssp_boxplot_helix_chainA_final_cycle.svg'}")


if __name__ == "__main__":
    main()
