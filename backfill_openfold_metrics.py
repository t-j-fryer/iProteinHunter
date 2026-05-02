#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_any(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys:
                return v
        for v in obj.values():
            got = get_any(v, keys)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = get_any(v, keys)
            if got is not None:
                return got
    return None


def parse_conf_metrics(conf_json: Path) -> Tuple[str, str]:
    try:
        data = json.loads(conf_json.read_text())
    except Exception:
        return "nan", "nan"
    iptm = get_any(data, {"iptm", "i_ptm", "iptm_score", "iptm+ptm"})
    plddt = get_any(data, {"complex_plddt", "avg_plddt", "plddt"})

    def to_str(x):
        try:
            return str(float(x))
        except Exception:
            return "nan"

    return to_str(iptm), to_str(plddt)


def find_openfold_outputs(openfold_root: Path) -> Optional[Tuple[Path, Path, str, str]]:
    if not openfold_root.exists():
        return None
    conf_files = sorted(openfold_root.glob("**/*_confidences_aggregated.json"))
    if not conf_files:
        return None
    conf = conf_files[0]
    seed_dir = conf.parent
    struct_candidates = sorted(seed_dir.glob("*_model.cif")) + sorted(seed_dir.glob("*_model.pdb"))
    if not struct_candidates:
        struct_candidates = sorted(openfold_root.glob("**/*_model.cif")) + sorted(openfold_root.glob("**/*_model.pdb"))
    if not struct_candidates:
        return None
    struct = struct_candidates[0]
    iptm, plddt = parse_conf_metrics(conf)
    return struct.resolve(), conf.resolve(), iptm, plddt


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fix_design_rows(campaign_root: Path) -> int:
    updated = 0
    for run_dir in sorted(campaign_root.glob("run_*")):
        metrics_path = run_dir / "metrics_per_cycle.csv"
        rows_path = run_dir / "rows.csv"
        if not metrics_path.exists():
            continue
        metrics_rows, metrics_fields = read_csv_rows(metrics_path)
        changed_metrics = False
        by_cycle: Dict[int, Tuple[str, str, str, str]] = {}

        for row in metrics_rows:
            try:
                cycle_int = int(str(row.get("cycle", "")).strip())
            except Exception:
                continue
            openfold_root = run_dir / f"cycle_{cycle_int:02d}" / "openfold3"
            found = find_openfold_outputs(openfold_root)
            if not found:
                continue
            struct, conf, iptm, plddt = found
            row["iptm"] = iptm
            row["complex_plddt"] = plddt
            row["confidence_json"] = str(conf)
            row["structure_path"] = str(struct)
            by_cycle[cycle_int] = (iptm, plddt, str(conf), str(struct))
            changed_metrics = True

        if changed_metrics:
            write_csv_rows(metrics_path, metrics_rows, metrics_fields)
            updated += 1

        if rows_path.exists() and by_cycle:
            rows_rows, rows_fields = read_csv_rows(rows_path)
            changed_rows = False
            for row in rows_rows:
                try:
                    cycle_int = int(str(row.get("cycle", "")).strip())
                except Exception:
                    continue
                if cycle_int not in by_cycle:
                    continue
                iptm, plddt, conf, struct = by_cycle[cycle_int]
                row["iptm"] = iptm
                row["complex_plddt"] = plddt
                row["confidence_json"] = conf
                row["structure_path"] = struct
                changed_rows = True
            if changed_rows:
                write_csv_rows(rows_path, rows_rows, rows_fields)
    return updated


def regenerate_summary_all_runs(campaign_root: Path) -> None:
    out_path = campaign_root / "summary_all_runs.csv"
    fields = ["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"]
    out_rows: List[Dict[str, str]] = []
    for run_dir in sorted(campaign_root.glob("run_*")):
        metrics_path = run_dir / "metrics_per_cycle.csv"
        if not metrics_path.exists():
            continue
        run_id = run_dir.name.replace("run_", "")
        try:
            run_int = int(run_id)
        except Exception:
            continue
        metrics_rows, _ = read_csv_rows(metrics_path)
        for m in metrics_rows:
            out_rows.append(
                {
                    "run": str(run_int),
                    "cycle": str(m.get("cycle", "")),
                    "iptm": str(m.get("iptm", "")),
                    "complex_plddt": str(m.get("complex_plddt", "")),
                    "binder_sequence": str(m.get("binder_sequence", "")),
                    "structure_path": str(m.get("structure_path", "")),
                    "confidence_json": str(m.get("confidence_json", "")),
                }
            )
    write_csv_rows(out_path, out_rows, fields)


def fix_post_openfold_rows(campaign_root: Path) -> int:
    updated = 0
    for run_dir in sorted(campaign_root.glob("run_*")):
        post_root = run_dir / "post_openfold3"
        if not post_root.exists():
            continue
        run_rows: List[Dict[str, str]] = []
        for cycle_dir in sorted(post_root.glob("cycle_*")):
            row_path = cycle_dir / "post_metrics_row.csv"
            if not row_path.exists():
                continue
            found = find_openfold_outputs(cycle_dir / "openfold3")
            if not found:
                continue
            struct, conf, iptm, plddt = found
            rows, fields = read_csv_rows(row_path)
            if not rows:
                continue
            row = rows[0]
            row["iptm"] = iptm
            row["complex_plddt"] = plddt
            row["structure_path"] = str(struct)
            row["confidence_json"] = str(conf)
            write_csv_rows(row_path, [row], fields)
            run_rows.append(row)
            updated += 1

        if run_rows:
            run_metrics_path = post_root / "post_metrics.csv"
            fields = ["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"]
            run_rows_sorted = sorted(run_rows, key=lambda r: int(str(r.get("cycle", "0"))))
            write_csv_rows(run_metrics_path, run_rows_sorted, fields)

    # Rebuild campaign summary_post_openfold3.csv
    summary_rows: List[Dict[str, str]] = []
    fields = ["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"]
    for run_dir in sorted(campaign_root.glob("run_*")):
        run_metrics_path = run_dir / "post_openfold3" / "post_metrics.csv"
        if not run_metrics_path.exists():
            continue
        rows, _ = read_csv_rows(run_metrics_path)
        summary_rows.extend(rows)
    summary_path = campaign_root / "summary_post_openfold3.csv"
    write_csv_rows(summary_path, summary_rows, fields)
    return updated


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill OpenFold iPTM/plddt/paths from saved aggregated confidence JSON files.")
    ap.add_argument(
        "--campaign-roots",
        nargs="+",
        required=True,
        help="Campaign root directories (e.g. .../output/dTF149).",
    )
    args = ap.parse_args()

    for root_raw in args.campaign_roots:
        root = Path(root_raw).resolve()
        if not root.exists():
            print(f"[skip] missing campaign root: {root}")
            continue
        design_updates = fix_design_rows(root)
        regenerate_summary_all_runs(root)
        post_updates = fix_post_openfold_rows(root)
        print(
            f"[ok] {root.name}: updated_design_runs={design_updates}, "
            f"updated_post_rows={post_updates}, "
            f"summary_all_runs={root / 'summary_all_runs.csv'}, "
            f"summary_post_openfold3={root / 'summary_post_openfold3.csv'}"
        )


if __name__ == "__main__":
    main()
