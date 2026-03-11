#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate AlphaFold3-style JSON inputs from iProteinHunter "
            "summary_all_runs.csv files."
        )
    )
    p.add_argument(
        "--campaign-root",
        default="",
        help="Campaign root (used only if --summary-csv is not provided).",
    )
    p.add_argument(
        "--summary-csv",
        default="",
        help="Path to one summary_all_runs.csv to convert (recommended).",
    )
    p.add_argument(
        "--template-json",
        required=True,
        help="Example AF3 JSON template file.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: <summary dir>/alphafold3_inputs).",
    )
    p.add_argument(
        "--mode",
        choices=("single_chain", "ppi", "protein_ligand"),
        default="single_chain",
        help="single_chain (legacy), ppi (binder+target), or protein_ligand.",
    )
    p.add_argument(
        "--design-predictor",
        default="",
        help="Design predictor label used in output names (default: infer from path).",
    )
    p.add_argument(
        "--design-condition",
        default="",
        help="Design condition label used in output names (default: infer from path).",
    )
    p.add_argument(
        "--exclude-cycle00",
        action="store_true",
        help="Exclude cycle 00 rows from conversion.",
    )
    p.add_argument(
        "--replace-x-with",
        default="",
        help="Replace unknown residue X in binder sequences with this residue (e.g. A).",
    )
    p.add_argument(
        "--skip-sequences-with-x",
        action="store_true",
        help="Skip any rows where binder sequence contains X.",
    )
    p.add_argument(
        "--target-chain-id",
        default="B",
        help="Target chain id for --mode ppi (default: B).",
    )
    p.add_argument(
        "--target-sequence",
        default="",
        help="Target sequence for --mode ppi.",
    )
    p.add_argument(
        "--target-query-json",
        default="",
        help="Optional path to target_query.json to auto-extract target sequence.",
    )
    p.add_argument(
        "--target-unpaired-msa-path",
        default="",
        help="Path to place in AF3 JSON as chain-B unpairedMsaPath (for --mode ppi).",
    )
    p.add_argument(
        "--target-msa-source",
        default="",
        help="Optional source A3M to copy/sanitize for chain-B MSA.",
    )
    p.add_argument(
        "--copy-target-msa-to",
        default="",
        help="Optional local destination A3M path to write sanitized target MSA.",
    )
    p.add_argument(
        "--ligand-yaml",
        default="",
        help="Boltz-style input YAML containing a ligand block (for --mode protein_ligand).",
    )
    p.add_argument(
        "--ligand-chain-id",
        default="B",
        help="Ligand chain id in AF3 JSON (for --mode protein_ligand).",
    )
    p.add_argument(
        "--ligand-smiles",
        default="",
        help="Ligand SMILES string (for --mode protein_ligand).",
    )
    p.add_argument(
        "--ligand-ccd-codes",
        default="",
        help="Comma-separated CCD code(s) for ligand (for --mode protein_ligand).",
    )
    return p.parse_args()


def load_template(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Template must be a JSON object: {path}")
    if "sequences" not in data or not isinstance(data["sequences"], list) or not data["sequences"]:
        raise SystemExit(f"Template missing sequences[0]: {path}")
    first = data["sequences"][0]
    if not isinstance(first, dict) or "protein" not in first:
        raise SystemExit(f"Template sequences[0] must contain protein: {path}")
    return data


def infer_labels(summary_csv: Path) -> tuple[str, str]:
    # .../design_<predictor>_post_.../summary_all_runs.csv
    # predictor='boltz' and condition='<folder_name>'
    parent_name = summary_csv.parent.name
    predictor = ""
    # Newer unconditional runs use folder names like:
    # unconditional_<predictor>_<condition>
    m = re.match(r"^unconditional_(boltz|intellifold|openfold-3-mlx|openfold3)_.+$", parent_name.lower())
    if m:
        predictor = m.group(1)
    # PPI campaign layout:
    # ppi_<predictor>_<condition>
    m = re.match(r"^ppi_(boltz|intellifold|openfold-3-mlx|openfold3)_.+$", parent_name.lower())
    if m:
        predictor = m.group(1)
    if parent_name.startswith("design_"):
        stem = parent_name[len("design_") :]
        predictor = stem.split("_post_")[0] if "_post_" in stem else stem
    if not predictor:
        lowered = parent_name.lower()
        for candidate in ("boltz", "intellifold", "openfold3", "openfold-3-mlx"):
            if lowered.startswith(candidate):
                predictor = candidate
                break
    if not predictor:
        predictor = "design"
    return predictor, parent_name


def read_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open() as f:
        return list(csv.DictReader(f))


def extract_target_sequence_from_query_json(path: Path, target_chain_id: str) -> str:
    with path.open() as f:
        data = json.load(f)
    queries = data.get("queries", {})
    for q in queries.values():
        for c in q.get("chains", []):
            chain_ids = c.get("chain_ids") or []
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            if target_chain_id in chain_ids:
                seq = str(c.get("sequence", "")).strip()
                if seq:
                    return seq
    return ""


def copy_sanitized_a3m(src: Path, dst: Path) -> None:
    b = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b.replace(b"\x00", b""))


def _strip_yaml_scalar(value: str) -> str:
    v = value.strip()
    if not v:
        return ""
    if " #" in v:
        v = v.split(" #", 1)[0].rstrip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        v = v[1:-1]
    return v.strip()


def _parse_yaml_list(value: str) -> List[str]:
    v = _strip_yaml_scalar(value)
    if not v:
        return []
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_strip_yaml_scalar(x) for x in inner.split(",") if _strip_yaml_scalar(x)]
    return [v]


def parse_ligand_from_boltz_yaml(path: Path) -> Dict[str, object]:
    lines = path.read_text().splitlines()
    in_ligand = False
    ligand_indent = -1
    ligand: Dict[str, object] = {}

    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        if stripped.startswith("- ligand:"):
            in_ligand = True
            ligand_indent = indent
            continue

        if in_ligand and indent <= ligand_indent and stripped.startswith("- "):
            break
        if not in_ligand or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "id":
            ligand["id"] = _strip_yaml_scalar(value)
        elif key == "smiles":
            ligand["smiles"] = _strip_yaml_scalar(value)
        elif key in ("ccd", "ccd_code"):
            parsed = _strip_yaml_scalar(value)
            if parsed:
                ligand["ccdCodes"] = [parsed]
        elif key == "ccd_codes":
            parsed = _parse_yaml_list(value)
            if parsed:
                ligand["ccdCodes"] = parsed

    return ligand


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else None
    campaign_root = Path(args.campaign_root).resolve() if args.campaign_root else None
    template_json = Path(args.template_json).resolve()
    template = load_template(template_json)

    if summary_csv is None:
        if campaign_root is None:
            raise SystemExit("Provide either --summary-csv or --campaign-root.")
        # Backward-compatible discovery for old unconditional campaigns.
        summary_files = sorted(campaign_root.glob("*/unconditional_*/summary_all_runs.csv"))
        if not summary_files:
            # Current layout: <campaign_root>/unconditional_*/summary_all_runs.csv
            summary_files = sorted(campaign_root.glob("unconditional_*/summary_all_runs.csv"))
        if not summary_files:
            # Generic campaign layout: <campaign_root>/<case_name>/summary_all_runs.csv
            summary_files = sorted(campaign_root.glob("*/summary_all_runs.csv"))
        if not summary_files:
            raise SystemExit(f"No summary_all_runs.csv files found under {campaign_root}")
    else:
        if not summary_csv.exists():
            raise SystemExit(f"summary_all_runs.csv not found: {summary_csv}")
        summary_files = [summary_csv]

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        base = summary_csv.parent if summary_csv else campaign_root
        if args.mode == "ppi":
            suffix = "alphafold3_inputs_ppi"
        elif args.mode == "protein_ligand":
            suffix = "alphafold3_inputs_protein_ligand"
        else:
            suffix = "alphafold3_inputs_single_chain"
        out_dir = base / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    target_seq = args.target_sequence.strip()
    if args.mode == "ppi" and not target_seq and args.target_query_json:
        target_seq = extract_target_sequence_from_query_json(Path(args.target_query_json).resolve(), args.target_chain_id)

    if args.mode == "ppi" and not target_seq:
        raise SystemExit("For --mode ppi, provide --target-sequence or --target-query-json.")
    if args.mode == "ppi" and not args.target_unpaired_msa_path.strip():
        raise SystemExit("For --mode ppi, provide --target-unpaired-msa-path.")
    if args.target_msa_source and args.copy_target_msa_to:
        copy_sanitized_a3m(Path(args.target_msa_source).resolve(), Path(args.copy_target_msa_to).resolve())

    total = 0
    skipped_x = 0
    skipped_cycle0 = 0
    index_rows = []
    for summary_csv in summary_files:
        predictor_infer, condition_infer = infer_labels(summary_csv)
        predictor = args.design_predictor.strip() or predictor_infer
        condition = args.design_condition.strip() or condition_infer
        rows = read_rows(summary_csv)

        ligand_spec: Dict[str, object] = {}
        if args.mode == "protein_ligand":
            ligand_spec = {"id": args.ligand_chain_id.strip() or "B"}
            smiles = args.ligand_smiles.strip()
            ccd_codes = [x.strip() for x in args.ligand_ccd_codes.split(",") if x.strip()]
            if smiles and ccd_codes:
                raise SystemExit("For --mode protein_ligand, use either --ligand-smiles or --ligand-ccd-codes, not both.")
            if smiles:
                ligand_spec["smiles"] = smiles
            elif ccd_codes:
                ligand_spec["ccdCodes"] = ccd_codes
            else:
                ligand_yaml = Path(args.ligand_yaml).resolve() if args.ligand_yaml else (summary_csv.parent / "_calibration" / "cycle_00" / "boltz_input.yaml")
                if not ligand_yaml.exists():
                    raise SystemExit(
                        f"For --mode protein_ligand, no ligand info was provided and default ligand YAML was not found: {ligand_yaml}"
                    )
                parsed = parse_ligand_from_boltz_yaml(ligand_yaml)
                if not parsed:
                    raise SystemExit(f"No ligand block found in YAML: {ligand_yaml}")
                if parsed.get("id"):
                    ligand_spec["id"] = str(parsed["id"]).strip()
                if "smiles" in parsed and parsed["smiles"]:
                    ligand_spec["smiles"] = parsed["smiles"]
                if "ccdCodes" in parsed and parsed["ccdCodes"]:
                    ligand_spec["ccdCodes"] = parsed["ccdCodes"]
            if ("smiles" in ligand_spec) == ("ccdCodes" in ligand_spec):
                raise SystemExit(
                    "For --mode protein_ligand, ligand must have exactly one of smiles or ccdCodes."
                )

        for row in rows:
            run = str(row.get("run", "")).strip()
            cycle = str(row.get("cycle", "")).strip()
            sequence = str(row.get("binder_sequence", "")).strip().upper()
            if not run or not cycle or not sequence:
                continue

            if args.exclude_cycle00 and cycle.isdigit() and int(cycle) == 0:
                skipped_cycle0 += 1
                continue
            if args.replace_x_with:
                sequence = sequence.replace("X", args.replace_x_with.strip().upper())
            elif args.skip_sequences_with_x and "X" in sequence:
                skipped_x += 1
                continue

            name = f"{predictor}_{condition}_run_{int(run):03d}_cycle_{int(cycle):02d}"
            out_json = out_dir / f"{name}.json"

            if args.mode == "single_chain":
                first_protein_template = deepcopy(template["sequences"][0]["protein"])
                first_protein_template["id"] = "A"
                first_protein_template["sequence"] = sequence
                payload_sequences = [{"protein": first_protein_template}]
            elif args.mode == "ppi":
                binder = {
                    "id": "A",
                    "sequence": sequence,
                    "templates": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                }
                target = {
                    "id": args.target_chain_id,
                    "sequence": target_seq,
                    "templates": [],
                    "unpairedMsaPath": args.target_unpaired_msa_path.strip(),
                    "pairedMsa": "",
                }
                payload_sequences = [{"protein": binder}, {"protein": target}]
            else:
                binder = {
                    "id": "A",
                    "sequence": sequence,
                    "templates": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                }
                ligand_payload: Dict[str, object] = {"id": ligand_spec["id"]}
                if "smiles" in ligand_spec:
                    ligand_payload["smiles"] = ligand_spec["smiles"]
                else:
                    ligand_payload["ccdCodes"] = ligand_spec["ccdCodes"]
                payload_sequences = [{"protein": binder}, {"ligand": ligand_payload}]

            payload = {
                "name": name,
                "modelSeeds": deepcopy(template.get("modelSeeds", [42])),
                "sequences": payload_sequences,
                "dialect": template.get("dialect", "alphafold3"),
                "version": template.get("version", 4),
            }

            with out_json.open("w") as wf:
                json.dump(payload, wf, indent=2)

            index_rows.append(
                {
                    "predictor": predictor,
                    "condition": condition,
                    "run": run,
                    "cycle": cycle,
                    "sequence": sequence,
                    "json_path": str(out_json),
                }
            )
            total += 1

    index_csv = out_dir / "index.csv"
    with index_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["predictor", "condition", "run", "cycle", "sequence", "json_path"],
        )
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    print(f"Wrote {total} AF3 JSON files to: {out_dir}")
    print(f"Wrote index: {index_csv}")
    if skipped_cycle0:
        print(f"Skipped cycle_00 rows: {skipped_cycle0}")
    if skipped_x:
        print(f"Skipped rows with X residues: {skipped_x}")
    if args.target_msa_source and args.copy_target_msa_to:
        print(f"Copied sanitized target MSA to: {Path(args.copy_target_msa_to).resolve()}")


if __name__ == "__main__":
    main()
