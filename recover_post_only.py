#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
BOLTZ = REPO_ROOT / "venvs/iProteinHunter_boltz/bin/boltz"
INTELLIFOLD = REPO_ROOT / "venvs/iProteinHunter_intellifold/bin/intellifold"
OPENFOLD = REPO_ROOT / "venvs/iProteinHunter_openfold3_mlx/bin/run_openfold"

DEFAULT_BOLTZ_FLAGS = [
    "--accelerator",
    "gpu",
    "--devices",
    "1",
    "--use_msa_server",
    "--msa_server_url",
    "https://api.colabfold.com",
    "--msa_pairing_strategy",
    "greedy",
]
DEFAULT_INTELLIFOLD_FLAGS = [
    "--precision",
    "no",
    "--num_workers",
    "0",
    "--model",
    "v2-flash",
    "--seed",
    "42",
    "--num_diffusion_samples",
    "1",
    "--override",
    "--use_msa_server",
    "--msa_pairing_strategy",
    "greedy",
]
DEFAULT_OPENFOLD_FLAGS = [
    "--num_diffusion_samples",
    "1",
    "--num_model_seeds",
    "1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover post-only scoring for existing iProteinHunter campaign outputs.")
    p.add_argument("--campaign-root", required=True, help="Path to design_* campaign folder.")
    p.add_argument("--template-yaml", required=True, help="Template YAML used for design run.")
    p.add_argument("--predictors", default="boltz,intellifold", help="Comma-separated post predictors to recover.")
    p.add_argument("--max-parallel", type=int, default=6, help="Max parallel post tasks.")
    p.add_argument("--design-predictor", default="openfold-3-mlx", help="Design predictor label for comparison tables.")
    p.add_argument("--limit", type=int, default=0, help="Optional per-predictor task limit for testing (0 = all).")
    p.add_argument("--log-file", default="", help="Optional recovery log path.")
    return p.parse_args()


def parse_template_target_msa(path: Path) -> str:
    cur = None
    in_protein = False
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s.startswith("- protein:"):
            in_protein = True
            cur = None
            continue
        if not in_protein:
            continue
        if s.startswith("-") and not s.startswith("- protein:"):
            in_protein = False
            continue
        if s.startswith("id:"):
            cur = s.split(":", 1)[1].strip().strip("'\"")
        if s.startswith("msa:") and cur and cur != "A":
            val = s.split(":", 1)[1].strip().strip("'\"")
            low = val.lower()
            if val and low not in {"empty", "none", "null"}:
                return val
    return ""


def parse_template_target_sequence(path: Path) -> str:
    cur = None
    seqs: dict[str, str] = {}
    in_protein = False
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s.startswith("- protein:"):
            in_protein = True
            cur = None
            continue
        if not in_protein:
            continue
        if s.startswith("-") and not s.startswith("- protein:"):
            in_protein = False
            continue
        if s.startswith("id:"):
            cur = s.split(":", 1)[1].strip().strip("'\"")
            continue
        if s.startswith("sequence:") and cur:
            seq = s.split(":", 1)[1].strip().strip("'\"")
            seqs[cur] = seq
    if seqs.get("B"):
        return seqs["B"]
    for cid in sorted(seqs):
        if cid != "A" and seqs[cid]:
            return seqs[cid]
    return ""


def parse_template_target_chain_id(path: Path) -> str:
    in_protein = False
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s.startswith("- protein:"):
            in_protein = True
            continue
        if not in_protein:
            continue
        if s.startswith("-") and not s.startswith("- protein:"):
            in_protein = False
            continue
        if s.startswith("id:"):
            cid = s.split(":", 1)[1].strip().strip("'\"")
            if cid and cid != "A":
                return cid
    return "B"


def make_yaml_with_binder_sequence(template_yaml: Path, out_yaml: Path, new_seq: str, target_msa_path: str) -> None:
    msa_path = target_msa_path.strip()
    in_binder = False
    in_protein = False
    need_msa_insert = False
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    with template_yaml.open() as fin, out_yaml.open("w") as fout:
        for line in fin:
            stripped = line.strip()

            if stripped.startswith("- protein:"):
                in_protein = True
                in_binder = False
                need_msa_insert = False
                fout.write(line)
                continue

            if in_protein and stripped.startswith("- ") and not stripped.startswith("- protein:"):
                in_protein = False
                in_binder = False
                need_msa_insert = False
                fout.write(line)
                continue

            if stripped.startswith("id:"):
                cid = stripped.split("id:", 1)[1].strip().strip("'\"")
                in_binder = cid == "A"
                need_msa_insert = bool(msa_path) and cid != "A"
                fout.write(line)
                continue

            if in_binder and stripped.startswith("sequence:"):
                indent = line.split("sequence:")[0]
                fout.write(f"{indent}sequence: {new_seq}\n")
                continue

            if need_msa_insert and stripped.startswith("msa:"):
                indent = line.split("msa:")[0]
                fout.write(f"{indent}msa: {msa_path}\n")
                need_msa_insert = False
                continue

            if need_msa_insert and stripped.startswith("sequence:"):
                fout.write(line)
                indent = line.split("sequence:")[0]
                fout.write(f"{indent}msa: {msa_path}\n")
                need_msa_insert = False
                continue

            fout.write(line)


def pick_target_msa_for_predictor(msa_path: str, predictor: str) -> str:
    def sanitize_a3m(path: Path) -> str:
        b = path.read_bytes()
        if b"\x00" not in b:
            return str(path)
        out = path.with_name(path.stem + ".sanitized.a3m")
        out.write_bytes(b.replace(b"\x00", b""))
        return str(out)

    if not msa_path:
        return ""
    p = Path(msa_path)
    if predictor in {"boltz", "intellifold"} and p.suffix.lower() == ".npz":
        root = p.parent.parent
        cand = root / "raw/main/uniref.a3m"
        if cand.exists():
            return sanitize_a3m(cand)
        a3ms = sorted((root / "raw/main").glob("*.a3m")) if (root / "raw/main").exists() else []
        if a3ms:
            return sanitize_a3m(a3ms[0])
    if predictor in {"boltz", "intellifold"} and p.suffix.lower() == ".a3m" and p.exists():
        return sanitize_a3m(p)
    if predictor == "openfold-3-mlx":
        if p.suffix.lower() in {".a3m", ".sto", ".npz"} and p.exists():
            return str(p)
        return ""
    return msa_path


def sanitize_protein_sequence_for_openfold(seq: str) -> tuple[str, int]:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    out = []
    changed = 0
    for ch in seq.strip().upper():
        if ch in allowed:
            out.append(ch)
        else:
            out.append("A")
            changed += 1
    return "".join(out), changed


def write_single_seq_a3m(seq: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f">query\n{seq}\n")


def write_openfold_runner_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "experiment_settings:",
                "  mode: predict",
                "",
                "pl_trainer_args:",
                "  accelerator: gpu",
                "  devices: 1",
                "",
                "model_update:",
                "  presets: [\"predict\", \"pae_enabled\"]",
                "  custom:",
                "    settings:",
                "      memory:",
                "        eval:",
                "          use_deepspeed_evo_attention: false",
                "          use_lma: false",
                "          use_mlx_attention: true",
                "          use_mlx_triangle_kernels: true",
                "          use_mlx_activation_functions: true",
                "",
            ]
        )
    )


def parse_yaml_sequences(path: Path) -> list[dict]:
    items = []
    cur = None
    list_key = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- ") and s.endswith(":"):
            key = s[2:-1].strip()
            if key in {"protein", "rna", "dna", "ligand"}:
                if cur:
                    items.append(cur)
                cur = {"kind": key}
                list_key = None
                continue
        if cur is None:
            continue
        if list_key and s.startswith("- "):
            v = s[2:].strip().strip("'\"")
            cur.setdefault(list_key, []).append(v)
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not v and k in {"ccd_codes", "chain_ids"}:
            cur[k] = []
            list_key = k
            continue
        list_key = None
        v = v.strip("'\"")
        if v.startswith("[") and v.endswith("]"):
            vals = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
            cur[k] = vals
        else:
            cur[k] = v
    if cur:
        items.append(cur)
    return items


def build_openfold_query_json(
    template_yaml: Path,
    binder_seq: str,
    query_name: str,
    out_json: Path,
    target_msa_path: str,
    binder_msa_path: str,
) -> bool:
    def usable(v: str) -> bool:
        low = (v or "").strip().lower()
        return bool((v or "").strip()) and low not in {"empty", "none", "null"}

    entries = parse_yaml_sequences(template_yaml)
    out_chains = []
    need_server = False

    for e in entries:
        kind = str(e.get("kind", "")).lower()
        cid = str(e.get("id", "")).strip()
        if not kind or not cid:
            continue

        if kind in {"protein", "rna", "dna"}:
            seq = str(e.get("sequence", "")).strip()
            msa = str(e.get("msa", "")).strip()
            if cid == "A" and kind == "protein":
                seq = binder_seq
                msa = binder_msa_path or ""
            elif kind in {"protein", "rna"} and (not usable(msa)) and target_msa_path:
                msa = target_msa_path
            if not seq:
                continue
            row = {"molecule_type": kind, "chain_ids": [cid], "sequence": seq}
            if kind in {"protein", "rna"}:
                if usable(msa):
                    row["main_msa_file_paths"] = [msa]
                else:
                    need_server = True
            out_chains.append(row)
            continue

        if kind == "ligand":
            row = {"molecule_type": "ligand", "chain_ids": [cid]}
            smiles = str(e.get("smiles", "")).strip()
            ccd_codes = e.get("ccd_codes")
            if smiles:
                row["smiles"] = smiles
            elif isinstance(ccd_codes, list) and ccd_codes:
                row["ccd_codes"] = [str(x) for x in ccd_codes if str(x).strip()]
            elif isinstance(ccd_codes, str) and ccd_codes.strip():
                row["ccd_codes"] = [ccd_codes.strip()]
            else:
                continue
            out_chains.append(row)

    payload = {
        "seeds": [42],
        "queries": {
            query_name: {
                "chains": out_chains,
                "use_msas": True,
                "use_main_msas": True,
                "use_paired_msas": False,
            }
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return need_server


def generate_openfold_target_msa_cache(campaign_root: Path, template_yaml: Path) -> str:
    target_seq = parse_template_target_sequence(template_yaml).strip()
    if not target_seq:
        return ""
    target_chain_id = parse_template_target_chain_id(template_yaml) or "B"

    msa_cache_dir = campaign_root / "msa_cache"
    out_dir = msa_cache_dir / "target_msa"
    query_json = msa_cache_dir / "target_query.json"
    log_file = msa_cache_dir / "target_msa.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "seeds": [42],
        "queries": {
            "target_msa_only": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [target_chain_id],
                        "sequence": target_seq,
                    }
                ]
            }
        },
    }
    query_json.write_text(json.dumps(payload, indent=2))
    cmd = [str(OPENFOLD), "align-msa-server", "--query_json", str(query_json), "--output_dir", str(out_dir)]
    env = os.environ.copy()
    env["KMP_USE_SHM"] = "0"
    run_cmd(cmd, log_file, env=env)

    # Prefer npz for OpenFold runtime speed.
    npz = sorted((out_dir / "main").glob("colabfold_main.npz"))
    if not npz:
        npz = sorted((out_dir / "main").glob("*.npz"))
    if npz:
        return str(npz[0])
    a3m = sorted((out_dir / "raw/main").glob("uniref90_hits.a3m"))
    if not a3m:
        a3m = sorted((out_dir / "raw/main").glob("uniref.a3m"))
    if not a3m:
        a3m = sorted((out_dir / "raw/main").glob("*.a3m"))
    if a3m:
        return str(a3m[0])
    return ""


def extract_metrics_from_conf_json(path: Path) -> tuple[float, float]:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return float("nan"), float("nan")

    def get_any(obj, keys: set[str]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in keys:
                    return v
            for v in obj.values():
                x = get_any(v, keys)
                if x is not None:
                    return x
        elif isinstance(obj, list):
            for v in obj:
                x = get_any(v, keys)
                if x is not None:
                    return x
        return None

    def to_float(v):
        try:
            return float(v)
        except Exception:
            return float("nan")

    iptm = get_any(d, {"iptm", "i_ptm", "iptm_score", "iptm+ptm"})
    plddt = get_any(d, {"complex_plddt", "avg_plddt", "plddt"})
    return to_float(iptm), to_float(plddt)


def run_cmd(cmd: list[str], log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")


def run_predict_boltz(input_yaml: Path, out_dir: Path, pred_min: Path) -> tuple[str, str, float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "predict.log"
    cmd = [str(BOLTZ), "predict", str(input_yaml), "--out_dir", str(out_dir), *DEFAULT_BOLTZ_FLAGS]
    run_cmd(cmd, log)

    roots = sorted(out_dir.glob("boltz_results_*"))
    if not roots:
        raise RuntimeError(f"Boltz output missing boltz_results_* in {out_dir}")
    leaf_candidates = sorted((roots[0] / "predictions").glob("*"))
    leaf_candidates = [x for x in leaf_candidates if x.is_dir()]
    if not leaf_candidates:
        raise RuntimeError(f"Boltz prediction leaf missing in {roots[0]}")
    leaf = leaf_candidates[0]

    confs = sorted(leaf.glob("confidence_*_model_0.json"))
    structs = sorted([*leaf.glob("*.cif"), *leaf.glob("*.pdb")])
    if not structs:
        raise RuntimeError(f"Boltz structure not found in {leaf}")

    pred_min.mkdir(parents=True, exist_ok=True)
    conf_path = ""
    if confs:
        conf_path = str(confs[0])
        (pred_min / "confidence.json").write_bytes(confs[0].read_bytes())
    struct = structs[0]
    if struct.suffix.lower() == ".cif":
        (pred_min / "model_0.cif").write_bytes(struct.read_bytes())
    else:
        (pred_min / "model_0.pdb").write_bytes(struct.read_bytes())

    iptm, plddt = (float("nan"), float("nan"))
    if (pred_min / "confidence.json").exists():
        iptm, plddt = extract_metrics_from_conf_json(pred_min / "confidence.json")
    return str(struct), conf_path, iptm, plddt


def run_predict_intellifold(input_yaml: Path, out_dir: Path, pred_min: Path) -> tuple[str, str, float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "predict.log"
    cmd = [str(INTELLIFOLD), "predict", str(input_yaml), "--out_dir", str(out_dir), *DEFAULT_INTELLIFOLD_FLAGS]
    env = os.environ.copy()
    env["KMP_USE_SHM"] = "0"
    run_cmd(cmd, log, env=env)

    stem = input_yaml.stem
    leaf = out_dir / stem / "predictions" / stem
    if not leaf.exists():
        raise RuntimeError(f"IntelliFold prediction leaf missing: {leaf}")
    confs = sorted(leaf.glob("*_summary_confidences.json"))
    structs = sorted(leaf.glob("*.cif"))
    if not structs:
        raise RuntimeError(f"IntelliFold structure not found in {leaf}")

    pred_min.mkdir(parents=True, exist_ok=True)
    conf_path = ""
    if confs:
        conf_path = str(confs[0])
        (pred_min / "confidence.json").write_bytes(confs[0].read_bytes())
    struct = structs[0]
    (pred_min / "model_0.cif").write_bytes(struct.read_bytes())

    iptm, plddt = (float("nan"), float("nan"))
    if (pred_min / "confidence.json").exists():
        iptm, plddt = extract_metrics_from_conf_json(pred_min / "confidence.json")
    return str(struct), conf_path, iptm, plddt


def run_predict_openfold(input_yaml: Path, binder_seq: str, query_name: str, out_dir: Path, pred_min: Path, target_msa_path: str) -> tuple[str, str, float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    query_json = out_dir / f"{query_name}_query.json"
    runner_yaml = out_dir / f"{query_name}_runner.yml"
    binder_msa_path = out_dir / "binder_msa/uniref90_hits.a3m"
    log = out_dir / "predict.log"

    binder_clean, _ = sanitize_protein_sequence_for_openfold(binder_seq)
    write_single_seq_a3m(binder_clean, binder_msa_path)
    use_server = build_openfold_query_json(input_yaml, binder_clean, query_name, query_json, target_msa_path, str(binder_msa_path))
    write_openfold_runner_yaml(runner_yaml)

    cmd = [
        str(OPENFOLD),
        "predict",
        "--query_json",
        str(query_json),
        "--output_dir",
        str(out_dir),
        "--runner_yaml",
        str(runner_yaml),
        "--use_msa_server",
        "true" if use_server else "false",
        *DEFAULT_OPENFOLD_FLAGS,
    ]
    env = os.environ.copy()
    env["KMP_USE_SHM"] = "0"
    run_cmd(cmd, log, env=env)

    leaf = out_dir / query_name / "seed_42"
    if not leaf.exists():
        seeds = sorted(out_dir.glob("**/seed_*"))
        if seeds:
            leaf = seeds[0]
    if not leaf.exists():
        raise RuntimeError(f"OpenFold seed output not found in {out_dir}")

    confs = sorted(leaf.glob("*_confidences_aggregated.json"))
    structs = sorted(list(leaf.glob("*_model.cif")) + list(leaf.glob("*_model.pdb")))
    if not structs:
        raise RuntimeError(f"OpenFold structure not found in {leaf}")

    pred_min.mkdir(parents=True, exist_ok=True)
    conf_path = ""
    if confs:
        conf_path = str(confs[0])
        (pred_min / "confidence.json").write_bytes(confs[0].read_bytes())
    struct = structs[0]
    if struct.suffix.lower() == ".cif":
        (pred_min / "model_0.cif").write_bytes(struct.read_bytes())
    else:
        (pred_min / "model_0.pdb").write_bytes(struct.read_bytes())

    iptm, plddt = (float("nan"), float("nan"))
    if (pred_min / "confidence.json").exists():
        iptm, plddt = extract_metrics_from_conf_json(pred_min / "confidence.json")
    return str(struct), conf_path, iptm, plddt


def fmt_num(v: float) -> str:
    return "nan" if (isinstance(v, float) and math.isnan(v)) else str(v)


def predictor_safe_name(predictor: str) -> str:
    p = predictor.strip().lower()
    if p in {"openfold-3-mlx", "openfold3", "openfold", "of3"}:
        return "openfold3"
    if p in {"intellifold", "boltz"}:
        return p
    return p.replace("-", "_")


def write_row_csv(path: Path, header: list[str], row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def read_tasks(task_tsv: Path) -> list[tuple[int, int, str]]:
    tasks = []
    if not task_tsv.exists():
        return tasks
    for line in task_tsv.read_text().splitlines():
        if not line.strip():
            continue
        run_s, cyc_s, seq = line.split("\t", 2)
        tasks.append((int(run_s), int(cyc_s), seq.strip()))
    return tasks


def recover_predictor(
    campaign_root: Path,
    template_yaml: Path,
    predictor: str,
    target_msa_path: str,
    max_parallel: int,
    limit: int,
    logf,
) -> None:
    pred_safe = predictor_safe_name(predictor)
    task_tsv = campaign_root / f"post_{pred_safe}_tasks.tsv"
    tasks_all = read_tasks(task_tsv)
    tasks: list[tuple[int, int, str]] = []
    for run_i, cyc_i, seq in tasks_all:
        run_tag = f"run_{run_i:03d}"
        cycle_tag = f"cycle_{cyc_i:02d}"
        done_row = campaign_root / run_tag / f"post_{pred_safe}" / cycle_tag / "post_metrics_row.csv"
        if done_row.exists():
            continue
        tasks.append((run_i, cyc_i, seq))
    if limit > 0:
        tasks = tasks[:limit]
    print(f"[{predictor}] tasks remaining: {len(tasks)} (from total {len(tasks_all)})", file=logf, flush=True)
    if not tasks:
        return

    target_msa_for_pred = pick_target_msa_for_predictor(target_msa_path, predictor)
    if predictor == "openfold-3-mlx" and not target_msa_for_pred:
        raise RuntimeError(
            "OpenFold recovery requires an OpenFold-compatible target MSA path (a3m/sto/npz)."
        )
    print(f"[{predictor}] using target msa: {target_msa_for_pred}", file=logf, flush=True)

    def run_one(task: tuple[int, int, str]) -> tuple[bool, str]:
        run_i, cyc_i, binder_seq = task
        run_tag = f"run_{run_i:03d}"
        cycle_tag = f"cycle_{cyc_i:02d}"
        post_cycle_root = campaign_root / run_tag / f"post_{pred_safe}" / cycle_tag
        input_yaml = post_cycle_root / "post_input.yaml"
        pred_min = post_cycle_root / "pred_min"
        try:
            make_yaml_with_binder_sequence(template_yaml, input_yaml, binder_seq, target_msa_for_pred)
            t0 = time.time()
            if predictor == "boltz":
                struct, conf, iptm, plddt = run_predict_boltz(input_yaml, post_cycle_root / "boltz", pred_min)
            elif predictor == "intellifold":
                struct, conf, iptm, plddt = run_predict_intellifold(input_yaml, post_cycle_root / "intellifold", pred_min)
            elif predictor == "openfold-3-mlx":
                qname = f"{run_tag}_post_{cyc_i:02d}"
                struct, conf, iptm, plddt = run_predict_openfold(
                    input_yaml, binder_seq, qname, post_cycle_root / "openfold3", pred_min, target_msa_for_pred
                )
            else:
                return False, f"Unsupported predictor: {predictor}"
            t1 = time.time()

            write_row_csv(
                post_cycle_root / "post_metrics_row.csv",
                ["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"],
                [str(run_i), str(cyc_i), fmt_num(iptm), fmt_num(plddt), binder_seq, struct, conf],
            )
            write_row_csv(
                post_cycle_root / "post_timing_row.csv",
                ["run", "cycle", "start_ts", "end_ts", "duration_sec"],
                [str(run_i), str(cyc_i), f"{t0:.6f}", f"{t1:.6f}", f"{max(0.0, t1-t0):.6f}"],
            )
            return True, f"{run_tag} {cycle_tag} ok"
        except Exception as e:
            return False, f"{run_tag} {cycle_tag} failed: {e}"

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                success, msg = fut.result()
            except Exception as e:
                success, msg = False, f"worker crashed: {e}"
            print(f"[{predictor}] {msg}", file=logf, flush=True)
            if success:
                ok += 1
            else:
                fail += 1
    print(f"[{predictor}] done: ok={ok}, fail={fail}", file=logf, flush=True)


def aggregate_post(campaign_root: Path, predictor: str, n_runs: int) -> None:
    pred_safe = predictor_safe_name(predictor)
    summary_csv = campaign_root / f"summary_post_{pred_safe}.csv"
    summary_timing_csv = campaign_root / f"summary_post_{pred_safe}_timing.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"])
    with summary_timing_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "cycle", "start_ts", "end_ts", "duration_sec"])

    for run_i in range(1, n_runs + 1):
        run_tag = f"run_{run_i:03d}"
        post_root = campaign_root / run_tag / f"post_{pred_safe}"
        if not post_root.exists():
            continue

        run_metrics = post_root / "post_metrics.csv"
        run_timing = post_root / "post_timing.csv"
        with run_metrics.open("w", newline="") as f:
            csv.writer(f).writerow(["run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"])
        with run_timing.open("w", newline="") as f:
            csv.writer(f).writerow(["run", "cycle", "start_ts", "end_ts", "duration_sec"])

        mrows = sorted(post_root.glob("cycle_*/post_metrics_row.csv"))
        trows = sorted(post_root.glob("cycle_*/post_timing_row.csv"))

        for row_file in mrows:
            with row_file.open() as rf:
                rows = list(csv.reader(rf))
            if len(rows) >= 2:
                with run_metrics.open("a", newline="") as f:
                    csv.writer(f).writerow(rows[1])
                with summary_csv.open("a", newline="") as f:
                    csv.writer(f).writerow(rows[1])

        for row_file in trows:
            with row_file.open() as rf:
                rows = list(csv.reader(rf))
            if len(rows) >= 2:
                with run_timing.open("a", newline="") as f:
                    csv.writer(f).writerow(rows[1])
                with summary_timing_csv.open("a", newline="") as f:
                    csv.writer(f).writerow(rows[1])


def build_comparison_tables(campaign_root: Path, design_predictor: str, post_predictors: Iterable[str], n_runs: int) -> None:
    cmp_scores = campaign_root / "comparison_scores_long.csv"
    cmp_timing = campaign_root / "comparison_timing_long.csv"
    with cmp_scores.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["stage", "predictor", "run", "cycle", "iptm", "complex_plddt", "binder_sequence", "structure_path", "confidence_json"]
        )
    with cmp_timing.open("w", newline="") as f:
        csv.writer(f).writerow(["stage", "predictor", "run", "phase", "cycle", "start_ts", "end_ts", "duration_sec"])

    for run_i in range(1, n_runs + 1):
        run_tag = f"run_{run_i:03d}"
        mfile = campaign_root / run_tag / "metrics_per_cycle.csv"
        if mfile.exists():
            with mfile.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    with cmp_scores.open("a", newline="") as out:
                        csv.writer(out).writerow(
                            [
                                "design",
                                design_predictor,
                                run_i,
                                row.get("cycle", ""),
                                row.get("iptm", ""),
                                row.get("complex_plddt", ""),
                                row.get("binder_sequence", ""),
                                row.get("structure_path", ""),
                                row.get("confidence_json", ""),
                            ]
                        )
        cfile = campaign_root / run_tag / "timing_cycles.csv"
        if cfile.exists():
            with cfile.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    with cmp_timing.open("a", newline="") as out:
                        csv.writer(out).writerow(
                            ["design", design_predictor, run_i, "cycle", row.get("cycle", ""), row.get("start_ts", ""), row.get("end_ts", ""), row.get("duration_sec", "")]
                        )
        rfile = campaign_root / run_tag / "timing_run.csv"
        if rfile.exists():
            with rfile.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    with cmp_timing.open("a", newline="") as out:
                        csv.writer(out).writerow(
                            ["design", design_predictor, run_i, "run", "all", row.get("start_ts", ""), row.get("end_ts", ""), row.get("duration_sec", "")]
                        )

    for pp in post_predictors:
        pred_safe = predictor_safe_name(pp)
        sfile = campaign_root / f"summary_post_{pred_safe}.csv"
        tfile = campaign_root / f"summary_post_{pred_safe}_timing.csv"
        if sfile.exists():
            with sfile.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    with cmp_scores.open("a", newline="") as out:
                        csv.writer(out).writerow(
                            [
                                "post",
                                pp,
                                row.get("run", ""),
                                f"{int(row.get('cycle', 0)):02d}" if row.get("cycle", "").isdigit() else row.get("cycle", ""),
                                row.get("iptm", ""),
                                row.get("complex_plddt", ""),
                                row.get("binder_sequence", ""),
                                row.get("structure_path", ""),
                                row.get("confidence_json", ""),
                            ]
                        )
        if tfile.exists():
            with tfile.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    with cmp_timing.open("a", newline="") as out:
                        csv.writer(out).writerow(
                            [
                                "post",
                                pp,
                                row.get("run", ""),
                                "post_cycle",
                                f"{int(row.get('cycle', 0)):02d}" if row.get("cycle", "").isdigit() else row.get("cycle", ""),
                                row.get("start_ts", ""),
                                row.get("end_ts", ""),
                                row.get("duration_sec", ""),
                            ]
                        )


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    template_yaml = Path(args.template_yaml).resolve()
    predictors = [x.strip() for x in args.predictors.split(",") if x.strip()]

    if not campaign_root.exists():
        raise SystemExit(f"Campaign root not found: {campaign_root}")
    if not template_yaml.exists():
        raise SystemExit(f"Template YAML not found: {template_yaml}")

    log_file = Path(args.log_file).resolve() if args.log_file else campaign_root / "post_recovery.log"
    with log_file.open("w") as logf:
        print(f"campaign_root={campaign_root}", file=logf, flush=True)
        print(f"template_yaml={template_yaml}", file=logf, flush=True)
        print(f"predictors={predictors}", file=logf, flush=True)
        print(f"max_parallel={args.max_parallel}", file=logf, flush=True)

        target_msa_path = ""
        msa_path_file = campaign_root / "msa_cache/target_msa_path.txt"
        if msa_path_file.exists():
            target_msa_path = msa_path_file.read_text().strip()
        if not target_msa_path:
            target_msa_path = parse_template_target_msa(template_yaml)
        if "openfold-3-mlx" in predictors:
            p = Path(target_msa_path) if target_msa_path else None
            is_compatible = bool(p and p.exists() and p.suffix.lower() in {".a3m", ".sto", ".npz"})
            if not is_compatible:
                openfold_target = generate_openfold_target_msa_cache(campaign_root, template_yaml)
                if openfold_target:
                    target_msa_path = openfold_target
        print(f"target_msa_path={target_msa_path}", file=logf, flush=True)

        summary_all = campaign_root / "summary_all_runs.csv"
        if not summary_all.exists():
            raise SystemExit(f"Missing summary_all_runs.csv: {summary_all}")
        with summary_all.open() as f:
            r = csv.DictReader(f)
            runs = {int(row["run"]) for row in r if str(row.get("run", "")).isdigit()}
        n_runs = max(runs) if runs else 0
        print(f"n_runs={n_runs}", file=logf, flush=True)

        for pred in predictors:
            recover_predictor(campaign_root, template_yaml, pred, target_msa_path, args.max_parallel, args.limit, logf)
            aggregate_post(campaign_root, pred, n_runs)

        build_comparison_tables(campaign_root, args.design_predictor, predictors, n_runs)
        print("done", file=logf, flush=True)

    print(f"Recovery complete. Log: {log_file}")


if __name__ == "__main__":
    main()
