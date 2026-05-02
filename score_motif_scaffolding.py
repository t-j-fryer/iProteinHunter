#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class AtomRecord:
    chain_id: str
    atom_name: str
    residue_name: str
    record_type: str
    element: str
    x: float
    y: float
    z: float
    label_seq_id: Optional[int]
    auth_seq_id: Optional[int]
    ordinal_seq_id: int
    model_num: int
    ins_code: str = ""


def parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip()
    if v in {"", ".", "?", "None"}:
        return None
    try:
        return int(v)
    except Exception:
        return None


def parse_ranges(value: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    raw = (value or "").strip()
    if not raw:
        return out
    for token in raw.split(";"):
        tok = token.strip()
        if not tok:
            continue
        if "-" not in tok:
            raise ValueError(f"Invalid range token '{tok}'")
        a, b = tok.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if start > end:
            raise ValueError(f"Invalid range '{tok}': start > end")
        out.append((start, end))
    return out


def expand_ranges(ranges: Iterable[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    for a, b in ranges:
        out.extend(range(a, b + 1))
    return out


def parse_positions(value: str) -> List[int]:
    raw = (value or "").strip()
    if not raw:
        return []
    return [int(x) for x in raw.split() if x.strip()]


def find_design_structure(run_root: Path, design: int, cycle: int) -> Optional[Path]:
    run_tag = f"run_{design:03d}"
    cyc_tag = f"cycle_{cycle:02d}"
    pred_min = run_root / run_tag / cyc_tag / "pred_min"
    for name in ("model_0.cif", "model_0.pdb"):
        p = pred_min / name
        if p.is_file():
            return p
    return None


def _tokenize_mmcif_row(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    try:
        return shlex.split(stripped, posix=True)
    except Exception:
        return stripped.split()


def parse_mmcif_atoms(path: Path) -> List[AtomRecord]:
    lines = path.read_text(errors="ignore").splitlines()
    atoms: List[AtomRecord] = []
    chain_res_seen: Dict[Tuple[int, str, str, str, str], int] = {}
    next_ord = 1

    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        cols: List[str] = []
        while j < len(lines) and lines[j].strip().startswith("_"):
            cols.append(lines[j].strip())
            j += 1
        if not cols or not any(c.startswith("_atom_site.") for c in cols):
            i = j
            continue

        col_idx = {c: idx for idx, c in enumerate(cols)}

        def get_tok(tokens: List[str], key: str, default: str = "") -> str:
            idx = col_idx.get(key)
            if idx is None or idx >= len(tokens):
                return default
            return tokens[idx]

        k = j
        while k < len(lines):
            s = lines[k].strip()
            if not s:
                k += 1
                continue
            if s.startswith("#"):
                break
            if s == "loop_" or s.startswith("_"):
                break

            tokens = _tokenize_mmcif_row(lines[k])
            if len(tokens) < len(cols):
                k += 1
                continue

            model_num = parse_int(get_tok(tokens, "_atom_site.pdbx_PDB_model_num", "1")) or 1
            if model_num != 1:
                k += 1
                continue

            group = get_tok(tokens, "_atom_site.group_PDB", "")
            if group not in {"ATOM", "HETATM"}:
                k += 1
                continue

            alt_id = get_tok(tokens, "_atom_site.label_alt_id", ".")
            if alt_id not in {".", "?", "A"}:
                k += 1
                continue

            atom_name = get_tok(tokens, "_atom_site.auth_atom_id", "")
            if atom_name in {"", ".", "?"}:
                atom_name = get_tok(tokens, "_atom_site.label_atom_id", "")
            atom_name = atom_name.strip()
            if not atom_name:
                k += 1
                continue

            chain_id = get_tok(tokens, "_atom_site.auth_asym_id", "")
            if chain_id in {"", ".", "?"}:
                chain_id = get_tok(tokens, "_atom_site.label_asym_id", "")
            chain_id = chain_id.strip()
            if not chain_id:
                k += 1
                continue

            label_seq = parse_int(get_tok(tokens, "_atom_site.label_seq_id", ""))
            auth_seq = parse_int(get_tok(tokens, "_atom_site.auth_seq_id", ""))
            ins_code = get_tok(tokens, "_atom_site.pdbx_PDB_ins_code", "")
            if ins_code in {".", "?"}:
                ins_code = ""

            label_comp = get_tok(tokens, "_atom_site.label_comp_id", "")
            residue_key = (
                model_num,
                chain_id,
                str(label_seq) if label_seq is not None else ".",
                str(auth_seq) if auth_seq is not None else ".",
                f"{ins_code}:{label_comp}",
            )
            if residue_key not in chain_res_seen:
                chain_res_seen[residue_key] = next_ord
                next_ord += 1
            ord_seq = chain_res_seen[residue_key]

            x = float(get_tok(tokens, "_atom_site.Cartn_x", "nan"))
            y = float(get_tok(tokens, "_atom_site.Cartn_y", "nan"))
            z = float(get_tok(tokens, "_atom_site.Cartn_z", "nan"))

            atoms.append(
                AtomRecord(
                    chain_id=chain_id,
                    atom_name=atom_name,
                    residue_name=get_tok(tokens, "_atom_site.auth_comp_id", "")
                    or get_tok(tokens, "_atom_site.label_comp_id", "")
                    or "UNK",
                    record_type=group if group in {"ATOM", "HETATM"} else "ATOM",
                    element=(get_tok(tokens, "_atom_site.type_symbol", "") or atom_name[:1]).strip().upper()[:2],
                    x=x,
                    y=y,
                    z=z,
                    label_seq_id=label_seq,
                    auth_seq_id=auth_seq,
                    ordinal_seq_id=ord_seq,
                    model_num=model_num,
                    ins_code=ins_code,
                )
            )
            k += 1
        i = k
    return atoms


def parse_pdb_atoms(path: Path) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    chain_res_seen: Dict[Tuple[int, str, int, str], int] = {}
    next_ord = 1
    for raw in path.read_text(errors="ignore").splitlines():
        rec = raw[:6].strip()
        if rec not in {"ATOM", "HETATM"}:
            continue
        alt = raw[16:17].strip()
        if alt not in {"", "A"}:
            continue
        atom_name = raw[12:16].strip()
        chain_id = raw[21:22].strip() or "_"
        resseq = parse_int(raw[22:26].strip()) or 0
        ins = raw[26:27].strip()
        model_num = 1

        key = (model_num, chain_id, resseq, ins)
        if key not in chain_res_seen:
            chain_res_seen[key] = next_ord
            next_ord += 1

        x = float(raw[30:38].strip())
        y = float(raw[38:46].strip())
        z = float(raw[46:54].strip())
        atoms.append(
            AtomRecord(
                chain_id=chain_id,
                atom_name=atom_name,
                residue_name=(raw[17:20].strip() or "UNK"),
                record_type="HETATM" if rec == "HETATM" else "ATOM",
                element=(raw[76:78].strip() or atom_name[:1]).upper()[:2],
                x=x,
                y=y,
                z=z,
                label_seq_id=resseq,
                auth_seq_id=resseq,
                ordinal_seq_id=chain_res_seen[key],
                model_num=model_num,
                ins_code=ins,
            )
        )
    return atoms


def load_atoms(path: Path) -> List[AtomRecord]:
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return parse_mmcif_atoms(path)
    if suffix == ".pdb":
        return parse_pdb_atoms(path)
    raise ValueError(f"Unsupported structure extension: {path}")


def build_residue_atom_map(
    atoms: List[AtomRecord],
    chain_id: str,
    index_mode: str,
    atom_names: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for a in atoms:
        if a.chain_id != chain_id:
            continue
        if a.atom_name not in atom_names:
            continue
        if index_mode == "label":
            idx = a.label_seq_id
        elif index_mode == "auth":
            idx = a.auth_seq_id
        elif index_mode == "ordinal":
            idx = a.ordinal_seq_id
        else:
            idx = a.label_seq_id if a.label_seq_id is not None else a.auth_seq_id
            if idx is None:
                idx = a.ordinal_seq_id
        if idx is None:
            continue
        if idx not in out:
            out[idx] = {}
        if a.atom_name not in out[idx]:
            out[idx][a.atom_name] = np.array([a.x, a.y, a.z], dtype=np.float64)
    return out


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if mobile.shape != target.shape:
        raise ValueError("kabsch input shape mismatch")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("kabsch expects Nx3 arrays")
    m_cent = mobile.mean(axis=0)
    t_cent = target.mean(axis=0)
    m0 = mobile - m_cent
    t0 = target - t_cent
    h = m0.T @ t0
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    aligned = m0 @ r + t_cent
    t = t_cent - (m_cent @ r)
    rmsd = math.sqrt(float(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rmsd, aligned, r, t


def atom_index(atom: AtomRecord, mode: str) -> Optional[int]:
    if mode == "label":
        return atom.label_seq_id
    if mode == "auth":
        return atom.auth_seq_id
    if mode == "ordinal":
        return atom.ordinal_seq_id
    idx = atom.label_seq_id if atom.label_seq_id is not None else atom.auth_seq_id
    if idx is None:
        idx = atom.ordinal_seq_id
    return idx


def transform_atoms(atoms: List[AtomRecord], r: np.ndarray, t: np.ndarray) -> List[AtomRecord]:
    out: List[AtomRecord] = []
    for a in atoms:
        xyz = np.array([a.x, a.y, a.z], dtype=np.float64)
        x2, y2, z2 = (xyz @ r + t).tolist()
        out.append(
            AtomRecord(
                chain_id=a.chain_id,
                atom_name=a.atom_name,
                residue_name=a.residue_name,
                record_type=a.record_type,
                element=a.element,
                x=float(x2),
                y=float(y2),
                z=float(z2),
                label_seq_id=a.label_seq_id,
                auth_seq_id=a.auth_seq_id,
                ordinal_seq_id=a.ordinal_seq_id,
                model_num=a.model_num,
                ins_code=a.ins_code,
            )
        )
    return out


def select_residue_atoms(
    atoms: List[AtomRecord],
    chain_id: str,
    index_mode: str,
    residue_positions: List[int],
    atom_names: Optional[List[str]] = None,
) -> List[AtomRecord]:
    wanted = set(residue_positions)
    atom_name_set = set(atom_names) if atom_names else None
    out: List[AtomRecord] = []
    for a in atoms:
        if a.chain_id != chain_id:
            continue
        idx = atom_index(a, index_mode)
        if idx is None or idx not in wanted:
            continue
        if atom_name_set is not None and a.atom_name not in atom_name_set:
            continue
        out.append(a)
    return out


def _pdb_atom_name_field(name: str) -> str:
    n = (name or "").strip()
    if len(n) >= 4:
        return n[:4]
    if len(n) == 1:
        return f" {n}  "
    if len(n) == 2:
        return f" {n} "
    return n.rjust(4)


def write_pdb(atoms: List[AtomRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        serial = 1
        for a in atoms:
            rec = "HETATM" if a.record_type == "HETATM" else "ATOM"
            atom_name = _pdb_atom_name_field(a.atom_name)
            resn = (a.residue_name or "UNK")[:3].rjust(3)
            chain = (a.chain_id or "A")[0]
            resseq = a.auth_seq_id if a.auth_seq_id is not None else (
                a.label_seq_id if a.label_seq_id is not None else a.ordinal_seq_id
            )
            if resseq is None:
                resseq = 1
            ins = (a.ins_code or "")[:1]
            elem = (a.element or (a.atom_name[:1] if a.atom_name else "C")).strip().upper()[:2].rjust(2)
            line = (
                f"{rec:<6}{serial:>5} {atom_name}"
                f" {resn} {chain}{int(resseq):>4}{ins:1}   "
                f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
                f"{1.00:>6.2f}{0.00:>6.2f}          {elem:>2}\n"
            )
            f.write(line)
            serial += 1
        f.write("END\n")


def relabel_chain(atoms: List[AtomRecord], chain_id: str) -> List[AtomRecord]:
    out: List[AtomRecord] = []
    cid = (chain_id or "A")[:1]
    for a in atoms:
        out.append(
            AtomRecord(
                chain_id=cid,
                atom_name=a.atom_name,
                residue_name=a.residue_name,
                record_type=a.record_type,
                element=a.element,
                x=a.x,
                y=a.y,
                z=a.z,
                label_seq_id=a.label_seq_id,
                auth_seq_id=a.auth_seq_id,
                ordinal_seq_id=a.ordinal_seq_id,
                model_num=a.model_num,
                ins_code=a.ins_code,
            )
        )
    return out


def make_yaml_with_binder_sequence(template_yaml: Path, out_yaml: Path, new_seq: str) -> None:
    in_binder = False
    in_protein = False
    with template_yaml.open() as fin, out_yaml.open("w") as fout:
        for line in fin:
            stripped = line.strip()
            if stripped.startswith("- protein:"):
                in_protein = True
                in_binder = False
                fout.write(line)
                continue
            if in_protein and stripped.startswith("- ") and not stripped.startswith("- protein:"):
                in_protein = False
                in_binder = False
                fout.write(line)
                continue
            if stripped.startswith("id:"):
                cid = stripped.split("id:", 1)[1].strip().strip("'\"")
                in_binder = cid == "A"
                fout.write(line)
                continue
            if in_binder and stripped.startswith("sequence:"):
                indent = line.split("sequence:")[0]
                fout.write(f"{indent}sequence: {new_seq}\n")
                continue
            fout.write(line)


def find_boltz_pred_structure(out_dir: Path) -> Path:
    roots = sorted(p for p in out_dir.glob("boltz_results_*") if p.is_dir())
    if not roots:
        raise RuntimeError(f"No boltz_results_* found in {out_dir}")
    root = roots[0]
    pred_dirs = sorted(p for p in (root / "predictions").glob("*/*") if p.is_dir())
    if not pred_dirs:
        raise RuntimeError(f"No predictions leaf found in {root}")
    leaf = pred_dirs[0]
    structs = sorted(list(leaf.glob("*.cif")) + list(leaf.glob("*.pdb")))
    if not structs:
        raise RuntimeError(f"No structure file in {leaf}")
    return structs[0]


def ensure_reference_structure(args: argparse.Namespace, run_root: Path) -> Path:
    if args.reference_structure:
        p = Path(args.reference_structure).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Reference structure not found: {p}")
        return p

    if not args.predict_reference_from_template:
        raise ValueError(
            "Provide --reference-structure, or set --predict-reference-from-template "
            "with --reference-template-yaml and --reference-sequence."
        )

    if not args.reference_template_yaml or not args.reference_sequence:
        raise ValueError(
            "--predict-reference-from-template requires --reference-template-yaml "
            "and --reference-sequence."
        )

    ref_dir = (run_root / "_reference_prediction").resolve()
    ref_dir.mkdir(parents=True, exist_ok=True)
    input_yaml = ref_dir / "reference_input.yaml"
    make_yaml_with_binder_sequence(
        Path(args.reference_template_yaml).resolve(),
        input_yaml,
        args.reference_sequence.strip(),
    )

    out_dir = ref_dir / "boltz"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.boltz_cli, "predict", str(input_yaml), "--out_dir", str(out_dir)]
    if args.boltz_extra.strip():
        cmd.extend(shlex.split(args.boltz_extra))
    else:
        cmd.extend(
            [
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
        )
    print(f"[info] Predicting reference structure with: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    struct = find_boltz_pred_structure(out_dir).resolve()
    print(f"[info] Reference structure: {struct}")
    return struct


def load_motif_rows(run_root: Path, motif_csv: Optional[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if motif_csv and motif_csv.is_file():
        with motif_csv.open() as f:
            rows.extend(list(csv.DictReader(f)))
        return rows

    root_csv = run_root / "motif_positions_by_cycle.csv"
    if root_csv.is_file():
        with root_csv.open() as f:
            rows.extend(list(csv.DictReader(f)))
        return rows

    for p in sorted(run_root.glob("run_*/motif_positions_by_cycle.csv")):
        with p.open() as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def score_row(
    row: Dict[str, str],
    run_root: Path,
    ref_res_map: Dict[int, Dict[str, np.ndarray]],
    ref_atoms_all: List[AtomRecord],
    reference_chain: str,
    reference_index_mode: str,
    design_chain: str,
    design_index_mode: str,
    atom_names: List[str],
    save_aligned_structures: bool = False,
    aligned_structures_dir: Optional[Path] = None,
    export_motif_backbone_only: bool = False,
) -> Dict[str, str]:
    out: Dict[str, str] = {
        "design": row.get("design", ""),
        "cycle": row.get("cycle", ""),
        "status": "error",
        "message": "",
        "structure_path": "",
        "n_atoms_aligned": "0",
        "global_rmsd": "",
        "segment_rmsd_global": "",
        "segment_rmsd_local": "",
        "aligned_design_pdb": "",
        "aligned_reference_motif_pdb": "",
        "aligned_design_motif_pdb": "",
        "aligned_overlay_motif_pdb": "",
    }
    try:
        design = int(str(row.get("design", "")).strip())
        cycle = int(str(row.get("cycle", "")).strip())
    except Exception:
        out["message"] = "invalid design/cycle"
        return out

    struct_path = find_design_structure(run_root, design, cycle)
    if struct_path is None:
        out["status"] = "missing_structure"
        out["message"] = "model_0.cif/pdb not found"
        return out
    out["structure_path"] = str(struct_path)

    source_ranges = parse_ranges(row.get("source_motif_ranges_1based", ""))
    design_ranges = parse_ranges(row.get("motif_ranges_1based", ""))
    if not source_ranges or not design_ranges:
        out["message"] = "missing motif ranges"
        return out
    if len(source_ranges) != len(design_ranges):
        out["message"] = "source/design segment count mismatch"
        return out

    src_positions = expand_ranges(source_ranges)
    des_positions = expand_ranges(design_ranges)
    if len(src_positions) != len(des_positions):
        out["message"] = "source/design motif lengths differ"
        return out

    listed_positions = parse_positions(row.get("motif_positions_1based", ""))
    if listed_positions and listed_positions != des_positions:
        out["message"] = "motif_positions_1based does not match motif_ranges_1based"
        return out

    atoms = load_atoms(struct_path)
    des_res_map = build_residue_atom_map(
        atoms,
        chain_id=design_chain,
        index_mode=design_index_mode,
        atom_names=atom_names,
    )

    mobile_coords: List[np.ndarray] = []
    target_coords: List[np.ndarray] = []
    segment_slices: List[Tuple[int, int, str]] = []

    cursor = 0
    for (src_a, src_b), (des_a, des_b) in zip(source_ranges, design_ranges):
        seg_src = list(range(src_a, src_b + 1))
        seg_des = list(range(des_a, des_b + 1))
        if len(seg_src) != len(seg_des):
            out["message"] = f"segment length mismatch {src_a}-{src_b} vs {des_a}-{des_b}"
            return out

        start = len(mobile_coords)
        for s_pos, d_pos in zip(seg_src, seg_des):
            r_atoms = ref_res_map.get(s_pos)
            d_atoms = des_res_map.get(d_pos)
            if r_atoms is None:
                out["message"] = f"reference residue {s_pos} missing"
                return out
            if d_atoms is None:
                out["message"] = f"design residue {d_pos} missing"
                return out
            for an in atom_names:
                if an not in r_atoms or an not in d_atoms:
                    out["message"] = f"missing atom {an} at src {s_pos} or des {d_pos}"
                    return out
                target_coords.append(r_atoms[an])
                mobile_coords.append(d_atoms[an])
        stop = len(mobile_coords)
        segment_slices.append((start, stop, f"{src_a}-{src_b}->{des_a}-{des_b}"))
        cursor = stop

    if len(mobile_coords) < 3:
        out["message"] = "not enough atoms for alignment"
        return out

    mobile = np.vstack(mobile_coords)
    target = np.vstack(target_coords)
    global_rmsd, aligned_mobile, rot, shift = kabsch_align(mobile, target)

    seg_global_parts: List[str] = []
    seg_local_parts: List[str] = []
    for a, b, label in segment_slices:
        seg_global = math.sqrt(float(np.mean(np.sum((aligned_mobile[a:b] - target[a:b]) ** 2, axis=1))))
        seg_local, _, _, _ = kabsch_align(mobile[a:b], target[a:b])
        seg_global_parts.append(f"{label}:{seg_global:.4f}")
        seg_local_parts.append(f"{label}:{seg_local:.4f}")

    out["status"] = "ok"
    out["message"] = ""
    out["n_atoms_aligned"] = str(mobile.shape[0])
    out["global_rmsd"] = f"{global_rmsd:.4f}"
    out["segment_rmsd_global"] = ";".join(seg_global_parts)
    out["segment_rmsd_local"] = ";".join(seg_local_parts)

    if save_aligned_structures:
        if aligned_structures_dir is None:
            raise ValueError("aligned_structures_dir is required when save_aligned_structures=True")
        aligned_structures_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"run_{design:03d}_cycle_{cycle:02d}"

        # Full aligned design structure (all atoms transformed with global motif fit).
        aligned_design_atoms = transform_atoms(atoms, rot, shift)
        aligned_design_pdb = aligned_structures_dir / f"{prefix}_design_aligned.pdb"
        write_pdb(aligned_design_atoms, aligned_design_pdb)
        out["aligned_design_pdb"] = str(aligned_design_pdb)

        # Motif-only exports for direct figure overlays.
        motif_atom_filter = atom_names if export_motif_backbone_only else None
        ref_motif_atoms = select_residue_atoms(
            ref_atoms_all,
            chain_id=reference_chain,
            index_mode=reference_index_mode,
            residue_positions=src_positions,
            atom_names=motif_atom_filter,
        )
        design_motif_atoms = select_residue_atoms(
            aligned_design_atoms,
            chain_id=design_chain,
            index_mode=design_index_mode,
            residue_positions=des_positions,
            atom_names=motif_atom_filter,
        )
        ref_motif_pdb = aligned_structures_dir / f"{prefix}_reference_motif.pdb"
        design_motif_pdb = aligned_structures_dir / f"{prefix}_design_motif_aligned.pdb"
        overlay_pdb = aligned_structures_dir / f"{prefix}_motif_overlay.pdb"
        write_pdb(ref_motif_atoms, ref_motif_pdb)
        write_pdb(design_motif_atoms, design_motif_pdb)

        # Overlay with forced distinct chains for straightforward visualization.
        overlay_atoms = relabel_chain(ref_motif_atoms, "R") + relabel_chain(design_motif_atoms, "D")
        write_pdb(overlay_atoms, overlay_pdb)
        out["aligned_reference_motif_pdb"] = str(ref_motif_pdb)
        out["aligned_design_motif_pdb"] = str(design_motif_pdb)
        out["aligned_overlay_motif_pdb"] = str(overlay_pdb)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score motif scaffolding quality by motif-focused structural alignment."
    )
    p.add_argument("--run-root", required=True, help="Run output root, e.g. output/heme_motif_test")
    p.add_argument("--motif-csv", default="", help="Optional motif_positions_by_cycle.csv path")

    p.add_argument("--reference-structure", default="", help="Reference PDB/CIF for original motif")
    p.add_argument(
        "--predict-reference-from-template",
        action="store_true",
        help="Predict reference structure with Boltz from template YAML + reference sequence.",
    )
    p.add_argument("--reference-template-yaml", default="", help="Template YAML for Boltz reference prediction")
    p.add_argument("--reference-sequence", default="", help="Chain A sequence for reference prediction")
    p.add_argument("--boltz-cli", default="boltz", help="Boltz CLI path/name")
    p.add_argument("--boltz-extra", default="", help="Extra args for boltz predict during reference prediction")

    p.add_argument("--reference-chain", default="A", help="Chain id in reference structure")
    p.add_argument("--design-chain", default="A", help="Chain id in design structures")
    p.add_argument(
        "--reference-index-mode",
        choices=["auto", "label", "auth", "ordinal"],
        default="auto",
        help="Residue index mode for reference structure",
    )
    p.add_argument(
        "--design-index-mode",
        choices=["auto", "label", "auth", "ordinal"],
        default="auto",
        help="Residue index mode for design structures",
    )
    p.add_argument(
        "--atom-names",
        default="N,CA,C,O",
        help="Comma-separated atom names to align (default backbone: N,CA,C,O)",
    )
    p.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: <run-root>/motif_scaffold_quality.csv)",
    )
    p.add_argument(
        "--save-aligned-structures",
        action="store_true",
        help="Write aligned PDB structures for each scored design/cycle.",
    )
    p.add_argument(
        "--aligned-structures-dir",
        default="",
        help="Directory for aligned structure exports (default: <run-root>/motif_aligned_structures).",
    )
    p.add_argument(
        "--export-motif-backbone-only",
        action="store_true",
        help="If set, motif exports include only atom names from --atom-names (default exports all motif atoms).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")

    motif_csv = Path(args.motif_csv).expanduser().resolve() if args.motif_csv else None
    rows = load_motif_rows(run_root, motif_csv)
    if not rows:
        raise SystemExit("No motif rows found. Provide --motif-csv or ensure run contains motif_positions CSV files.")

    atom_names = [x.strip() for x in args.atom_names.split(",") if x.strip()]
    if not atom_names:
        raise SystemExit("--atom-names produced an empty set")

    ref_path = ensure_reference_structure(args, run_root)
    ref_atoms = load_atoms(ref_path)
    ref_res_map = build_residue_atom_map(
        ref_atoms,
        chain_id=args.reference_chain,
        index_mode=args.reference_index_mode,
        atom_names=atom_names,
    )
    if not ref_res_map:
        raise SystemExit(
            f"No residues loaded for reference chain '{args.reference_chain}' "
            f"from {ref_path} with index mode '{args.reference_index_mode}'."
        )

    aligned_dir: Optional[Path] = None
    if args.save_aligned_structures:
        aligned_dir = (
            Path(args.aligned_structures_dir).expanduser().resolve()
            if args.aligned_structures_dir
            else (run_root / "motif_aligned_structures").resolve()
        )
        aligned_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, str]] = []
    ok_count = 0
    for row in rows:
        scored = score_row(
            row=row,
            run_root=run_root,
            ref_res_map=ref_res_map,
            ref_atoms_all=ref_atoms,
            reference_chain=args.reference_chain,
            reference_index_mode=args.reference_index_mode,
            design_chain=args.design_chain,
            design_index_mode=args.design_index_mode,
            atom_names=atom_names,
            save_aligned_structures=args.save_aligned_structures,
            aligned_structures_dir=aligned_dir,
            export_motif_backbone_only=args.export_motif_backbone_only,
        )
        scored["reference_structure"] = str(ref_path)
        scored["reference_chain"] = args.reference_chain
        scored["design_chain"] = args.design_chain
        out_rows.append(scored)
        if scored["status"] == "ok":
            ok_count += 1

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else run_root / "motif_scaffold_quality.csv"
    fieldnames = [
        "design",
        "cycle",
        "status",
        "message",
        "structure_path",
        "reference_structure",
        "reference_chain",
        "design_chain",
        "n_atoms_aligned",
        "global_rmsd",
        "segment_rmsd_global",
        "segment_rmsd_local",
        "aligned_design_pdb",
        "aligned_reference_motif_pdb",
        "aligned_design_motif_pdb",
        "aligned_overlay_motif_pdb",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[done] scored_rows={len(out_rows)} ok={ok_count} out={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
