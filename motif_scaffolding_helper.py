#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Motif:
    start_pos: int  # 0-based inclusive
    end_pos: int  # 0-based inclusive
    shifted_start: int | None = None  # 0-based inclusive
    shifted_end: int | None = None  # 0-based inclusive

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos + 1


def parse_shifted_motifs_arg(motif_arg_str: str) -> List[Motif]:
    """
    Parse motifs from either:
      - JSON list of dicts: [{"start_pos": 10, "end_pos": 20}, ...] (0-based assumed)
      - range string: "31-45,63-106" (1-based user input -> convert to 0-based)
    """
    raw = (motif_arg_str or "").strip()
    if not raw:
        return []

    try:
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError("JSON motif specification must be a list of objects")
        motifs: List[Motif] = []
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(f"Motif #{i + 1} must be an object")
            if "start_pos" not in item or "end_pos" not in item:
                raise ValueError(f"Motif #{i + 1} missing start_pos/end_pos")
            start = int(item["start_pos"])
            end = int(item["end_pos"])
            motifs.append(Motif(start, end))
        return motifs
    except json.JSONDecodeError:
        pass

    motifs = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        if "-" not in tok:
            raise ValueError(f"Invalid motif range token '{tok}', expected start-end")
        a, b = tok.split("-", 1)
        start = int(a.strip()) - 1
        end = int(b.strip()) - 1
        motifs.append(Motif(start, end))
    return motifs


def parse_fixed_positions_arg(fixed_arg_str: str) -> List[int]:
    """
    Parse fixed positions as 1-based indices from comma-separated values.
    """
    raw = (fixed_arg_str or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 1:
            raise ValueError(f"Fixed positions must be 1-based positive integers, got {v}")
        out.append(v)
    return out


def validate_motifs(motifs: List[Motif], source_sequence: str) -> None:
    if not motifs:
        raise ValueError("No motifs parsed from --motif-positions")
    n = len(source_sequence)
    prev_end = -1
    for i, m in enumerate(motifs):
        if m.start_pos < 0 or m.end_pos < 0:
            raise ValueError(
                f"Motif #{i + 1} has negative coordinates: {m.start_pos}-{m.end_pos}"
            )
        if m.start_pos > m.end_pos:
            raise ValueError(
                f"Motif #{i + 1} has start_pos > end_pos: {m.start_pos}-{m.end_pos}"
            )
        if m.end_pos >= n:
            raise ValueError(
                f"Motif #{i + 1} out of bounds for source sequence length {n}: "
                f"{m.start_pos}-{m.end_pos}"
            )
        if m.start_pos <= prev_end:
            raise ValueError(
                "Motifs must be in ascending non-overlapping order; "
                f"motif #{i + 1} starts at {m.start_pos}, previous ended at {prev_end}"
            )
        prev_end = m.end_pos


def min_length_required(motifs: List[Motif], gap_between_motifs: int) -> int:
    total_motif_len = sum(m.length for m in motifs)
    internal_gap = max(0, len(motifs) - 1) * gap_between_motifs
    return total_motif_len + max(1, internal_gap)


def pick_binder_length(min_len: int, max_len: int, motifs: List[Motif], gap_between_motifs: int, rng: random.Random) -> int:
    if max_len < min_len:
        raise ValueError("--binder-max-len must be >= --binder-min-len")
    needed = min_length_required(motifs, gap_between_motifs)
    lo = max(min_len, needed)
    if max_len < lo:
        raise ValueError(
            f"Binder length range [{min_len}, {max_len}] is too short for motifs; "
            f"need at least {needed} residues."
        )
    return rng.randint(lo, max_len)


def randomize_motif_positions(motifs: List[Motif], length: int, gap_between_motifs: int, rng: random.Random) -> List[Motif]:
    """
    Place motifs in-order into a scaffold of given length.
    - Preserves motif order
    - Internal gaps >= gap_between_motifs
    - Terminal gaps unconstrained
    """
    motif_lengths = [m.length for m in motifs]
    total_motif_len = sum(motif_lengths)
    if length <= total_motif_len:
        raise ValueError(f"length={length} must exceed total_motif_len={total_motif_len}")

    num_motifs = len(motifs)
    min_required_gap = max(0, num_motifs - 1) * gap_between_motifs
    total_gap = length - total_motif_len
    if total_gap < min_required_gap:
        raise ValueError(
            f"Need at least {min_required_gap} gap residues, only {total_gap} available."
        )

    num_gaps = num_motifs + 1
    gaps = [0] * num_gaps
    for i in range(1, num_motifs):
        gaps[i] = gap_between_motifs

    remaining = total_gap - sum(gaps)
    if remaining > 0:
        cuts = sorted(rng.sample(range(remaining + num_gaps - 1), num_gaps - 1))
        extras = []
        prev = -1
        for c in cuts:
            extras.append(c - prev - 1)
            prev = c
        extras.append(remaining + num_gaps - 1 - prev - 1)
        gaps = [g + e for g, e in zip(gaps, extras)]

    out: List[Motif] = []
    pos = gaps[0]
    for motif, L, gap_after in zip(motifs, motif_lengths, gaps[1:]):
        out.append(
            Motif(
                start_pos=motif.start_pos,
                end_pos=motif.end_pos,
                shifted_start=pos,
                shifted_end=pos + L - 1,
            )
        )
        pos = pos + L + gap_after
    return out


def build_initial_sequence(source_sequence: str, shifted_motifs: List[Motif], binder_length: int) -> str:
    seq = ["X"] * binder_length
    for m in shifted_motifs:
        assert m.shifted_start is not None and m.shifted_end is not None
        motif_seq = source_sequence[m.start_pos : m.end_pos + 1]
        seq[m.shifted_start : m.shifted_end + 1] = list(motif_seq)
    return "".join(seq)


def build_fixed_residues(shifted_motifs: List[Motif], fixed_positions_1b: List[int]) -> str:
    fixed = []
    unmatched = []
    if fixed_positions_1b:
        for fixed_pos in fixed_positions_1b:
            hit = False
            for m in shifted_motifs:
                assert m.shifted_start is not None and m.shifted_end is not None
                motif_start_1b = m.start_pos + 1
                motif_end_1b = m.end_pos + 1
                if motif_start_1b <= fixed_pos <= motif_end_1b:
                    pdb_pos_1b = m.shifted_start + fixed_pos - m.start_pos
                    fixed.append(pdb_pos_1b)
                    hit = True
            if not hit:
                unmatched.append(fixed_pos)
        if unmatched:
            raise ValueError(
                "--motif-fixed-positions contained positions outside all motifs: "
                + ",".join(str(x) for x in unmatched)
            )
    else:
        for m in shifted_motifs:
            assert m.shifted_start is not None and m.shifted_end is not None
            fixed.extend(range(m.shifted_start + 1, m.shifted_end + 2))

    dedup = sorted(set(fixed))
    return " ".join(f"A{p}" for p in dedup)


def shifted_summary(shifted_motifs: List[Motif]) -> str:
    bits = []
    for m in shifted_motifs:
        assert m.shifted_start is not None and m.shifted_end is not None
        bits.append(
            f"orig:{m.start_pos + 1}-{m.end_pos + 1}->shift:{m.shifted_start + 1}-{m.shifted_end + 1}"
        )
    return "; ".join(bits)


def cmd_validate(args: argparse.Namespace) -> int:
    motifs = parse_shifted_motifs_arg(args.motif_positions)
    validate_motifs(motifs, args.source_sequence)
    _ = parse_fixed_positions_arg(args.motif_fixed_positions)
    if args.gap_between_motifs < 0:
        raise ValueError("--gap-between-motifs must be >= 0")
    if args.max_length < args.min_length:
        raise ValueError("--binder-max-len must be >= --binder-min-len")
    needed = min_length_required(motifs, args.gap_between_motifs)
    if args.max_length < needed:
        raise ValueError(
            f"Binder length range [{args.min_length}, {args.max_length}] is too short for motifs; "
            f"need at least {needed} residues."
        )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    motifs = parse_shifted_motifs_arg(args.motif_positions)
    validate_motifs(motifs, args.source_sequence)
    fixed_positions_1b = parse_fixed_positions_arg(args.motif_fixed_positions)
    if args.gap_between_motifs < 0:
        raise ValueError("--gap-between-motifs must be >= 0")

    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    length = pick_binder_length(
        min_len=args.min_length,
        max_len=args.max_length,
        motifs=motifs,
        gap_between_motifs=args.gap_between_motifs,
        rng=rng,
    )
    shifted = randomize_motif_positions(
        motifs=motifs,
        length=length,
        gap_between_motifs=args.gap_between_motifs,
        rng=rng,
    )
    init_seq = build_initial_sequence(args.source_sequence, shifted, length)
    fixed_residues = build_fixed_residues(shifted, fixed_positions_1b)

    payload = {
        "binder_length": length,
        "init_sequence": init_seq,
        "fixed_residues": fixed_residues,
        "shifted_motifs": [
            {
                "start_pos": m.start_pos,
                "end_pos": m.end_pos,
                "shifted_start": m.shifted_start,
                "shifted_end": m.shifted_end,
            }
            for m in shifted
        ],
        "shifted_summary": shifted_summary(shifted),
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Motif scaffolding helper for iProteinHunter"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, include_len: bool) -> None:
        sp.add_argument("--motif-positions", required=True, type=str)
        sp.add_argument("--source-sequence", required=True, type=str)
        sp.add_argument("--motif-fixed-positions", default="", type=str)
        sp.add_argument("--gap-between-motifs", default=8, type=int)
        if include_len:
            sp.add_argument("--min-length", required=True, type=int)
            sp.add_argument("--max-length", required=True, type=int)

    p_val = sub.add_parser("validate")
    add_common(p_val, include_len=True)
    p_val.set_defaults(func=cmd_validate)

    p_gen = sub.add_parser("generate")
    add_common(p_gen, include_len=True)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=cmd_generate)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
