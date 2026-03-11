#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${IPROTEINHUNTER_ROOT:-$SCRIPT_DIR}"

BOLTZ_VENV="${REPO_ROOT}/venvs/iProteinHunter_boltz"
LIGAND_VENV="${REPO_ROOT}/venvs/iProteinHunter_ligandmpnn"
INTELLIFOLD_VENV="${REPO_ROOT}/venvs/iProteinHunter_intellifold"
OPENFOLD_VENV="${REPO_ROOT}/venvs/iProteinHunter_openfold3_mlx"

BOLTZ_CLI="boltz"
INTELLIFOLD_CLI="intellifold"
OPENFOLD_CLI="run_openfold"

LIGANDMPNN_REPO="${REPO_ROOT}/src/LigandMPNN"
LIGANDMPNN_RUN="python run.py"
LIGANDMPNN_CHECKPOINT_SOLUBLE="${LIGANDMPNN_REPO}/model_params/solublempnn_v_48_020.pt"
LIGANDMPNN_CHECKPOINT_LIGAND="${LIGANDMPNN_REPO}/model_params/ligandmpnn_v_32_010_25.pt"

TEMPLATE_YAML="${REPO_ROOT}/examples/aCbx_bind.yaml"
BASE_RUN_ROOT="${REPO_ROOT}/output"

RUN_NAME="test_run"
PREDICTOR="boltz"
POST_PREDICTOR="none"
POST_MODE="all"
POST_IPTM_THRESHOLD="0.70"
POST_INCLUDE_CYCLE00=0

N_RUNS=3
N_CYCLES=5
CPU_ONLY=0

NO_PARALLEL=0
MAX_PARALLEL_USER="auto"
MPS_AWARE=1
MPS_MAX_PARALLEL="auto"
MPS_MEM_FRACTION="0.90"
MPS_CPU_CAP="12"
MEM_BUDGET_GB="auto"
MEM_SAFETY="0.95"

BINDER_MIN_LEN=65
BINDER_MAX_LEN=150
BINDER_PERCENT_X=50

HELIX_KILL=0
NEGATIVE_HELIX_CONSTANT="0.5"
LOOP_KILL="0"
UNK_PATCH_MODE="auto"
UNCONDITIONAL_MULTIMER_MODE=0
UNCONDITIONAL_MULTIMER_SIZE="3"
UNCONDITIONAL_MULTIMER_CHAINS="A,B,C"

LIGAND_TEMP_DEFAULT="0.10"
LIGAND_TEMP_CYCLE01="0.30"
LIGAND_BIAS_AA_DEFAULT=""
LIGAND_BIAS_AA_CYCLE01=""
INTELLIFOLD_MODEL="v2-flash"
BOLTZ_USE_POTENTIALS_MODE="auto"
BOLTZ_USE_POTENTIALS_DEFAULT=0

IPTM_THRESHOLD="0.80"

BOLTZ_EXTRA_FLAGS_DEFAULT=(
  "--accelerator" "gpu"
  "--devices" "1"
  "--use_msa_server"
  "--msa_server_url" "https://api.colabfold.com"
  "--msa_pairing_strategy" "greedy"
)

INTELLIFOLD_EXTRA_FLAGS_DEFAULT=(
  "--precision" "no"
  "--num_workers" "0"
  "--seed" "42"
  "--num_diffusion_samples" "1"
  "--override"
  "--use_msa_server"
  "--msa_pairing_strategy" "greedy"
)

OPENFOLD_EXTRA_FLAGS_DEFAULT=(
  "--num_diffusion_samples" "1"
  "--num_model_seeds" "1"
)

LIGANDMPNN_EXTRA_FLAGS_DEFAULT=(
  "--seed" "111"
  "--chains_to_design" "A"
)

LIGANDMPNN_MODEL_FLAGS=()
LIGANDMPNN_MODEL_LABEL="auto"

BOLTZ_EXTRA_CLI_STRING=""
INTELLIFOLD_EXTRA_CLI_STRING=""
OPENFOLD_EXTRA_CLI_STRING=""
LIGAND_EXTRA_CLI_STRING=""

usage() {
  cat <<EOF2
Usage: $(basename "$0") [options]

Core:
  --predictor TOOL                 boltz | intellifold | openfold-3-mlx
  --post-predictor LIST            none | TOOL[,TOOL] (default: ${POST_PREDICTOR})
  --post-mode MODE                 none | all | iptm (default: ${POST_MODE})
  --post-iptm-threshold T          default: ${POST_IPTM_THRESHOLD}
  --post-include-cycle00           include cycle_00 in post stage
  --run-name NAME                  default: ${RUN_NAME}
  --num-runs N                     default: ${N_RUNS}
  --num-opt-cycles N               optimization cycles after cycle_00 (default: ${N_CYCLES})
  --num-cycles N                   alias of --num-opt-cycles
  --model NAME                     IntelliFold model when used (default: ${INTELLIFOLD_MODEL})
  --template-yaml PATH             default: ${TEMPLATE_YAML}
  --out-root PATH                  default: ${BASE_RUN_ROOT}

Binder:
  --binder-min-len N               default: ${BINDER_MIN_LEN}
  --binder-max-len N               default: ${BINDER_MAX_LEN}
  --binder-percent-x P             default: ${BINDER_PERCENT_X}
                                   (OpenFold-3 uses A/N/G/H/F/S/Y spikes at this rate; others use X)

Design controls:
  --helix-kill
  --negative-helix-constant X      0..1 helix-kill strength (default: ${NEGATIVE_HELIX_CONSTANT})
  --loopkill X                     0..1 loop-kill strength (default: ${LOOP_KILL})
  --unconditional-trimer           alias of --unconditional-multimer-size 3
  --unconditional-multimer-size N  unconditional N-mer mode (A..), copy init seq to all chains
  --unk-patch-mode MODE            auto | ala | ala_gly | ala_gly_ser
  --ligand-temp-cycle1 T           default: ${LIGAND_TEMP_CYCLE01}
  --ligand-temp-cycle01 T          alias of --ligand-temp-cycle1
  --ligand-temp-other T            default: ${LIGAND_TEMP_DEFAULT}
  --ligand-temp T                  alias of --ligand-temp-other
  --mpnn-bias-aa-cycle1 STR        passed to LigandMPNN --bias_AA for cycle_00->01 redesign
  --mpnn-bias-aa-other STR         passed to LigandMPNN --bias_AA for later redesign cycles
  --boltz-use-potentials           force Boltz design to use potentials
  --boltz-no-potentials            force Boltz design to not use potentials

Filtering:
  --iptm-threshold T               default: ${IPTM_THRESHOLD}

Parallelism:
  --no-parallel
  --max-parallel N|auto            default: ${MAX_PARALLEL_USER}
  --mps-aware                      default on
  --no-mps-aware
  --mps-max-parallel N|auto        compatibility knob
  --mps-mem-fraction F             compatibility knob
  --mps-cpu-cap N                  compatibility knob
  --mem-budget-gb X|auto           compatibility knob
  --mem-safety S                   compatibility knob

Hardware:
  --cpu-only                       force CPU where supported

Extra flags passthrough:
  --boltz-extra "ARGS"
  --intellifold-extra "ARGS"
  --openfold-extra "ARGS"
  --ligand-extra "ARGS"

  -h, --help
EOF2
}

norm_predictor() {
  local raw
  raw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    boltz) echo "boltz" ;;
    intellifold) echo "intellifold" ;;
    openfold-3-mlx|openfold3|openfold|of3) echo "openfold-3-mlx" ;;
    none|"") echo "none" ;;
    *) return 1 ;;
  esac
}

safe_predictor_name() {
  local p
  p="$(norm_predictor "$1")" || return 1
  case "$p" in
    boltz) echo "boltz" ;;
    intellifold) echo "intellifold" ;;
    openfold-3-mlx) echo "openfold3" ;;
    none) echo "none" ;;
  esac
}

post_predictors_count() {
  if declare -p POST_PREDICTORS >/dev/null 2>&1; then
    echo "${#POST_PREDICTORS[@]}"
  else
    echo "0"
  fi
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictor) PREDICTOR="$2"; shift 2 ;;
    --post-predictor) POST_PREDICTOR="$2"; shift 2 ;;
    --post-mode) POST_MODE="$2"; shift 2 ;;
    --post-iptm-threshold) POST_IPTM_THRESHOLD="$2"; shift 2 ;;
    --post-include-cycle00) POST_INCLUDE_CYCLE00=1; shift 1 ;;

    --run-name) RUN_NAME="$2"; shift 2 ;;
    --num-runs) N_RUNS="$2"; shift 2 ;;
    --num-opt-cycles|--num-cycles) N_CYCLES="$2"; shift 2 ;;
    --model) INTELLIFOLD_MODEL="$2"; shift 2 ;;
    --template-yaml) TEMPLATE_YAML="$2"; shift 2 ;;
    --out-root) BASE_RUN_ROOT="$2"; shift 2 ;;

    --binder-min-len) BINDER_MIN_LEN="$2"; shift 2 ;;
    --binder-max-len) BINDER_MAX_LEN="$2"; shift 2 ;;
    --binder-percent-x) BINDER_PERCENT_X="$2"; shift 2 ;;

    --helix-kill) HELIX_KILL=1; shift 1 ;;
    --negative-helix-constant) NEGATIVE_HELIX_CONSTANT="$2"; shift 2 ;;
    --loopkill) LOOP_KILL="$2"; shift 2 ;;
    --unconditional-trimer) UNCONDITIONAL_MULTIMER_MODE=1; UNCONDITIONAL_MULTIMER_SIZE="3"; shift 1 ;;
    --unconditional-multimer-size) UNCONDITIONAL_MULTIMER_MODE=1; UNCONDITIONAL_MULTIMER_SIZE="$2"; shift 2 ;;
    --unk-patch-mode) UNK_PATCH_MODE="$2"; shift 2 ;;

    --ligand-temp-cycle1|--ligand-temp-cycle01) LIGAND_TEMP_CYCLE01="$2"; shift 2 ;;
    --ligand-temp-other|--ligand-temp) LIGAND_TEMP_DEFAULT="$2"; shift 2 ;;
    --mpnn-bias-aa-cycle1) LIGAND_BIAS_AA_CYCLE01="$2"; shift 2 ;;
    --mpnn-bias-aa-other) LIGAND_BIAS_AA_DEFAULT="$2"; shift 2 ;;
    --boltz-use-potentials) BOLTZ_USE_POTENTIALS_MODE="on"; shift 1 ;;
    --boltz-no-potentials) BOLTZ_USE_POTENTIALS_MODE="off"; shift 1 ;;

    --iptm-threshold) IPTM_THRESHOLD="$2"; shift 2 ;;

    --no-parallel) NO_PARALLEL=1; shift 1 ;;
    --max-parallel) MAX_PARALLEL_USER="$2"; shift 2 ;;
    --mps-aware) MPS_AWARE=1; shift 1 ;;
    --no-mps-aware) MPS_AWARE=0; shift 1 ;;
    --mps-max-parallel) MPS_MAX_PARALLEL="$2"; shift 2 ;;
    --mps-mem-fraction) MPS_MEM_FRACTION="$2"; shift 2 ;;
    --mps-cpu-cap) MPS_CPU_CAP="$2"; shift 2 ;;
    --mem-budget-gb) MEM_BUDGET_GB="$2"; shift 2 ;;
    --mem-safety) MEM_SAFETY="$2"; shift 2 ;;

    --cpu-only) CPU_ONLY=1; shift 1 ;;

    --boltz-extra) BOLTZ_EXTRA_CLI_STRING="$2"; shift 2 ;;
    --intellifold-extra) INTELLIFOLD_EXTRA_CLI_STRING="$2"; shift 2 ;;
    --openfold-extra) OPENFOLD_EXTRA_CLI_STRING="$2"; shift 2 ;;
    --ligand-extra) LIGAND_EXTRA_CLI_STRING="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

PREDICTOR="$(norm_predictor "$PREDICTOR")" || die "Unsupported --predictor: ${PREDICTOR}"

POST_PREDICTOR_RAW="${POST_PREDICTOR}"
POST_PREDICTORS=()
if [[ -n "${POST_PREDICTOR_RAW}" && "$(printf '%s' "${POST_PREDICTOR_RAW}" | tr '[:upper:]' '[:lower:]')" != "none" ]]; then
  IFS=',' read -r -a _tmp_post <<< "${POST_PREDICTOR_RAW}"
  for p in "${_tmp_post[@]}"; do
    p="$(echo "$p" | xargs)"
    [[ -z "$p" ]] && continue
    p="$(norm_predictor "$p")" || die "Unsupported --post-predictor entry: ${p}"
    [[ "$p" == "none" ]] && continue
    POST_PREDICTORS+=("$p")
  done
fi

case "${POST_MODE}" in
  none|all|iptm) : ;;
  *) die "--post-mode must be none, all, or iptm" ;;
esac

if [[ "${POST_MODE}" == "none" ]]; then
  POST_PREDICTORS=()
fi

python3 - "$N_RUNS" "$N_CYCLES" "$IPTM_THRESHOLD" "$POST_IPTM_THRESHOLD" "$MEM_SAFETY" "$LIGAND_TEMP_DEFAULT" "$LIGAND_TEMP_CYCLE01" "$NEGATIVE_HELIX_CONSTANT" "$LOOP_KILL" <<'PY'
import sys
ints = ["--num-runs", "--num-opt-cycles"]
for idx, name in enumerate(ints, start=1):
    try:
        v = int(sys.argv[idx])
    except Exception:
        raise SystemExit(f"{name} must be an integer")
    if v < 1:
        raise SystemExit(f"{name} must be >= 1")
for idx, name in ((3,"--iptm-threshold"),(4,"--post-iptm-threshold"),(5,"--mem-safety"),(6,"--ligand-temp-other"),(7,"--ligand-temp-cycle1")):
    try:
        float(sys.argv[idx])
    except Exception:
        raise SystemExit(f"{name} must be numeric")
try:
    nhc = float(sys.argv[8])
except Exception:
    raise SystemExit("--negative-helix-constant must be numeric")
if not (0.0 <= nhc <= 1.0):
    raise SystemExit("--negative-helix-constant must be between 0 and 1")
try:
    lkc = float(sys.argv[9])
except Exception:
    raise SystemExit("--loopkill must be numeric")
if not (0.0 <= lkc <= 1.0):
    raise SystemExit("--loopkill must be between 0 and 1")
PY

if [[ "${UNK_PATCH_MODE}" == "auto" ]]; then
  if [[ "${HELIX_KILL}" -eq 1 ]]; then
    UNK_PATCH_MODE="ala_gly_ser"
  else
    UNK_PATCH_MODE="ala"
  fi
fi
case "${UNK_PATCH_MODE}" in
  ala|ala_gly|ala_gly_ser) : ;;
  *) die "Invalid --unk-patch-mode: ${UNK_PATCH_MODE}" ;;
esac

if [[ "${UNCONDITIONAL_MULTIMER_MODE}" -eq 1 ]]; then
  python3 - "${UNCONDITIONAL_MULTIMER_SIZE}" <<'PY'
import sys
try:
    n = int(sys.argv[1])
except Exception:
    raise SystemExit("--unconditional-multimer-size must be an integer")
if n < 3 or n > 8:
    raise SystemExit("--unconditional-multimer-size must be between 3 and 8")
PY
  UNCONDITIONAL_MULTIMER_CHAINS="$(python3 - "${UNCONDITIONAL_MULTIMER_SIZE}" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(chr(ord("A")+i) for i in range(n)))
PY
)"
  # In unconditional N-mer mode redesign all designed chains each cycle.
  LIGANDMPNN_EXTRA_FLAGS_DEFAULT=(
    "--seed" "111"
    "--chains_to_design" "${UNCONDITIONAL_MULTIMER_CHAINS}"
  )
fi

BOLTZ_EXTRA_FLAGS=("${BOLTZ_EXTRA_FLAGS_DEFAULT[@]}")
if [[ -n "${BOLTZ_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  _arr=(${BOLTZ_EXTRA_CLI_STRING})
  BOLTZ_EXTRA_FLAGS+=("${_arr[@]}")
fi

INTELLIFOLD_EXTRA_FLAGS=("${INTELLIFOLD_EXTRA_FLAGS_DEFAULT[@]}" "--model" "${INTELLIFOLD_MODEL}")
if [[ -n "${INTELLIFOLD_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  _arr=(${INTELLIFOLD_EXTRA_CLI_STRING})
  INTELLIFOLD_EXTRA_FLAGS+=("${_arr[@]}")
fi

OPENFOLD_EXTRA_FLAGS=("${OPENFOLD_EXTRA_FLAGS_DEFAULT[@]}")
if [[ -n "${OPENFOLD_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  _arr=(${OPENFOLD_EXTRA_CLI_STRING})
  OPENFOLD_EXTRA_FLAGS+=("${_arr[@]}")
fi

LIGANDMPNN_EXTRA_FLAGS=("${LIGANDMPNN_EXTRA_FLAGS_DEFAULT[@]}")
if [[ -n "${LIGAND_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  _arr=(${LIGAND_EXTRA_CLI_STRING})
  LIGANDMPNN_EXTRA_FLAGS+=("${_arr[@]}")
fi

if [[ "${CPU_ONLY}" -eq 1 ]]; then
  _tmp_flags=()
  _skip_next=0
  for tok in "${BOLTZ_EXTRA_FLAGS[@]}"; do
    if [[ "${_skip_next}" -eq 1 ]]; then
      _skip_next=0
      continue
    fi
    if [[ "${tok}" == "--accelerator" || "${tok}" == "--devices" ]]; then
      _skip_next=1
      continue
    fi
    _tmp_flags+=("${tok}")
  done
  BOLTZ_EXTRA_FLAGS=("${_tmp_flags[@]}" "--accelerator" "cpu" "--devices" "1")
fi

[[ -d "${REPO_ROOT}" ]] || die "Repo root not found: ${REPO_ROOT}"
[[ -f "${TEMPLATE_YAML}" ]] || die "Template YAML not found: ${TEMPLATE_YAML}"
[[ -x "${BOLTZ_VENV}/bin/python" ]] || die "Boltz venv not found: ${BOLTZ_VENV}"
[[ -x "${LIGAND_VENV}/bin/python" ]] || die "LigandMPNN venv not found: ${LIGAND_VENV}"
[[ -x "${INTELLIFOLD_VENV}/bin/python" ]] || die "IntelliFold venv not found: ${INTELLIFOLD_VENV}"
[[ -x "${OPENFOLD_VENV}/bin/python" ]] || die "OpenFold venv not found: ${OPENFOLD_VENV}"
[[ -f "${LIGANDMPNN_REPO}/run.py" ]] || die "LigandMPNN run.py not found: ${LIGANDMPNN_REPO}/run.py"

HAS_SMALL_MOLECULE_LIGAND="$(python3 - "${TEMPLATE_YAML}" <<'PY'
import sys
path = sys.argv[1]
for raw in open(path):
    if raw.strip().startswith("- ligand:"):
        print("1")
        raise SystemExit(0)
print("0")
PY
)"

if [[ "${HAS_SMALL_MOLECULE_LIGAND}" == "1" ]]; then
  [[ -f "${LIGANDMPNN_CHECKPOINT_LIGAND}" ]] || die "LigandMPNN checkpoint not found: ${LIGANDMPNN_CHECKPOINT_LIGAND}"
  LIGANDMPNN_MODEL_FLAGS=(
    "--model_type" "ligand_mpnn"
    "--checkpoint_ligand_mpnn" "${LIGANDMPNN_CHECKPOINT_LIGAND}"
  )
  LIGANDMPNN_MODEL_LABEL="ligand_mpnn"
else
  [[ -f "${LIGANDMPNN_CHECKPOINT_SOLUBLE}" ]] || die "SolubleMPNN checkpoint not found: ${LIGANDMPNN_CHECKPOINT_SOLUBLE}"
  LIGANDMPNN_MODEL_FLAGS=(
    "--model_type" "soluble_mpnn"
    "--checkpoint_soluble_mpnn" "${LIGANDMPNN_CHECKPOINT_SOLUBLE}"
  )
  LIGANDMPNN_MODEL_LABEL="soluble_mpnn"
fi

mkdir -p "${BASE_RUN_ROOT}"

now_epoch() {
  python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
}

calc_duration() {
  python3 - "$1" "$2" <<'PY'
import sys
s=float(sys.argv[1]); e=float(sys.argv[2])
print(f"{max(0.0,e-s):.6f}")
PY
}

total_ram_bytes() { sysctl -n hw.memsize 2>/dev/null || echo "0"; }

default_mem_budget_mb() {
  local total_b
  total_b="$(total_ram_bytes)"
  python3 - "$total_b" <<'PY'
import sys
total=int(sys.argv[1])
budget=int(total*0.90/1024/1024)
print(max(1024, budget))
PY
}

floor_mul() {
  python3 - "$1" "$2" <<'PY'
import sys, math
a=float(sys.argv[1]); b=float(sys.argv[2])
print(int(math.floor(a*b)))
PY
}

float_ge() {
  python3 - "$1" "$2" <<'PY'
import sys
a=float(sys.argv[1]); b=float(sys.argv[2])
raise SystemExit(0 if a>=b else 1)
PY
}

is_float() {
  python3 - "$1" <<'PY'
import sys
try:
    float(sys.argv[1])
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

generate_random_binder_seq() {
  local min_len="$1"
  local max_len="$2"
  local percent_x="$3"
  local helix_kill="$4"
  local neg_helix_constant="$5"
  local loop_kill="${6:-0}"
  local predictor="${7:-}"
  python3 - "$min_len" "$max_len" "$percent_x" "$helix_kill" "$neg_helix_constant" "$loop_kill" "$predictor" <<'PY'
import sys, random
min_len, max_len, pct_x, helix_kill, neg_helix_constant, loop_kill, predictor = sys.argv[1:8]
min_len = int(min_len); max_len = int(max_len)
pct_x = float(pct_x); helix_kill = int(helix_kill)
neg_helix_constant = max(0.0, min(1.0, float(neg_helix_constant)))
loop_kill = max(0.0, min(1.0, float(loop_kill)))
if max_len < min_len:
    raise SystemExit("binder-max-len must be >= binder-min-len")
L = random.randint(min_len, max_len)
p_x = max(0.0, min(1.0, pct_x / 100.0))
n_x = max(0, min(L, int(round(L * p_x))))
AA_POOL = list("ADEFGHIKLMNPQRSTVWY")
HELIX_PRONE = set("AEKLMQ")
# Tuned so 0.5 reproduces approximately previous defaults:
# - global helix-prone downweight ~0.55
# - local i-4 helix suppression ~0.15
base_helix_penalty = max(0.10, 1.0 - 0.90 * neg_helix_constant)
local_helix_penalty = max(0.02, 1.0 - 1.70 * neg_helix_constant)
ser_boost = 1.0 + 1.0 * neg_helix_constant
proline_scale = max(0.0, 1.0 - loop_kill)

base_weights = []
for aa in AA_POOL:
    w = 1.0
    if helix_kill and aa in HELIX_PRONE:
        w *= base_helix_penalty
    if helix_kill and aa == "S":
        w *= ser_boost
    if aa == "P":
        w *= proline_scale
    base_weights.append(w)
seq = [None] * L
idx = list(range(L)); random.shuffle(idx)
x_positions = set(idx[:n_x])
of3_spike_pool = list("ANGHFSY")
for i in range(L):
    if i in x_positions:
        if predictor == "openfold-3-mlx":
            seq[i] = random.choice(of3_spike_pool)
        else:
            seq[i] = "X"
        continue
    if not helix_kill:
        seq[i] = random.choices(AA_POOL, weights=base_weights, k=1)[0]
        continue
    w = base_weights[:]
    if i >= 4 and seq[i-4] in HELIX_PRONE:
        for j, aa in enumerate(AA_POOL):
            if aa in HELIX_PRONE:
                w[j] *= local_helix_penalty
    seq[i] = random.choices(AA_POOL, weights=w, k=1)[0]
print("".join(seq))
PY
}

compute_loopkill_mpnn_bias() {
  local base_bias="${1:-}"
  local loop_kill="${2:-0}"
  python3 - "$base_bias" "$loop_kill" <<'PY'
import sys
base = sys.argv[1].strip()
lk = max(0.0, min(1.0, float(sys.argv[2])))
if lk <= 0.0:
    print(base)
    raise SystemExit(0)
if lk >= 1.0:
    # loopkill=1 uses omit_AA to remove proline entirely.
    print(base)
    raise SystemExit(0)
bias_p = -0.1 - 1.9 * lk
if base:
    print(f"{base},P:{bias_p:.6f}")
else:
    print(f"P:{bias_p:.6f}")
PY
}

make_yaml_with_binder_sequence() {
  local template_yaml="$1"
  local out_yaml="$2"
  local new_seq="$3"
  local target_msa_path="${4:-}"
  local multimer_mode="${5:-0}"
  local multimer_size="${6:-3}"
  python3 - "$template_yaml" "$out_yaml" "$new_seq" "$target_msa_path" "$multimer_mode" "$multimer_size" <<'PY'
import sys
from pathlib import Path
template, out, new_seq, msa_path, multimer_mode, multimer_size = sys.argv[1:7]
msa_path = msa_path or None
multimer_mode = int(multimer_mode)
multimer_size = int(multimer_size)

def parse_multimer_state(seq_in: str, n_chains: int):
    s = (seq_in or "").strip()
    if not s:
        raise SystemExit("Empty sequence state for multimer mode.")
    parts = [p.strip() for p in s.split(":")]
    parts = [p for p in parts if p]
    if not parts:
        raise SystemExit("Empty sequence state for multimer mode.")
    if len(parts) >= n_chains:
        return parts[:n_chains]
    fill = parts[-1]
    return parts + [fill] * (n_chains - len(parts))

if multimer_mode == 1:
    chain_ids = [chr(ord("A") + i) for i in range(multimer_size)]
    chain_seqs = parse_multimer_state(new_seq, multimer_size)
    out_path = Path(out)
    msa_dir = out_path.parent / "single_msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    def write_single_seq_a3m(path: Path, seq: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f">query\n{seq}\n>|||DUMMY||\n{seq}\n")
        return str(path)

    seq_to_msa = {}
    chain_rows = list(zip(chain_ids, chain_seqs))
    chain_to_msa = {}
    for cid, seq in chain_rows:
        if seq not in seq_to_msa:
            # Keep OpenFold-compatible MSA basename while still allowing multiple unique sequences.
            seq_idx = len(seq_to_msa) + 1
            msa_path = msa_dir / f"seq_{seq_idx}" / "uniref90_hits.a3m"
            seq_to_msa[seq] = write_single_seq_a3m(msa_path, seq)
        chain_to_msa[cid] = seq_to_msa[seq]

    with out_path.open("w") as fout:
        fout.write("sequences:\n")
        for cid, seq in chain_rows:
            fout.write("  - protein:\n")
            fout.write(f"      id: {cid}\n")
            fout.write(f"      sequence: {seq}\n")
            fout.write(f"      msa: {chain_to_msa[cid]}\n")
        fout.write("version: 1\n")
    raise SystemExit(0)

in_binder = False
in_protein = False
need_msa_insert = False

with open(template) as fin, open(out, "w") as fout:
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
            in_binder = (cid == "A")
            need_msa_insert = (msa_path is not None and cid != "A")
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
PY
}

extract_target_sequence_from_yaml() {
  local template_yaml="$1"
  python3 - "$template_yaml" <<'PY'
import sys
path = sys.argv[1]
cur=None
seqs={}
in_protein=False
for raw in open(path):
    s=raw.strip()
    if s.startswith("- protein:"):
        in_protein=True
        cur=None
        continue
    if not in_protein:
        continue
    if s.startswith("-") and not s.startswith("- protein:"):
        in_protein=False
        continue
    if s.startswith("id:"):
        cur=s.split(":",1)[1].strip().strip("'\"")
        continue
    if s.startswith("sequence:") and cur:
        seq=s.split(":",1)[1].strip().strip("'\"")
        seqs[cur]=seq
if "B" in seqs and seqs["B"]:
    print(seqs["B"])
else:
    for cid in sorted(seqs):
        if cid != "A" and seqs[cid]:
            print(seqs[cid])
            break
PY
}

extract_target_msa_from_yaml() {
  local template_yaml="$1"
  python3 - "$template_yaml" <<'PY'
import sys
path = sys.argv[1]
cur=None
in_protein=False
for raw in open(path):
    s=raw.strip()
    if s.startswith("- protein:"):
        in_protein=True
        cur=None
        continue
    if not in_protein:
        continue
    if s.startswith("-") and not s.startswith("- protein:"):
        in_protein=False
        continue
    if s.startswith("id:"):
        cur=s.split(":",1)[1].strip().strip("'\"")
        continue
    if s.startswith("msa:") and cur and cur != "A":
        val=s.split(":",1)[1].strip().strip("'\"")
        low=val.lower()
        if val and low not in {"empty", "none", "null"}:
            print(val)
            break
PY
}

template_has_boltz_partner() {
  local template_yaml="$1"
  python3 - "$template_yaml" <<'PY'
import sys
path = sys.argv[1]

kind = None
in_entry = False
has_partner = False

for raw in open(path):
    s = raw.strip()
    if s.startswith("- protein:"):
        kind = "protein"
        in_entry = True
        continue
    if s.startswith("- ligand:"):
        # For binder-vs-ligand cases, ligand is the partner.
        print("1")
        raise SystemExit(0)

    if in_entry and s.startswith("id:"):
        cid = s.split(":", 1)[1].strip().strip("'\"")
        if kind == "protein" and cid and cid != "A":
            has_partner = True
        continue

    if in_entry and s.startswith("- ") and not (s.startswith("- protein:") or s.startswith("- ligand:")):
        in_entry = False

print("1" if has_partner else "0")
PY
}

template_has_small_molecule_ligand() {
  local template_yaml="$1"
  python3 - "$template_yaml" <<'PY'
import sys
path = sys.argv[1]
for raw in open(path):
    if raw.strip().startswith("- ligand:"):
        print("1")
        raise SystemExit(0)
print("0")
PY
}

extract_target_chain_id_from_yaml() {
  local template_yaml="$1"
  python3 - "$template_yaml" <<'PY'
import sys
path = sys.argv[1]
cur = None
in_protein = False
for raw in open(path):
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
        cid = s.split(":", 1)[1].strip().strip("'\"")
        if cid and cid != "A":
            print(cid)
            break
PY
}

is_openfold_msa_path_compatible() {
  local msa_path="${1:-}"
  [[ -n "${msa_path}" ]] || return 1
  case "${msa_path##*.}" in
    a3m|sto|npz) return 0 ;;
    *) return 1 ;;
  esac
}

pick_target_msa_for_predictor() {
  local msa_path="$1"
  local predictor="$2"
  sanitize_a3m_for_post() {
    local in_path="$1"
    [[ -f "${in_path}" ]] || { echo "${in_path}"; return 0; }
    local out_path
    out_path="$(python3 - "${in_path}" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
b = p.read_bytes()
if b"\x00" in b:
    out = p.with_name(p.stem + ".sanitized.a3m")
    out.write_bytes(b.replace(b"\x00", b""))
    print(str(out))
else:
    print(str(p))
PY
)"
    echo "${out_path}"
  }

  if [[ -z "${msa_path}" ]]; then
    echo ""
    return 0
  fi
  case "${predictor}" in
    openfold-3-mlx)
      # OpenFold only accepts specific MSA formats (a3m/sto/npz). If we have a
      # dedicated cached path for OpenFold, always prefer it.
      if [[ -n "${OPENFOLD_TARGET_MSA_PATH:-}" ]]; then
        echo "${OPENFOLD_TARGET_MSA_PATH}"
        return 0
      fi
      if is_openfold_msa_path_compatible "${msa_path}"; then
        echo "${msa_path}"
      else
        # Returning empty here avoids passing incompatible files (e.g. Boltz CSV)
        # into OpenFold query JSON as main_msa_file_paths.
        echo ""
      fi
      return 0
      ;;
    boltz|intellifold)
      if [[ "${msa_path##*.}" == "npz" ]]; then
        local root_dir cand
        root_dir="$(cd "$(dirname "${msa_path}")/.." 2>/dev/null && pwd || true)"
        cand=""
        if [[ -n "${root_dir}" && -f "${root_dir}/raw/main/uniref90_hits.a3m" ]]; then
          cand="${root_dir}/raw/main/uniref90_hits.a3m"
        elif [[ -n "${root_dir}" && -f "${root_dir}/raw/main/uniref.a3m" ]]; then
          cand="${root_dir}/raw/main/uniref.a3m"
        elif [[ -n "${root_dir}" ]]; then
          cand="$(find "${root_dir}/raw/main" -maxdepth 1 -type f -name '*.a3m' 2>/dev/null | sort | head -n 1 || true)"
        fi
        if [[ -n "${cand}" ]]; then
          sanitize_a3m_for_post "${cand}"
          return 0
        fi
        if [[ "${predictor}" == "boltz" && -n "${OPENFOLD_TARGET_MSA_PATH:-}" ]]; then
          local of_root of_cand
          of_root="$(cd "$(dirname "${OPENFOLD_TARGET_MSA_PATH}")/.." 2>/dev/null && pwd || true)"
          of_cand=""
          if [[ -n "${of_root}" && -f "${of_root}/raw/main/uniref90_hits.a3m" ]]; then
            of_cand="${of_root}/raw/main/uniref90_hits.a3m"
          elif [[ -n "${of_root}" && -f "${of_root}/raw/main/uniref.a3m" ]]; then
            of_cand="${of_root}/raw/main/uniref.a3m"
          fi
          if [[ -n "${of_cand}" ]]; then
            sanitize_a3m_for_post "${of_cand}"
            return 0
          fi
        fi
        if [[ "${predictor}" == "boltz" ]]; then
          # Avoid feeding unknown NPZ formats (e.g. IntelliFold internal cache)
          # to Boltz directly; let Boltz resolve MSA itself.
          echo ""
          return 0
        fi
      fi
      if [[ "${msa_path##*.}" == "a3m" ]]; then
        sanitize_a3m_for_post "${msa_path}"
        return 0
      fi
      ;;
  esac
  echo "${msa_path}"
}

write_single_seq_a3m() {
  local seq="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  {
    echo ">query"
    echo "$seq"
    # OpenFold MSA parsing can drop single-row alignments; keep one dummy hit row.
    echo ">|||DUMMY||"
    echo "$seq"
  } > "$out"
}

sanitize_protein_sequence_for_openfold() {
  local seq="$1"
  python3 - "$seq" <<'PY'
import sys
s = sys.argv[1].strip().upper()
allowed = set("ACDEFGHIKLMNPQRSTVWY")
changed = 0
out = []
for ch in s:
    if ch in allowed:
        out.append(ch)
    else:
        out.append("A")
        changed += 1
print(f"{''.join(out)}|{changed}")
PY
}

write_openfold_runner_yaml() {
  local out_yaml="$1"
  local acc
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    acc="cpu"
    cat > "${out_yaml}" <<EOF2
experiment_settings:
  mode: predict

pl_trainer_args:
  accelerator: ${acc}
  devices: 1

model_update:
  presets: ["predict", "pae_enabled"]
  custom:
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false
EOF2
  else
    acc="gpu"
    cat > "${out_yaml}" <<EOF2
experiment_settings:
  mode: predict

pl_trainer_args:
  accelerator: ${acc}
  devices: 1

model_update:
  presets: ["predict", "pae_enabled"]
  custom:
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false
          use_lma: false
          use_mlx_attention: true
          use_mlx_triangle_kernels: true
          use_mlx_activation_functions: true
EOF2
  fi
}

generate_openfold_target_msa_cache() {
  local template_yaml="$1"
  local msa_cache_dir="$2"
  local target_seq="$3"
  local target_chain_id="$4"

  local query_json out_dir log_file rc
  query_json="${msa_cache_dir}/target_query.json"
  out_dir="${msa_cache_dir}/target_msa"
  log_file="${msa_cache_dir}/target_msa.log"
  mkdir -p "${msa_cache_dir}" "${out_dir}"

  python3 - "${target_seq}" "${target_chain_id}" "${query_json}" <<'PY'
import json, sys
seq, cid, out_json = sys.argv[1:4]
payload = {
    "seeds": [42],
    "queries": {
        "target_msa_only": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [cid],
                    "sequence": seq,
                }
            ]
        }
    },
}
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
PY

  source "${OPENFOLD_VENV}/bin/activate"
  set +e
  KMP_USE_SHM=0 "${OPENFOLD_CLI}" align-msa-server \
    --query_json "${query_json}" \
    --output_dir "${out_dir}" \
    >"${log_file}" 2>&1
  rc=$?
  set -e
  deactivate || true
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 120 "${log_file}" >&2 || true
    die "OpenFold target MSA calibration failed (rc=${rc})."
  fi

  local msa_a3m msa_path
  msa_a3m="$(find "${out_dir}/raw/main" -maxdepth 1 -type f -name 'uniref90_hits.a3m' | head -n 1 || true)"
  if [[ -z "${msa_a3m}" ]]; then
    msa_a3m="$(find "${out_dir}/raw/main" -maxdepth 1 -type f -name 'uniref.a3m' | head -n 1 || true)"
  fi
  if [[ -z "${msa_a3m}" ]]; then
    msa_a3m="$(find "${out_dir}/raw/main" -maxdepth 1 -type f -name '*.a3m' | sort | head -n 1 || true)"
  fi
  if [[ -n "${msa_a3m}" ]]; then
    local canonical_a3m
    canonical_a3m="${out_dir}/raw/main/uniref90_hits.a3m"
    python3 - "${msa_a3m}" "${canonical_a3m}" <<'PY'
from pathlib import Path
import sys
src, dst = map(Path, sys.argv[1:3])
data = src.read_bytes().replace(b"\x00", b"")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(data)
PY
  fi

  # Prefer NPZ for OpenFold runtime speed; post predictors map NPZ -> cached A3M.
  msa_path="$(find "${out_dir}/main" -maxdepth 1 -type f -name 'colabfold_main.npz' | head -n 1 || true)"
  if [[ -z "${msa_path}" ]]; then
    msa_path="$(find "${out_dir}/main" -maxdepth 1 -type f -name '*.npz' | head -n 1 || true)"
  fi
  if [[ -z "${msa_path}" && -n "${msa_a3m}" ]]; then
    msa_path="${out_dir}/raw/main/uniref90_hits.a3m"
  fi
  if [[ -z "${msa_path}" ]]; then
    tail -n 120 "${log_file}" >&2 || true
    die "OpenFold target MSA cache was not produced in ${out_dir}."
  fi

  echo "${msa_path}" > "${msa_cache_dir}/target_msa_path.txt"
  echo "${msa_path}"
}

build_openfold_query_json() {
  local template_yaml="$1"
  local binder_seq="$2"
  local query_name="$3"
  local out_json="$4"
  local target_msa_path="$5"
  local binder_msa_path="$6"

  python3 - "$template_yaml" "$binder_seq" "$query_name" "$out_json" "$target_msa_path" "$binder_msa_path" <<'PY'
import json, sys
from pathlib import Path

template, binder_seq, query_name, out_json, target_msa_path, binder_msa_path = sys.argv[1:7]

def strip_quotes(v: str) -> str:
    v = v.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    return v

def usable(v: str) -> bool:
    low = v.strip().lower()
    return bool(v.strip()) and low not in {"empty", "none", "null"}

def parse_yaml_sequences(path: str):
    items = []
    cur = None
    list_key = None
    for raw in open(path):
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
            cur.setdefault(list_key, []).append(strip_quotes(s[2:].strip()))
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
        if v.startswith("[") and v.endswith("]"):
            vals = [strip_quotes(x.strip()) for x in v[1:-1].split(",") if x.strip()]
            cur[k] = vals
        else:
            cur[k] = strip_quotes(v)
    if cur:
        items.append(cur)
    return items

def chain_id_of(item):
    cid = item.get("id")
    if cid:
        return str(cid)
    cids = item.get("chain_ids")
    if isinstance(cids, list) and cids:
        return str(cids[0])
    if isinstance(cids, str) and cids:
        return cids
    return ""

entries = parse_yaml_sequences(template)
out_chains = []
need_server = False

for e in entries:
    kind = str(e.get("kind", "")).lower()
    cid = chain_id_of(e)
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
        row = {
            "molecule_type": kind,
            "chain_ids": [cid],
            "sequence": seq,
        }
        if kind in {"protein", "rna"}:
            if usable(msa):
                row["main_msa_file_paths"] = [msa]
            else:
                need_server = True
        out_chains.append(row)
        continue

    if kind == "ligand":
        row = {
            "molecule_type": "ligand",
            "chain_ids": [cid],
        }
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

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)

print("true" if need_server else "false")
PY
}

extract_redesigned_sequence_from_fastas() {
  local multimer_mode="${UNCONDITIONAL_MULTIMER_MODE:-0}"
  local multimer_size="${UNCONDITIONAL_MULTIMER_SIZE:-3}"
  python3 - "$multimer_mode" "$multimer_size" "$@" <<'PY'
import sys
from pathlib import Path
multimer_mode = int(sys.argv[1])
multimer_size = int(sys.argv[2])
paths=[Path(p) for p in sys.argv[3:]]
if not paths:
    raise SystemExit("No FASTA files passed")
seqs=[]
for p in paths:
    cur=[]
    for line in p.read_text().splitlines():
        s=line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur=[]
        else:
            cur.append(s)
    if cur:
        seqs.append("".join(cur))
if not seqs:
    raise SystemExit("No sequences found")
pick = seqs[1] if len(seqs) >= 2 else seqs[0]
if multimer_mode == 1:
    parts = [p.strip() for p in pick.split(":") if p.strip()]
    if not parts:
        raise SystemExit("No redesign sequence found in FASTA")
    if len(parts) >= multimer_size:
        use = parts[:multimer_size]
    else:
        use = parts + [parts[-1]] * (multimer_size - len(parts))
    print(":".join(use))
else:
    print(pick.split(":",1)[0] if ":" in pick else pick)
PY
}

patch_cif_unk() {
  local in_cif="$1"
  local out_cif="$2"
  local mode="$3"
  python3 - "$in_cif" "$out_cif" "$mode" <<'PY'
import sys, random, re
src, dst, mode = sys.argv[1:4]
text = open(src).read()
pat = re.compile(r'(?<!\S)UNK(?!\S)')
if mode == "ala":
    out = pat.sub("ALA", text)
elif mode == "ala_gly":
    out = pat.sub(lambda _: "GLY" if random.random() < 0.5 else "ALA", text)
elif mode == "ala_gly_ser":
    out = pat.sub(lambda _: random.choice(("ALA", "GLY", "SER")), text)
else:
    raise SystemExit(f"Unknown mode: {mode}")
open(dst,"w").write(out)
PY
}

convert_cif_to_pdb() {
  local in_cif="$1"
  local out_pdb="$2"
  "${BOLTZ_VENV}/bin/python" - "$in_cif" "$out_pdb" <<'PY'
import sys, gemmi

in_path, out_path = sys.argv[1:3]

def has_atoms(structure):
    for model in structure:
        for chain in model:
            for res in chain:
                for _ in res:
                    return True
    return False

try:
    st = gemmi.read_structure(in_path)
except Exception:
    st = None

# Fast path: standard mmCIF readable by gemmi Structure API.
if st is not None and len(st) > 0 and has_atoms(st):
    st.write_pdb(out_path)
    raise SystemExit(0)

# Fallback path for CIFs missing fields expected by gemmi.read_structure
# (e.g. no _atom_site.occupancy in some OpenFold outputs).
doc = gemmi.cif.read_file(in_path)
block = doc.sole_block()
tab = block.find_mmcif_category('_atom_site.')
if len(tab) == 0:
    raise SystemExit(f"No _atom_site category in {in_path}")

tags = [str(t) for t in tab.tags]
idx_full = {t: i for i, t in enumerate(tags)}
idx_short = {}
for t, i in idx_full.items():
    key = t.split('.', 1)[1] if '.' in t else t
    idx_short[key] = i

def get(row, name, default=''):
    if name in idx_short:
        return row[idx_short[name]]
    full = f"_atom_site.{name}"
    if full in idx_full:
        return row[idx_full[full]]
    return default

def as_float(v, default=0.0):
    try:
        if v in ('', '.', '?'):
            return default
        return float(v)
    except Exception:
        return default

def as_int(v, default=0):
    try:
        if v in ('', '.', '?'):
            return default
        return int(float(v))
    except Exception:
        return default

lines = []
serial = 1
for row in tab:
    rec = (get(row, 'group_PDB', 'ATOM') or 'ATOM').strip().upper()
    if rec not in ('ATOM', 'HETATM'):
        continue
    atom_name = (get(row, 'auth_atom_id') or get(row, 'label_atom_id') or 'X').strip()
    alt = (get(row, 'label_alt_id', ' ') or ' ').strip()
    if alt in ('.', '?', ''):
        alt = ' '
    resname = (get(row, 'auth_comp_id') or get(row, 'label_comp_id') or 'UNK').strip()[:3]
    chain = (get(row, 'auth_asym_id') or get(row, 'label_asym_id') or 'A').strip()
    chain = chain[0] if chain else 'A'
    resseq = as_int(get(row, 'auth_seq_id') or get(row, 'label_seq_id'), 1)
    icode = (get(row, 'pdbx_PDB_ins_code', ' ') or ' ').strip()
    if icode in ('.', '?', ''):
        icode = ' '
    x = as_float(get(row, 'Cartn_x'))
    y = as_float(get(row, 'Cartn_y'))
    z = as_float(get(row, 'Cartn_z'))
    occ = as_float(get(row, 'occupancy'), 1.0)
    b = as_float(get(row, 'B_iso_or_equiv'), 0.0)
    elem = (get(row, 'type_symbol') or atom_name[:1] or 'X').strip().upper()[:2]
    charge = (get(row, 'pdbx_formal_charge', '') or '').strip()
    if charge in ('.', '?'):
        charge = ''
    if charge and len(charge) == 1 and charge in '+-':
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
    raise SystemExit(f"Failed to convert CIF to PDB atoms for {in_path}")

with open(out_path, 'w') as fh:
    for ln in lines:
        fh.write(ln + "\n")
    fh.write("END\n")
PY
}

extract_metrics_from_conf_json() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import sys, json
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    print("nan,nan")
    raise SystemExit(0)

def get_any(obj, keys):
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

iptm = get_any(d, {"iptm", "i_ptm", "iptm_score", "iptm+ptm"})
plddt = get_any(d, {"complex_plddt", "avg_plddt", "plddt"})

def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

print(f"{to_float(iptm)},{to_float(plddt)}")
PY
}

find_boltz_results_root() {
  local boltz_out="$1"
  find "${boltz_out}" -maxdepth 2 -type d -name "boltz_results_*" | head -n 1 || true
}

find_boltz_pred_leaf() {
  local boltz_results_root="$1"
  find "${boltz_results_root}/predictions" -maxdepth 2 -type d -print | grep -v "/predictions$" | head -n 1 || true
}

run_predict_boltz() {
  local input_yaml="$1"
  local out_dir="$2"
  local pred_min="$3"
  local use_potentials="${4:-0}"
  local predict_log
  predict_log="${out_dir}/predict.log"

  mkdir -p "${out_dir}"
  local cmd=("${BOLTZ_CLI}" predict "${input_yaml}" --out_dir "${out_dir}" "${BOLTZ_EXTRA_FLAGS[@]}")
  if [[ "${use_potentials}" -eq 1 ]]; then
    local _has_pot=0 tok
    for tok in "${cmd[@]}"; do
      if [[ "${tok}" == "--use_potentials" ]]; then
        _has_pot=1
        break
      fi
    done
    if [[ "${_has_pot}" -eq 0 ]]; then
      cmd+=("--use_potentials")
    fi
  fi
  source "${BOLTZ_VENV}/bin/activate"
  "${cmd[@]}" >"${predict_log}" 2>&1 &
  local pid=$!
  local tick=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 1
    tick=$((tick + 1))
    if (( tick % 30 == 0 )); then
      local rss_kb
      rss_kb="$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{print $1}' || echo 0)"
      echo ">>> Boltz still running (pid=${pid}, elapsed=${tick}s, rss_kb=${rss_kb})" >&2
    fi
  done
  wait "${pid}"
  local rc=$?
  deactivate || true
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: Boltz prediction failed (rc=${rc}) for ${input_yaml}" >&2
    tail -n 80 "${predict_log}" >&2 || true
    die "Boltz prediction failed (rc=${rc}) for ${input_yaml}"
  fi

  local root leaf conf struct
  root="$(find_boltz_results_root "${out_dir}")"
  [[ -n "$root" ]] || die "Boltz output missing boltz_results_* in ${out_dir}"
  leaf="$(find_boltz_pred_leaf "${root}")"
  [[ -n "$leaf" ]] || die "Boltz predictions leaf not found in ${root}"

  conf="$(find "${leaf}" -maxdepth 1 -type f -name 'confidence_*_model_0.json' | sort | head -n 1 || true)"
  struct="$(find "${leaf}" -maxdepth 1 -type f \( -name '*.cif' -o -name '*.pdb' \) | sort | head -n 1 || true)"
  [[ -n "$struct" ]] || die "Boltz structure not found in ${leaf}"

  mkdir -p "${pred_min}"
  if [[ -n "$conf" ]]; then cp -f "$conf" "${pred_min}/confidence.json"; fi
  if [[ "${struct##*.}" == "cif" ]]; then
    cp -f "$struct" "${pred_min}/model_0.cif"
  else
    cp -f "$struct" "${pred_min}/model_0.pdb"
  fi

  local iptm plddt
  iptm="nan"; plddt="nan"
  if [[ -f "${pred_min}/confidence.json" ]]; then
    IFS=',' read -r iptm plddt <<< "$(extract_metrics_from_conf_json "${pred_min}/confidence.json")"
    echo "$iptm" > "${pred_min}/iptm.txt" || true
  fi

  echo "${struct}|${conf}|${iptm}|${plddt}"
}

run_boltz_predict_monitored() {
  local yaml="$1"
  local out_dir="$2"
  local peak_file="$3"
  local use_potentials="${4:-0}"
  local predict_log
  predict_log="${out_dir}/calibration_predict.log"

  mkdir -p "${out_dir}"
  rm -f "${peak_file}"

  local cmd=("${BOLTZ_CLI}" predict "${yaml}" --out_dir "${out_dir}" "${BOLTZ_EXTRA_FLAGS[@]}")
  if [[ "${use_potentials}" -eq 1 ]]; then
    local _has_pot=0 tok
    for tok in "${cmd[@]}"; do
      if [[ "${tok}" == "--use_potentials" ]]; then
        _has_pot=1
        break
      fi
    done
    if [[ "${_has_pot}" -eq 0 ]]; then
      cmd+=("--use_potentials")
    fi
  fi
  source "${BOLTZ_VENV}/bin/activate"
  "${cmd[@]}" >"${predict_log}" 2>&1 &
  local pid=$!

  local peak_kb=0
  local tick=0
  while kill -0 "${pid}" 2>/dev/null; do
    local rss_kb
    rss_kb="$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{print $1}' || echo 0)"
    if [[ -n "${rss_kb}" && "${rss_kb}" =~ ^[0-9]+$ ]]; then
      if (( rss_kb > peak_kb )); then
        peak_kb="${rss_kb}"
      fi
    fi
    tick=$((tick + 1))
    if (( tick % 30 == 0 )); then
      echo ">>> Calibration Boltz still running (pid=${pid}, elapsed=${tick}s, rss_kb=${rss_kb})" >&2
    fi
    sleep 0.5
  done

  wait "${pid}"
  local rc=$?
  deactivate || true

  local peak_mb
  peak_mb="$(python3 - "$peak_kb" <<'PY'
import sys
kb=int(sys.argv[1])
print(max(1, int(kb/1024)))
PY
)"
  echo "${peak_mb}" > "${peak_file}"
  return "${rc}"
}

run_intellifold_predict_monitored() {
  local yaml="$1"
  local out_dir="$2"
  local peak_file="$3"
  local predict_log
  predict_log="${out_dir}/calibration_predict.log"

  mkdir -p "${out_dir}"
  rm -f "${peak_file}"

  source "${INTELLIFOLD_VENV}/bin/activate"
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    ACCELERATE_USE_CPU=true KMP_USE_SHM=0 "${INTELLIFOLD_CLI}" predict "${yaml}" --out_dir "${out_dir}" "${INTELLIFOLD_EXTRA_FLAGS[@]}" >"${predict_log}" 2>&1 &
  else
    KMP_USE_SHM=0 "${INTELLIFOLD_CLI}" predict "${yaml}" --out_dir "${out_dir}" "${INTELLIFOLD_EXTRA_FLAGS[@]}" >"${predict_log}" 2>&1 &
  fi
  local pid=$!

  local peak_kb=0
  local tick=0
  while kill -0 "${pid}" 2>/dev/null; do
    local rss_kb
    rss_kb="$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{print $1}' || echo 0)"
    if [[ -n "${rss_kb}" && "${rss_kb}" =~ ^[0-9]+$ ]]; then
      if (( rss_kb > peak_kb )); then
        peak_kb="${rss_kb}"
      fi
    fi
    tick=$((tick + 1))
    if (( tick % 30 == 0 )); then
      echo ">>> Calibration IntelliFold still running (pid=${pid}, elapsed=${tick}s, rss_kb=${rss_kb})" >&2
    fi
    sleep 0.5
  done

  wait "${pid}"
  local rc=$?
  deactivate || true

  local peak_mb
  peak_mb="$(python3 - "$peak_kb" <<'PY'
import sys
kb=int(sys.argv[1])
print(max(1, int(kb/1024)))
PY
)"
  echo "${peak_mb}" > "${peak_file}"
  return "${rc}"
}

run_predict_intellifold() {
  local input_yaml="$1"
  local out_dir="$2"
  local pred_min="$3"
  local predict_log rc
  predict_log="${out_dir}/predict.log"

  local input_stem
  input_stem="$(basename "${input_yaml%.*}")"

  mkdir -p "${out_dir}"
  source "${INTELLIFOLD_VENV}/bin/activate"
  set +e
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    ACCELERATE_USE_CPU=true KMP_USE_SHM=0 "${INTELLIFOLD_CLI}" predict "${input_yaml}" --out_dir "${out_dir}" "${INTELLIFOLD_EXTRA_FLAGS[@]}" >"${predict_log}" 2>&1
  else
    KMP_USE_SHM=0 "${INTELLIFOLD_CLI}" predict "${input_yaml}" --out_dir "${out_dir}" "${INTELLIFOLD_EXTRA_FLAGS[@]}" >"${predict_log}" 2>&1
  fi
  rc=$?
  set -e
  deactivate || true
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: IntelliFold prediction failed (rc=${rc}) for ${input_yaml}" >&2
    tail -n 80 "${predict_log}" >&2 || true
    die "IntelliFold prediction failed (rc=${rc}) for ${input_yaml}"
  fi

  local leaf conf struct
  leaf="${out_dir}/${input_stem}/predictions/${input_stem}"
  [[ -d "$leaf" ]] || die "IntelliFold predictions leaf not found: ${leaf}"

  conf="$(find "${leaf}" -maxdepth 1 -type f -name '*_summary_confidences.json' | sort | head -n 1 || true)"
  struct="$(find "${leaf}" -maxdepth 1 -type f -name '*.cif' | sort | head -n 1 || true)"
  [[ -n "$struct" ]] || die "IntelliFold structure not found in ${leaf}"

  mkdir -p "${pred_min}"
  if [[ -n "$conf" ]]; then cp -f "$conf" "${pred_min}/confidence.json"; fi
  cp -f "$struct" "${pred_min}/model_0.cif"

  local iptm plddt
  iptm="nan"; plddt="nan"
  if [[ -f "${pred_min}/confidence.json" ]]; then
    IFS=',' read -r iptm plddt <<< "$(extract_metrics_from_conf_json "${pred_min}/confidence.json")"
    echo "$iptm" > "${pred_min}/iptm.txt" || true
  fi

  echo "${struct}|${conf}|${iptm}|${plddt}"
}

run_predict_openfold() {
  local input_yaml="$1"
  local binder_seq="$2"
  local query_name="$3"
  local out_dir="$4"
  local pred_min="$5"
  local target_msa_path="$6"

  local query_json runner_yaml binder_msa_path use_server
  local predict_log rc
  local binder_seq_clean changed_count
  query_json="${out_dir}/${query_name}_query.json"
  runner_yaml="${out_dir}/${query_name}_runner.yml"
  # Use an OF3-style filename so default MSA settings ingest it without custom runner config.
  binder_msa_path="${out_dir}/binder_msa/uniref90_hits.a3m"
  predict_log="${out_dir}/predict.log"
  mkdir -p "${out_dir}"

  IFS='|' read -r binder_seq_clean changed_count <<< "$(sanitize_protein_sequence_for_openfold "${binder_seq}")"
  if [[ "${changed_count}" =~ ^[0-9]+$ ]] && (( changed_count > 0 )); then
    echo "WARN: OpenFold input sequence had ${changed_count} non-standard residues; replaced with 'A' for prediction." >&2
  fi

  write_single_seq_a3m "${binder_seq_clean}" "${binder_msa_path}"
  use_server="$(build_openfold_query_json "${input_yaml}" "${binder_seq_clean}" "${query_name}" "${query_json}" "${target_msa_path}" "${binder_msa_path}")"
  write_openfold_runner_yaml "${runner_yaml}"

  source "${OPENFOLD_VENV}/bin/activate"
  set +e
  KMP_USE_SHM=0 "${OPENFOLD_CLI}" predict \
    --query_json "${query_json}" \
    --output_dir "${out_dir}" \
    --runner_yaml "${runner_yaml}" \
    --use_msa_server "${use_server}" \
    "${OPENFOLD_EXTRA_FLAGS[@]}" \
    >"${predict_log}" 2>&1
  rc=$?
  set -e
  deactivate || true
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: OpenFold prediction failed (rc=${rc}) for ${input_yaml}" >&2
    tail -n 80 "${predict_log}" >&2 || true
    die "OpenFold prediction failed (rc=${rc}) for ${input_yaml}"
  fi

  local leaf conf struct
  leaf="${out_dir}/${query_name}/seed_42"
  [[ -d "$leaf" ]] || leaf="$(find "${out_dir}" -maxdepth 4 -type d -name 'seed_*' | sort | head -n 1 || true)"
  if [[ -z "$leaf" || ! -d "$leaf" ]]; then
    echo "ERROR: OpenFold seed output not found in ${out_dir}" >&2
    tail -n 120 "${predict_log}" >&2 || true
    die "OpenFold seed output not found in ${out_dir}"
  fi

  conf="$(find "${leaf}" -maxdepth 1 -type f -name '*_confidences_aggregated.json' | sort | head -n 1 || true)"
  struct="$(find "${leaf}" -maxdepth 1 -type f \( -name '*_model.cif' -o -name '*_model.pdb' \) | sort | head -n 1 || true)"
  [[ -n "$struct" ]] || die "OpenFold structure not found in ${leaf}"

  mkdir -p "${pred_min}"
  if [[ -n "$conf" ]]; then cp -f "$conf" "${pred_min}/confidence.json"; fi
  if [[ "${struct##*.}" == "cif" ]]; then
    cp -f "$struct" "${pred_min}/model_0.cif"
  else
    cp -f "$struct" "${pred_min}/model_0.pdb"
  fi

  local iptm plddt
  iptm="nan"; plddt="nan"
  if [[ -f "${pred_min}/confidence.json" ]]; then
    IFS=',' read -r iptm plddt <<< "$(extract_metrics_from_conf_json "${pred_min}/confidence.json")"
    echo "$iptm" > "${pred_min}/iptm.txt" || true
  fi

  echo "${struct}|${conf}|${iptm}|${plddt}"
}

run_predictor_calibration_once() {
  local predictor="$1"
  local target_msa_path="$2"
  local cal_dir cal_yaml cal_seq qname result
  cal_dir="${EXPT_ROOT}/_calibration/cycle_00"
  cal_yaml="${cal_dir}/boltz_input.yaml"
  qname="calibration_input"
  mkdir -p "${cal_dir}"

  # Calibrate with the longest binder in the configured range.
  cal_seq="$(generate_random_binder_seq "${BINDER_MAX_LEN}" "${BINDER_MAX_LEN}" "${BINDER_PERCENT_X}" "${HELIX_KILL}" "${NEGATIVE_HELIX_CONSTANT}" "${LOOP_KILL}" "${predictor}")"
  make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${cal_yaml}" "${cal_seq}" "${target_msa_path}" "${UNCONDITIONAL_MULTIMER_MODE}" "${UNCONDITIONAL_MULTIMER_SIZE}"
  result="$(run_predictor_once "${predictor}" "${cal_yaml}" "${cal_seq}" "${qname}" "${cal_dir}" "${target_msa_path}" "design")"
  [[ -n "${result}" ]] || die "Calibration prediction produced no output."
}

run_predictor_once() {
  local predictor="$1"
  local cycle_yaml="$2"
  local binder_seq="$3"
  local query_name="$4"
  local cycle_dir="$5"
  local target_msa_path="$6"
  local phase="${7:-design}"

  local pred_min
  pred_min="${cycle_dir}/pred_min"

  case "${predictor}" in
    boltz)
      local _use_pot=0
      if [[ "${phase}" == "design" && "${BOLTZ_USE_POTENTIALS_DEFAULT}" -eq 1 ]]; then
        _use_pot=1
      fi
      run_predict_boltz "${cycle_yaml}" "${cycle_dir}/boltz" "${pred_min}" "${_use_pot}"
      ;;
    intellifold)
      run_predict_intellifold "${cycle_yaml}" "${cycle_dir}/intellifold" "${pred_min}"
      ;;
    openfold-3-mlx)
      run_predict_openfold "${cycle_yaml}" "${binder_seq}" "${query_name}" "${cycle_dir}/openfold3" "${pred_min}" "${target_msa_path}"
      ;;
    *)
      die "Unsupported predictor: ${predictor}"
      ;;
  esac
}

run_ligandmpnn_redesign() {
  local cycle_dir="$1"
  local struct_path="$2"
  local cycle_idx="$3"
  cycle_dir="$(cd "${cycle_dir}" && pwd)"

  local input_struct
  input_struct="${struct_path}"

  if [[ "${struct_path##*.}" == "cif" && "${cycle_idx}" -eq 0 ]]; then
    local patched
    patched="${cycle_dir}/pred_min/model_0_UNKPATCH.cif"
    patch_cif_unk "${struct_path}" "${patched}" "${UNK_PATCH_MODE}"
    input_struct="${patched}"
  fi

  local lmpnn_input
  if [[ "${input_struct##*.}" == "cif" ]]; then
    lmpnn_input="${cycle_dir}/model_for_ligandmpnn.pdb"
    convert_cif_to_pdb "${input_struct}" "${lmpnn_input}"
  else
    lmpnn_input="${input_struct}"
    cp -f "${input_struct}" "${cycle_dir}/model_for_ligandmpnn.pdb" || true
  fi

  local temp
  local bias_aa
  local omit_aa
  if [[ "${cycle_idx}" -eq 0 ]]; then
    temp="${LIGAND_TEMP_CYCLE01}"
    bias_aa="${LIGAND_BIAS_AA_CYCLE01}"
  else
    temp="${LIGAND_TEMP_DEFAULT}"
    bias_aa="${LIGAND_BIAS_AA_DEFAULT}"
  fi
  bias_aa="$(compute_loopkill_mpnn_bias "${bias_aa}" "${LOOP_KILL}")"
  omit_aa="C"
  if python3 - "${LOOP_KILL}" <<'PY'
import sys
v=float(sys.argv[1])
raise SystemExit(0 if v >= 1.0 else 1)
PY
  then
    omit_aa="CP"
  fi

  local ligand_out
  ligand_out="${cycle_dir}/ligandmpnn"
  mkdir -p "${ligand_out}"

  source "${LIGAND_VENV}/bin/activate"
  pushd "${LIGANDMPNN_REPO}" >/dev/null
  # LigandMPNN can fail on macOS CPU when OpenMP shared memory is unavailable.
  if [[ -n "${bias_aa}" ]]; then
    KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${LIGANDMPNN_RUN} \
      --pdb_path "${lmpnn_input}" \
      --out_folder "${ligand_out}" \
      --temperature "${temp}" \
      --omit_AA "${omit_aa}" \
      --bias_AA "${bias_aa}" \
      "${LIGANDMPNN_MODEL_FLAGS[@]}" \
      "${LIGANDMPNN_EXTRA_FLAGS[@]}" \
      > "${ligand_out}/ligandmpnn.log" 2>&1
  else
    KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${LIGANDMPNN_RUN} \
      --pdb_path "${lmpnn_input}" \
      --out_folder "${ligand_out}" \
      --temperature "${temp}" \
      --omit_AA "${omit_aa}" \
      "${LIGANDMPNN_MODEL_FLAGS[@]}" \
      "${LIGANDMPNN_EXTRA_FLAGS[@]}" \
      > "${ligand_out}/ligandmpnn.log" 2>&1
  fi
  popd >/dev/null
  deactivate

  local seqs_dir fasta
  seqs_dir="${ligand_out}/seqs"
  [[ -d "$seqs_dir" ]] || die "LigandMPNN seqs dir not found: ${seqs_dir}"
  fasta="$(ls -1 "${seqs_dir}"/*.fa 2>/dev/null | head -n 1 || true)"
  [[ -n "$fasta" ]] || die "No LigandMPNN FASTA found in ${seqs_dir}"

  mkdir -p "${cycle_dir}/ligandmpnn_min"
  cp -f "$fasta" "${cycle_dir}/ligandmpnn_min/seqs.fa"

  extract_redesigned_sequence_from_fastas "${seqs_dir}"/*.fa
}

export_cif() {
  local run_tag="$1"
  local cycle_idx="$2"
  local pred_min="$3"
  local iptm="$4"

  local cycle_tag
  cycle_tag="$(printf "cycle_%02d" "$cycle_idx")"

  local cif_path="${pred_min}/model_0.cif"
  [[ -f "$cif_path" ]] || return 0

  local base_name="${run_tag}_${cycle_tag}_model_0.cif"
  cp -f "$cif_path" "${CIFS_ALL_DIR}/${base_name}"

  if [[ "$cycle_idx" -eq 0 ]]; then
    return 0
  fi

  if is_float "$iptm" && float_ge "$iptm" "$IPTM_THRESHOLD"; then
    cp -f "$cif_path" "${CIFS_PASS_DIR}/${base_name}"
  fi
}

wait_for_slot() {
  local limit="$1"
  while true; do
    local running
    running="$(jobs -pr | wc -l | awk '{print $1}')"
    if (( running < limit )); then
      break
    fi
    sleep 0.2
  done
}

run_one_design() {
  local run_index="$1"
  local target_msa_path="$2"

  local run_tag run_root state_seq metrics_csv rows_csv timing_cycle_csv timing_run_csv
  run_tag="$(printf "run_%03d" "$run_index")"
  run_root="${EXPT_ROOT}/${run_tag}"
  mkdir -p "${run_root}"

  state_seq="${run_root}/state_current_seq.txt"
  metrics_csv="${run_root}/metrics_per_cycle.csv"
  rows_csv="${run_root}/rows.csv"
  timing_cycle_csv="${run_root}/timing_cycles.csv"
  timing_run_csv="${run_root}/timing_run.csv"

  echo "cycle,iptm,complex_plddt,confidence_json,structure_path,binder_sequence" > "${metrics_csv}"
  echo "run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json" > "${rows_csv}"
  echo "cycle,start_ts,end_ts,duration_sec" > "${timing_cycle_csv}"

  local run_start run_end run_dur
  run_start="$(now_epoch)"
  echo ">>> Starting ${run_tag}"

  local current_seq
  current_seq="$(generate_random_binder_seq "${BINDER_MIN_LEN}" "${BINDER_MAX_LEN}" "${BINDER_PERCENT_X}" "${HELIX_KILL}" "${NEGATIVE_HELIX_CONSTANT}" "${LOOP_KILL}" "${PREDICTOR}")"
  if [[ "${UNCONDITIONAL_MULTIMER_MODE}" -eq 1 ]]; then
    current_seq="$(python3 - "${current_seq}" "${UNCONDITIONAL_MULTIMER_SIZE}" <<'PY'
import sys
seq = sys.argv[1]
n = int(sys.argv[2])
print(":".join([seq] * n))
PY
)"
  fi
  echo "$current_seq" > "$state_seq"

  local cycle
  for cycle in $(seq 0 "${N_CYCLES}"); do
    local cycle_tag cycle_dir cycle_yaml qname
    cycle_tag="$(printf "cycle_%02d" "$cycle")"
    cycle_dir="${run_root}/${cycle_tag}"
    cycle_yaml="${cycle_dir}/boltz_input.yaml"
    qname="${run_tag}_${cycle_tag}"
    mkdir -p "${cycle_dir}"

    local target_msa_for_pred
    target_msa_for_pred="$(pick_target_msa_for_predictor "${target_msa_path}" "${PREDICTOR}")"
    make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${cycle_yaml}" "${current_seq}" "${target_msa_for_pred}" "${UNCONDITIONAL_MULTIMER_MODE}" "${UNCONDITIONAL_MULTIMER_SIZE}"

    local cstart cend cdur result struct conf iptm plddt
    cstart="$(now_epoch)"
    echo ">>> ${run_tag} ${cycle_tag}: ${PREDICTOR} predict..."
    result="$(run_predictor_once "${PREDICTOR}" "${cycle_yaml}" "${current_seq}" "${qname}" "${cycle_dir}" "${target_msa_for_pred}" "design")"
    cend="$(now_epoch)"
    cdur="$(calc_duration "$cstart" "$cend")"
    echo "$(printf '%02d' "$cycle"),${cstart},${cend},${cdur}" >> "${timing_cycle_csv}"

    IFS='|' read -r struct conf iptm plddt <<< "$result"
    [[ -n "${iptm:-}" ]] || iptm="nan"
    [[ -n "${plddt:-}" ]] || plddt="nan"
    local conf_val struct_val
    conf_val="${conf:-}"
    struct_val="${struct:-}"

    echo "$(printf '%02d' "$cycle"),${iptm},${plddt},${conf_val},${struct_val},${current_seq}" >> "${metrics_csv}"
    echo "${run_index},${cycle},${iptm},${plddt},${current_seq},${struct_val},${conf_val}" >> "${rows_csv}"
    echo ">>> ${run_tag} ${cycle_tag}: done in ${cdur}s (iPTM=${iptm})"

    export_cif "$run_tag" "$cycle" "${cycle_dir}/pred_min" "$iptm"

    if (( cycle < N_CYCLES )); then
      echo ">>> ${run_tag} ${cycle_tag}: LigandMPNN redesign..."
      local redesign_struct
      if [[ -f "${cycle_dir}/pred_min/model_0.cif" ]]; then
        redesign_struct="${cycle_dir}/pred_min/model_0.cif"
      elif [[ -f "${cycle_dir}/pred_min/model_0.pdb" ]]; then
        redesign_struct="${cycle_dir}/pred_min/model_0.pdb"
      else
        die "Missing model_0.cif/model_0.pdb in ${cycle_dir}/pred_min"
      fi
      current_seq="$(run_ligandmpnn_redesign "${cycle_dir}" "${redesign_struct}" "$cycle")"
      echo "$current_seq" > "$state_seq"
    fi
  done

  run_end="$(now_epoch)"
  run_dur="$(calc_duration "$run_start" "$run_end")"
  echo "run,start_ts,end_ts,duration_sec" > "${timing_run_csv}"
  echo "${run_tag},${run_start},${run_end},${run_dur}" >> "${timing_run_csv}"
  echo ">>> Finished ${run_tag} in ${run_dur}s"
}

run_post_task() {
  local predictor="$1"
  local run_index="$2"
  local cycle_index="$3"
  local binder_seq="$4"
  local target_msa_path="$5"

  local pred_safe run_tag cycle_tag post_root post_cycle_root input_yaml qname
  pred_safe="$(safe_predictor_name "$predictor")"
  run_tag="$(printf "run_%03d" "$run_index")"
  cycle_tag="$(printf "cycle_%02d" "$cycle_index")"

  post_root="${EXPT_ROOT}/${run_tag}/post_${pred_safe}"
  post_cycle_root="${post_root}/${cycle_tag}"
  mkdir -p "${post_cycle_root}"

  input_yaml="${post_cycle_root}/post_input.yaml"
  qname="${run_tag}_post_$(printf '%02d' "$cycle_index")"
  local target_msa_for_post
  target_msa_for_post="$(pick_target_msa_for_predictor "${target_msa_path}" "${predictor}")"
  make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${input_yaml}" "${binder_seq}" "${target_msa_for_post}" "${UNCONDITIONAL_MULTIMER_MODE}" "${UNCONDITIONAL_MULTIMER_SIZE}"

  local t0 t1 dt result struct conf iptm plddt
  t0="$(now_epoch)"
  result="$(run_predictor_once "${predictor}" "${input_yaml}" "${binder_seq}" "${qname}" "${post_cycle_root}" "${target_msa_for_post}" "post")"
  t1="$(now_epoch)"
  dt="$(calc_duration "$t0" "$t1")"

  IFS='|' read -r struct conf iptm plddt <<< "$result"
  [[ -n "${iptm:-}" ]] || iptm="nan"
  [[ -n "${plddt:-}" ]] || plddt="nan"

  {
    echo "run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json"
    echo "${run_index},${cycle_index},${iptm},${plddt},${binder_seq},${struct},${conf}"
  } > "${post_cycle_root}/post_metrics_row.csv"

  {
    echo "run,cycle,start_ts,end_ts,duration_sec"
    echo "${run_index},${cycle_index},${t0},${t1},${dt}"
  } > "${post_cycle_root}/post_timing_row.csv"
}

aggregate_post_predictor() {
  local predictor="$1"
  local pred_safe
  pred_safe="$(safe_predictor_name "$predictor")"

  local summary_csv summary_timing_csv
  summary_csv="${EXPT_ROOT}/summary_post_${pred_safe}.csv"
  summary_timing_csv="${EXPT_ROOT}/summary_post_${pred_safe}_timing.csv"

  echo "run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json" > "${summary_csv}"
  echo "run,cycle,start_ts,end_ts,duration_sec" > "${summary_timing_csv}"

  local run_index run_tag post_root
  for run_index in $(seq 1 "${N_RUNS}"); do
    run_tag="$(printf "run_%03d" "$run_index")"
    post_root="${EXPT_ROOT}/${run_tag}/post_${pred_safe}"
    [[ -d "$post_root" ]] || continue

    local run_metrics run_timing
    run_metrics="${post_root}/post_metrics.csv"
    run_timing="${post_root}/post_timing.csv"
    echo "run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json" > "${run_metrics}"
    echo "run,cycle,start_ts,end_ts,duration_sec" > "${run_timing}"

    local row
    while IFS= read -r row; do
      tail -n +2 "$row" >> "$run_metrics"
      tail -n +2 "$row" >> "$summary_csv"
    done < <(find "$post_root" -type f -name 'post_metrics_row.csv' | sort)

    while IFS= read -r row; do
      tail -n +2 "$row" >> "$run_timing"
      tail -n +2 "$row" >> "$summary_timing_csv"
    done < <(find "$post_root" -type f -name 'post_timing_row.csv' | sort)
  done
}

build_comparison_tables() {
  local cmp_scores cmp_timing
  cmp_scores="${EXPT_ROOT}/comparison_scores_long.csv"
  cmp_timing="${EXPT_ROOT}/comparison_timing_long.csv"

  echo "stage,predictor,run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json" > "${cmp_scores}"
  echo "stage,predictor,run,phase,cycle,start_ts,end_ts,duration_sec" > "${cmp_timing}"

  local run_index run_tag
  for run_index in $(seq 1 "${N_RUNS}"); do
    run_tag="$(printf "run_%03d" "$run_index")"
    if [[ -f "${EXPT_ROOT}/${run_tag}/metrics_per_cycle.csv" ]]; then
      tail -n +2 "${EXPT_ROOT}/${run_tag}/metrics_per_cycle.csv" | awk -F',' -v p="$PREDICTOR" -v r="$run_index" 'BEGIN{OFS=","} {print "design",p,r,$1,$2,$3,$6,$5,$4}' >> "${cmp_scores}"
    fi
    if [[ -f "${EXPT_ROOT}/${run_tag}/timing_cycles.csv" ]]; then
      tail -n +2 "${EXPT_ROOT}/${run_tag}/timing_cycles.csv" | awk -F',' -v p="$PREDICTOR" -v r="$run_index" 'BEGIN{OFS=","} {print "design",p,r,"cycle",$1,$2,$3,$4}' >> "${cmp_timing}"
    fi
    if [[ -f "${EXPT_ROOT}/${run_tag}/timing_run.csv" ]]; then
      tail -n +2 "${EXPT_ROOT}/${run_tag}/timing_run.csv" | awk -F',' -v p="$PREDICTOR" -v r="$run_index" 'BEGIN{OFS=","} {print "design",p,r,"run","all",$2,$3,$4}' >> "${cmp_timing}"
    fi
  done

  if [[ "$(post_predictors_count)" -gt 0 ]]; then
    local pp
    for pp in "${POST_PREDICTORS[@]}"; do
      local psafe sfile tfile
      psafe="$(safe_predictor_name "$pp")"
      sfile="${EXPT_ROOT}/summary_post_${psafe}.csv"
      tfile="${EXPT_ROOT}/summary_post_${psafe}_timing.csv"

      if [[ -f "$sfile" ]]; then
        tail -n +2 "$sfile" | awk -F',' -v p="$pp" 'BEGIN{OFS=","} {print "post",p,$1,sprintf("%02d",$2),$3,$4,$5,$6,$7}' >> "${cmp_scores}"
      fi
      if [[ -f "$tfile" ]]; then
        tail -n +2 "$tfile" | awk -F',' -v p="$pp" 'BEGIN{OFS=","} {print "post",p,$1,"post_cycle",sprintf("%02d",$2),$3,$4,$5}' >> "${cmp_timing}"
      fi
    done
  fi
}

EXPT_ROOT="${BASE_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${EXPT_ROOT}"

PASS_TAG="$(python3 - "$IPTM_THRESHOLD" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}".replace(".","p"))
PY
)"

CIFS_ALL_DIR="${EXPT_ROOT}/cifs_all"
CIFS_PASS_DIR="${EXPT_ROOT}/cifs_iptm_ge_${PASS_TAG}"
mkdir -p "${CIFS_ALL_DIR}" "${CIFS_PASS_DIR}"

TARGET_MSA_PATH="$(extract_target_msa_from_yaml "${TEMPLATE_YAML}" || true)"
TARGET_SEQ="$(extract_target_sequence_from_yaml "${TEMPLATE_YAML}" || true)"
TARGET_CHAIN_ID="$(extract_target_chain_id_from_yaml "${TEMPLATE_YAML}" || true)"
OPENFOLD_TARGET_MSA_PATH=""
PEAK_RSS_MB=0
CAL_PRED_DONE=0
MSA_CACHE_DIR="${EXPT_ROOT}/msa_cache"
mkdir -p "${MSA_CACHE_DIR}"

# Default Boltz design behavior: use potentials when designing a binder
# against a protein partner or ligand partner.
if [[ "${BOLTZ_USE_POTENTIALS_MODE}" == "on" ]]; then
  BOLTZ_USE_POTENTIALS_DEFAULT=1
elif [[ "${BOLTZ_USE_POTENTIALS_MODE}" == "off" ]]; then
  BOLTZ_USE_POTENTIALS_DEFAULT=0
elif [[ "${PREDICTOR}" == "boltz" ]]; then
  if [[ "$(template_has_boltz_partner "${TEMPLATE_YAML}")" == "1" ]]; then
    BOLTZ_USE_POTENTIALS_DEFAULT=1
  fi
fi

# Target-MSA calibration/cache for all predictors.
if [[ -n "${TARGET_MSA_PATH}" ]]; then
  echo "==> Using target MSA from template: ${TARGET_MSA_PATH}"
else
  case "${PREDICTOR}" in
    openfold-3-mlx)
      if [[ -n "${TARGET_SEQ}" ]]; then
        [[ -n "${TARGET_CHAIN_ID}" ]] || TARGET_CHAIN_ID="B"
        echo "==> Calibration: generating OpenFold target MSA cache for chain ${TARGET_CHAIN_ID}..."
        TARGET_MSA_PATH="$(generate_openfold_target_msa_cache "${TEMPLATE_YAML}" "${MSA_CACHE_DIR}" "${TARGET_SEQ}" "${TARGET_CHAIN_ID}")"
        echo "==> Cached target MSA: ${TARGET_MSA_PATH}"
      else
        # Unconditional/monomer templates have no target chain; skip target-MSA calibration.
        echo "==> No target chain sequence found; skipping OpenFold target-MSA calibration."
      fi
      ;;
    boltz)
      echo "==> Calibration: generating target MSA cache with Boltz..."
      CAL_DIR="${EXPT_ROOT}/_calibration/cycle_00"
      mkdir -p "${CAL_DIR}"
      CAL_YAML="${CAL_DIR}/boltz_input.yaml"
      CAL_SEQ="$(generate_random_binder_seq "${BINDER_MAX_LEN}" "${BINDER_MAX_LEN}" "${BINDER_PERCENT_X}" "${HELIX_KILL}" "${NEGATIVE_HELIX_CONSTANT}" "${LOOP_KILL}" "boltz")"
      make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${CAL_YAML}" "${CAL_SEQ}" "" "${UNCONDITIONAL_MULTIMER_MODE}" "${UNCONDITIONAL_MULTIMER_SIZE}"

      PEAK_MB_FILE="${EXPT_ROOT}/calibration_peak_rss_mb.txt"
      if ! run_boltz_predict_monitored "${CAL_YAML}" "${CAL_DIR}/boltz" "${PEAK_MB_FILE}"; then
        die "Calibration Boltz run failed."
      fi
      PEAK_RSS_MB="$(cat "${PEAK_MB_FILE}" 2>/dev/null || echo 0)"

      TARGET_MSA_DISC="$(find "${CAL_DIR}/boltz" -type f -name '*.csv' -path '*/msa/*' | head -n 1 || true)"
      if [[ -n "${TARGET_MSA_DISC}" ]]; then
        cp -f "${TARGET_MSA_DISC}" "${MSA_CACHE_DIR}/target_msa.csv"
        TARGET_MSA_PATH="${MSA_CACHE_DIR}/target_msa.csv"
        echo "==> Cached target MSA: ${TARGET_MSA_PATH}"
      else
        echo "==> Calibration did not find target MSA CSV; proceeding without cached target MSA."
      fi
      if [[ "${PREDICTOR}" == "boltz" ]]; then
        CAL_PRED_DONE=1
      fi
      ;;
    intellifold)
      echo "==> Calibration: generating target MSA cache with IntelliFold..."
      CAL_DIR="${EXPT_ROOT}/_calibration/cycle_00"
      mkdir -p "${CAL_DIR}"
      CAL_YAML="${CAL_DIR}/boltz_input.yaml"
      CAL_SEQ="$(generate_random_binder_seq "${BINDER_MAX_LEN}" "${BINDER_MAX_LEN}" "${BINDER_PERCENT_X}" "${HELIX_KILL}" "${NEGATIVE_HELIX_CONSTANT}" "${LOOP_KILL}" "intellifold")"
      make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${CAL_YAML}" "${CAL_SEQ}" "" "${UNCONDITIONAL_MULTIMER_MODE}" "${UNCONDITIONAL_MULTIMER_SIZE}"

      PEAK_MB_FILE="${EXPT_ROOT}/calibration_peak_rss_mb.txt"
      if ! run_intellifold_predict_monitored "${CAL_YAML}" "${CAL_DIR}/intellifold" "${PEAK_MB_FILE}"; then
        die "Calibration IntelliFold run failed."
      fi
      PEAK_RSS_MB="$(cat "${PEAK_MB_FILE}" 2>/dev/null || echo 0)"

      # IntelliFold requires a3m/csv for explicit MSA input; do not cache NPZ
      # as primary TARGET_MSA_PATH for IntelliFold runs.
      TARGET_MSA_DISC="$(find "${CAL_DIR}/intellifold" -type f -path '*/msa/*.csv' | sort | head -n 1 || true)"
      if [[ -n "${TARGET_MSA_DISC}" ]]; then
        cp -f "${TARGET_MSA_DISC}" "${MSA_CACHE_DIR}/target_msa.csv"
        TARGET_MSA_PATH="${MSA_CACHE_DIR}/target_msa.csv"
        echo "==> Cached target MSA: ${TARGET_MSA_PATH}"
      else
        TARGET_MSA_DISC="$(find "${CAL_DIR}/intellifold" -type f -path '*/msa/*/uniref.a3m' | sort | head -n 1 || true)"
        if [[ -n "${TARGET_MSA_DISC}" ]]; then
          cp -f "${TARGET_MSA_DISC}" "${MSA_CACHE_DIR}/target_uniref.a3m"
          TARGET_MSA_PATH="${MSA_CACHE_DIR}/target_uniref.a3m"
          echo "==> Cached target MSA: ${TARGET_MSA_PATH}"
        else
          echo "==> Calibration did not find IntelliFold MSA CSV/A3M; proceeding without cached target MSA."
        fi
      fi

      # This is already a full predictor calibration run at max binder length.
      CAL_PRED_DONE=1
      ;;
  esac
fi

# Ensure an OpenFold-compatible target MSA cache exists whenever OpenFold is
# used in either design or post-prediction stages.
NEED_OPENFOLD_TARGET_MSA=0
if [[ "${PREDICTOR}" == "openfold-3-mlx" ]]; then
  NEED_OPENFOLD_TARGET_MSA=1
fi
if [[ "${NEED_OPENFOLD_TARGET_MSA}" -eq 0 ]]; then
  for _pp in "${POST_PREDICTORS[@]:-}"; do
    if [[ "${_pp}" == "openfold-3-mlx" ]]; then
      NEED_OPENFOLD_TARGET_MSA=1
      break
    fi
  done
fi

if [[ "${NEED_OPENFOLD_TARGET_MSA}" -eq 1 ]]; then
  if is_openfold_msa_path_compatible "${TARGET_MSA_PATH}"; then
    OPENFOLD_TARGET_MSA_PATH="${TARGET_MSA_PATH}"
  elif [[ -n "${TARGET_SEQ}" ]]; then
    [[ -n "${TARGET_CHAIN_ID}" ]] || TARGET_CHAIN_ID="B"
    echo "==> Preparing OpenFold-compatible target MSA cache for chain ${TARGET_CHAIN_ID}..."
    OPENFOLD_TARGET_MSA_PATH="$(generate_openfold_target_msa_cache "${TEMPLATE_YAML}" "${MSA_CACHE_DIR}" "${TARGET_SEQ}" "${TARGET_CHAIN_ID}")"
    echo "==> Cached OpenFold target MSA: ${OPENFOLD_TARGET_MSA_PATH}"
  else
    # Protein-ligand style inputs can have no target protein chain.
    OPENFOLD_TARGET_MSA_PATH=""
  fi

  if [[ "${PREDICTOR}" == "openfold-3-mlx" && -n "${OPENFOLD_TARGET_MSA_PATH}" ]]; then
    TARGET_MSA_PATH="${OPENFOLD_TARGET_MSA_PATH}"
  fi
fi

# Predictor-specific calibration run (one longest-length prediction) for all design tools.
if [[ "${CAL_PRED_DONE}" -eq 0 ]]; then
  echo "==> Calibration: running one ${PREDICTOR} prediction at binder length ${BINDER_MAX_LEN}..."
  run_predictor_calibration_once "${PREDICTOR}" "${TARGET_MSA_PATH}"
fi

CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
if [[ "${MAX_PARALLEL_USER}" == "auto" ]]; then
  if [[ "${MEM_BUDGET_GB}" == "auto" ]]; then
    MEM_BUDGET_MB="$(default_mem_budget_mb)"
  else
    MEM_BUDGET_MB="$(python3 - "${MEM_BUDGET_GB}" <<'PY'
import sys
gb=float(sys.argv[1]); print(max(1024, int(gb*1024)))
PY
)"
  fi
  SAFE_MB="$(floor_mul "${MEM_BUDGET_MB}" "${MEM_SAFETY}")"
  if [[ -z "${PEAK_RSS_MB}" || "${PEAK_RSS_MB}" == "0" ]]; then
    PEAK_RSS_MB=4096
  fi
  AUTO_MAX_BY_MEM="$(python3 - "${SAFE_MB}" "${PEAK_RSS_MB}" <<'PY'
import sys
safe=int(sys.argv[1]); peak=max(1,int(sys.argv[2]))
print(max(1, safe//peak))
PY
)"
  AUTO_MAX_BY_CPU="$(python3 - "${CPU_CORES}" "${MPS_CPU_CAP}" <<'PY'
import sys, math
cores=int(sys.argv[1]); cap_raw=sys.argv[2]
if cap_raw == "auto":
    cap=max(1,int(math.floor(cores*0.75)))
else:
    cap=max(1,int(float(cap_raw)))
print(max(1,min(cap,cores)))
PY
)"
  MAX_PARALLEL="$(python3 - "${AUTO_MAX_BY_MEM}" "${AUTO_MAX_BY_CPU}" "${N_RUNS}" <<'PY'
import sys
print(max(1, min(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))))
PY
)"
else
  MAX_PARALLEL="${MAX_PARALLEL_USER}"
fi

if [[ "${NO_PARALLEL}" -eq 1 ]]; then
  MAX_PARALLEL=1
fi

if [[ "$MAX_PARALLEL" == "auto" ]]; then
  MAX_PARALLEL=1
fi
if (( MAX_PARALLEL < 1 )); then MAX_PARALLEL=1; fi
if (( MAX_PARALLEL > N_RUNS )); then MAX_PARALLEL="$N_RUNS"; fi

echo "========================================"
echo "iProteinHunter"
echo "Repo root               : ${REPO_ROOT}"
echo "Run name                : ${RUN_NAME}"
echo "Predictor               : ${PREDICTOR}"
if [[ "$(post_predictors_count)" -gt 0 ]]; then
  echo "Post predictor(s)       : ${POST_PREDICTORS[*]}"
else
  echo "Post predictor(s)       : none"
fi
echo "Runs                    : ${N_RUNS}"
echo "Optimization cycles     : ${N_CYCLES} (plus cycle_00 seed)"
echo "MAX_PARALLEL            : ${MAX_PARALLEL}"
if [[ "${MAX_PARALLEL_USER}" == "auto" ]]; then
  echo "Auto parallel (mem/cpu) : ${AUTO_MAX_BY_MEM:-na}/${AUTO_MAX_BY_CPU:-na}"
  echo "Calibration peak RSS MB : ${PEAK_RSS_MB:-na}"
fi
echo "CPU only                : ${CPU_ONLY}"
echo "MPNN redesign model     : ${LIGANDMPNN_MODEL_LABEL}"
echo "MPNN temp cycle_00->01  : ${LIGAND_TEMP_CYCLE01}"
echo "MPNN temp other cycles  : ${LIGAND_TEMP_DEFAULT}"
echo "MPNN bias cycle_00->01  : ${LIGAND_BIAS_AA_CYCLE01:-none}"
echo "MPNN bias other cycles  : ${LIGAND_BIAS_AA_DEFAULT:-none}"
echo "Loop-kill constant      : ${LOOP_KILL}"
if [[ "${UNCONDITIONAL_MULTIMER_MODE}" -eq 1 ]]; then
  echo "Unconditional multimer  : ${UNCONDITIONAL_MULTIMER_SIZE}-mer (${UNCONDITIONAL_MULTIMER_CHAINS})"
fi
if [[ "${PREDICTOR}" == "boltz" ]]; then
  echo "Boltz design potentials : ${BOLTZ_USE_POTENTIALS_DEFAULT} (mode=${BOLTZ_USE_POTENTIALS_MODE})"
fi
echo "Target sequence length  : ${#TARGET_SEQ}"
if [[ -n "${TARGET_MSA_PATH}" ]]; then
  echo "Target MSA path         : ${TARGET_MSA_PATH}"
else
  echo "Target MSA path         : (none in template; tool default behavior)"
fi
if [[ -n "${OPENFOLD_TARGET_MSA_PATH:-}" ]]; then
  echo "OpenFold target MSA     : ${OPENFOLD_TARGET_MSA_PATH}"
fi
echo "========================================"

for run_index in $(seq 1 "${N_RUNS}"); do
  wait_for_slot "${MAX_PARALLEL}"
  (
    run_one_design "${run_index}" "${TARGET_MSA_PATH}"
  ) &
done
wait

echo "run,cycle,iptm,complex_plddt,binder_sequence,structure_path,confidence_json" > "${EXPT_ROOT}/summary_all_runs.csv"
for run_index in $(seq 1 "${N_RUNS}"); do
  run_tag="$(printf "run_%03d" "$run_index")"
  if [[ -f "${EXPT_ROOT}/${run_tag}/metrics_per_cycle.csv" ]]; then
    tail -n +2 "${EXPT_ROOT}/${run_tag}/metrics_per_cycle.csv" | awk -F',' -v r="$run_index" 'BEGIN{OFS=","} {print r,$1,$2,$3,$6,$5,$4}' >> "${EXPT_ROOT}/summary_all_runs.csv"
  fi
done

echo "run,start_ts,end_ts,duration_sec" > "${EXPT_ROOT}/summary_timing_design.csv"
for run_index in $(seq 1 "${N_RUNS}"); do
  run_tag="$(printf "run_%03d" "$run_index")"
  if [[ -f "${EXPT_ROOT}/${run_tag}/timing_run.csv" ]]; then
    tail -n +2 "${EXPT_ROOT}/${run_tag}/timing_run.csv" >> "${EXPT_ROOT}/summary_timing_design.csv"
  fi
done

if [[ "$(post_predictors_count)" -gt 0 ]]; then
  for post_pred in "${POST_PREDICTORS[@]}"; do
    pred_safe="$(safe_predictor_name "$post_pred")"
    task_tsv="${EXPT_ROOT}/post_${pred_safe}_tasks.tsv"

    python3 - "${EXPT_ROOT}/summary_all_runs.csv" "${POST_MODE}" "${POST_IPTM_THRESHOLD}" "${POST_INCLUDE_CYCLE00}" "${task_tsv}" <<'PY'
import csv, sys
summary, mode, thr, include0, out_tsv = sys.argv[1:6]
thr=float(thr)
include0=(include0 == "1")
rows=[]
with open(summary) as f:
    r=csv.DictReader(f)
    for row in r:
        cyc=str(row.get("cycle",""))
        try:
            cnum=int(cyc)
        except Exception:
            continue
        if cnum == 0 and not include0:
            continue
        if mode == "iptm":
            try:
                if float(row.get("iptm","nan")) < thr:
                    continue
            except Exception:
                continue
        rows.append((int(row["run"]), cnum, row.get("binder_sequence","")))
rows.sort()
with open(out_tsv,"w") as f:
    for r,c,seq in rows:
        f.write(f"{r}\t{c}\t{seq}\n")
print(len(rows))
PY

    if [[ -s "$task_tsv" ]]; then
      while IFS=$'\t' read -r run_i cyc_i bseq; do
        wait_for_slot "${MAX_PARALLEL}"
        (
          run_post_task "$post_pred" "$run_i" "$cyc_i" "$bseq" "$TARGET_MSA_PATH"
        ) &
      done < "$task_tsv"
      wait
    fi

    aggregate_post_predictor "$post_pred"
  done
fi

build_comparison_tables

echo
echo "Done."
echo "Output root: ${EXPT_ROOT}"
echo "Design summary: ${EXPT_ROOT}/summary_all_runs.csv"
if [[ "$(post_predictors_count)" -gt 0 ]]; then
  for pp in "${POST_PREDICTORS[@]}"; do
    psafe="$(safe_predictor_name "$pp")"
    echo "Post summary (${pp}): ${EXPT_ROOT}/summary_post_${psafe}.csv"
  done
fi
