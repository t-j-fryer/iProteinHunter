#!/usr/bin/env bash
set -euo pipefail

########################################
# iProteinHunter Boltz + LigandMPNN pipeline
# (macOS Apple Silicon, venv-based install)
#
# Defaults assume this script lives somewhere inside the iProteinHunter repo.
# It auto-detects REPO_ROOT as the directory containing this script.
########################################

########################################
# USER CONFIG (defaults)
########################################

# Repo root = folder containing THIS script, unless overridden by env var.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${IPROTEINHUNTER_ROOT:-$SCRIPT_DIR}"

# --- Boltz setup (venv) ---
BOLTZ_VENV="${REPO_ROOT}/venvs/iProteinHunter_boltz"
BOLTZ_CLI="boltz"

# --- LigandMPNN setup (venv + repo path) ---
LIGAND_VENV="${REPO_ROOT}/venvs/iProteinHunter_ligandmpnn"
LIGANDMPNN_REPO="${REPO_ROOT}/src/LigandMPNN"
LIGANDMPNN_RUN="python run.py"

# --- Input / Output defaults (CLI-overridable) ---
TEMPLATE_YAML="${REPO_ROOT}/examples/aCbx_bind.yaml"
BASE_RUN_ROOT="${REPO_ROOT}/output"

# Run defaults
RUN_NAME="test_run"
N_RUNS=3
N_CYCLES=5

# Binder sequence options (per run, chain A)
BINDER_MIN_LEN=65
BINDER_MAX_LEN=150
BINDER_PERCENT_X=50   # % X, enforced via n_X = round(L * P/100)

# Boltz CLI default options
# Note: Boltz docs define --accelerator [gpu,cpu,tpu] and gpu is default.
# On Apple Silicon, "gpu" is the correct CLI value for using hardware accel.
BOLTZ_EXTRA_FLAGS_DEFAULT=(
  "--accelerator" "gpu"
  "--devices" "1"
  "--use_msa_server"
  "--msa_server_url" "https://api.colabfold.com"
  "--msa_pairing_strategy" "greedy"
)

# LigandMPNN CLI default options
LIGANDMPNN_EXTRA_FLAGS_DEFAULT=(
  "--seed" "111"
  "--chains_to_design" "A"
  "--omit_AA" "C"
)

# CLI extra flags (strings; will be split into arrays)
BOLTZ_EXTRA_CLI_STRING=""
LIGAND_EXTRA_CLI_STRING=""

# Global target MSA CSV file discovered after run_001/cycle_01
FIRST_TARGET_MSA_PATH=""

########################################
# CLI PARSING
########################################

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-name NAME          Experiment name (default: ${RUN_NAME})
  --num-runs N             Number of runs (default: ${N_RUNS})
  --num-cycles N           Cycles per run (default: ${N_CYCLES})

  --template-yaml PATH     Input Boltz YAML template (default: ${TEMPLATE_YAML})
                           - Must contain chains A (binder) and B/C/... (target).
                           - Only chain A's sequence is modified.

  --out-root PATH          Base output directory (default: ${BASE_RUN_ROOT})

  --binder-min-len N       Min binder length (default: ${BINDER_MIN_LEN})
  --binder-max-len N       Max binder length (default: ${BINDER_MAX_LEN})
  --binder-percent-x P     Percent of binder positions set to 'X' (default: ${BINDER_PERCENT_X})
                           (n_X = round(L * P/100))

  --boltz-extra "ARGS"     Extra flags appended to Boltz
  --ligand-extra "ARGS"    Extra flags appended to LigandMPNN

  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --num-runs) N_RUNS="$2"; shift 2 ;;
    --num-cycles) N_CYCLES="$2"; shift 2 ;;
    --template-yaml) TEMPLATE_YAML="$2"; shift 2 ;;
    --out-root) BASE_RUN_ROOT="$2"; shift 2 ;;
    --binder-min-len) BINDER_MIN_LEN="$2"; shift 2 ;;
    --binder-max-len) BINDER_MAX_LEN="$2"; shift 2 ;;
    --binder-percent-x) BINDER_PERCENT_X="$2"; shift 2 ;;
    --boltz-extra) BOLTZ_EXTRA_CLI_STRING="$2"; shift 2 ;;
    --ligand-extra) LIGAND_EXTRA_CLI_STRING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Build final flag arrays
BOLTZ_EXTRA_FLAGS=("${BOLTZ_EXTRA_FLAGS_DEFAULT[@]}")
if [[ -n "${BOLTZ_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  BOLTZ_EXTRA_FLAGS_CLI=(${BOLTZ_EXTRA_CLI_STRING})
  BOLTZ_EXTRA_FLAGS+=("${BOLTZ_EXTRA_FLAGS_CLI[@]}")
fi

LIGANDMPNN_EXTRA_FLAGS=("${LIGANDMPNN_EXTRA_FLAGS_DEFAULT[@]}")
if [[ -n "${LIGAND_EXTRA_CLI_STRING}" ]]; then
  # shellcheck disable=SC2206
  LIGAND_EXTRA_FLAGS_CLI=(${LIGAND_EXTRA_CLI_STRING})
  LIGANDMPNN_EXTRA_FLAGS+=("${LIGAND_EXTRA_FLAGS_CLI[@]}")
fi

########################################
# PREFLIGHT
########################################

die() { echo "ERROR: $*" 1>&2; exit 1; }

[[ -d "${REPO_ROOT}" ]] || die "REPO_ROOT not found: ${REPO_ROOT}"
[[ -x "${BOLTZ_VENV}/bin/python" ]] || die "Boltz venv not found: ${BOLTZ_VENV}"
[[ -x "${LIGAND_VENV}/bin/python" ]] || die "LigandMPNN venv not found: ${LIGAND_VENV}"
[[ -f "${TEMPLATE_YAML}" ]] || die "Template YAML not found: ${TEMPLATE_YAML}"
[[ -f "${LIGANDMPNN_REPO}/run.py" ]] || die "LigandMPNN run.py not found: ${LIGANDMPNN_REPO}/run.py"

mkdir -p "${BASE_RUN_ROOT}"

########################################
# HELPER FUNCTIONS
########################################

# Generate binder seq with exact-ish %X, no C
generate_random_binder_seq () {
  local min_len="$1"
  local max_len="$2"
  local percent_x="$3"

  python3 - "$min_len" "$max_len" "$percent_x" << 'PY'
import sys, random

min_len, max_len, pct_x = sys.argv[1:4]
min_len = int(min_len)
max_len = int(max_len)
pct_x = float(pct_x)

if max_len < min_len:
    raise SystemExit("binder-max-len must be >= binder-min-len")

L = random.randint(min_len, max_len)
p_x = max(0.0, min(1.0, pct_x / 100.0))
n_X = int(round(L * p_x))
n_X = max(0, min(L, n_X))

aas = "ADEFGHIKLMNPQRSTVWY"  # 19 AAs, no C
seq = [random.choice(aas) for _ in range(L)]

indices = list(range(L))
random.shuffle(indices)
for i in indices[:n_X]:
    seq[i] = "X"

print("".join(seq))
PY
}

# YAML: replace sequence of chain A only; optionally inject msa: <msa_path> for non-A chains
make_yaml_with_binder_sequence () {
  local template_yaml="$1"
  local out_yaml="$2"
  local new_seq="$3"
  local msa_path="$4"  # can be empty string

  python3 - "$template_yaml" "$out_yaml" "$new_seq" "$msa_path" << 'PY'
import sys

template, out, new_seq, msa_path = sys.argv[1:5]
msa_path = msa_path or None

in_binder = False

with open(template) as fin, open(out, "w") as fout:
    for line in fin:
        stripped = line.strip()

        # Detect chain id
        if stripped.startswith("id:"):
            parts = stripped.split("id:", 1)[1].strip()
            chain_id = parts.strip(" '\"")
            in_binder = (chain_id == "A")
            fout.write(line)
            continue

        # Replace binder sequence (chain A)
        if in_binder and stripped.startswith("sequence:"):
            indent = line.split("sequence:")[0]
            fout.write(f"{indent}sequence: {new_seq}\n")
            continue

        # For non-binder chains, if msa_path is given:
        if not in_binder and msa_path is not None:
            # If template already has msa:, overwrite it
            if stripped.startswith("msa:"):
                indent = line.split("msa:")[0]
                fout.write(f"{indent}msa: {msa_path}\n")
                continue
            # If we see sequence: and no msa yet, inject msa right after
            if stripped.startswith("sequence:"):
                fout.write(line)
                indent = line.split("sequence:")[0]
                fout.write(f"{indent}msa: {msa_path}\n")
                continue

        # Default: pass through line unchanged
        fout.write(line)
PY
}

# Patch CIF: UNK -> ALA (for cycle_01 of each run)
patch_cif_unk_to_ala () {
  local in_cif="$1"
  local out_cif="$2"

  python3 - "$in_cif" "$out_cif" << 'PY'
import sys
src, dst = sys.argv[1:3]
with open(src) as f:
    data = f.read()
patched = data.replace(" UNK ", " ALA ")
with open(dst, "w") as f:
    f.write(patched)
PY
}

# Extract redesigned binder (chain A) from LigandMPNN FASTAs
extract_redesigned_sequence_from_fastas () {
  python3 - "$@" << 'PY'
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    raise SystemExit("No FASTA files passed.")

def iter_seqs(path: Path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line)
        if seq:
            yield "".join(seq)

all_seqs = []
for p in paths:
    all_seqs.extend(iter_seqs(p))

if not all_seqs:
    raise SystemExit("No sequences found in FASTA files.")

# LigandMPNN: first seq = input, second = redesigned
if len(all_seqs) >= 2:
    seq_full = all_seqs[1]
else:
    seq_full = all_seqs[0]

# Multi-chain: "A:B:C:D" — we just want chain A
if ":" in seq_full:
    binder_seq = seq_full.split(":", 1)[0]
else:
    binder_seq = seq_full

print(binder_seq)
PY
}

# Extract iPTM from Boltz confidence JSON
get_iptm_from_conf_json () {
  local json_path="$1"

  python3 - "$json_path" << 'PY'
import sys, json

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

def find_iptm(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in {"iptm", "i_ptm", "iptm+ptm", "iptm_score"}:
                return v
            val = find_iptm(v)
            if val is not None:
                return val
    elif isinstance(obj, list):
        for v in obj:
            val = find_iptm(v)
            if val is not None:
                return val
    return None

val = find_iptm(d)
if val is None:
    print("nan")
else:
    try:
        print(float(val))
    except Exception:
        print("nan")
PY
}

# Plot iPTM vs cycle from per-run CSV
plot_iptm_vs_cycle () {
  local csv_path="$1"
  local out_png="$2"

  python3 - "$csv_path" "$out_png" << 'PY'
import sys, csv
import matplotlib.pyplot as plt

csv_path, out_png = sys.argv[1:3]

cycles = []
iptms = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            c = int(row["cycle"])
            ip = float(row["iptm"])
        except (ValueError, KeyError):
            continue
        if str(row["iptm"]).lower() == "nan":
            continue
        cycles.append(c)
        iptms.append(ip)

if not cycles:
    sys.exit(0)

plt.figure()
plt.plot(cycles, iptms, marker="o")
plt.xlabel("Cycle")
plt.ylabel("iPTM")
plt.title("Boltz iPTM vs Cycle")
plt.xticks(cycles)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
PY
}

# Plot global iPTM vs run (each cycle is a coloured dot with legend)
plot_iptm_vs_run () {
  local csv_path="$1"
  local out_png="$2"

  python3 - "$csv_path" "$out_png" << 'PY'
import sys, csv
import matplotlib.pyplot as plt

csv_path, out_png = sys.argv[1:3]

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            run = int(row["run"])
            cycle = int(row["cycle"])
            iptm = float(row["iptm"])
        except Exception:
            continue
        if str(row["iptm"]).lower() == "nan":
            continue
        rows.append((run, cycle, iptm))

if not rows:
    sys.exit(0)

cycles = sorted({c for _, c, _ in rows})

colormap = plt.get_cmap("tab10")
colors = {cycle: colormap((cycle - 1) % 10) for cycle in cycles}

plt.figure(figsize=(7,5))
for run, cycle, iptm in rows:
    plt.scatter(run, iptm, color=colors[cycle], s=40, alpha=0.9)

handles = [
    plt.Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=colors[c], markeredgecolor='none',
               markersize=8, label=f"Cycle {c}")
    for c in cycles
]

plt.legend(handles=handles, title="Cycle", loc="best", frameon=True)
plt.xlabel("Run")
plt.ylabel("iPTM")
plt.title("Boltz iPTM per Run (each cycle = coloured dot)")
plt.xticks(sorted(sorted(set(r for r, _, _ in rows))))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
PY
}

########################################
# MAIN
########################################

echo "========================================"
echo "iProteinHunter: Boltz + LigandMPNN (Binder mode)"
echo "Repo root         : ${REPO_ROOT}"
echo "Boltz venv         : ${BOLTZ_VENV}"
echo "LigandMPNN venv    : ${LIGAND_VENV}"
echo "LigandMPNN repo    : ${LIGANDMPNN_REPO}"
echo "Run name           : ${RUN_NAME}"
echo "Number of runs     : ${N_RUNS}"
echo "Cycles per run     : ${N_CYCLES}"
echo "Base run root      : ${BASE_RUN_ROOT}"
echo "Template YAML      : ${TEMPLATE_YAML}"
echo "Binder len range   : ${BINDER_MIN_LEN}-${BINDER_MAX_LEN}"
echo "Binder % X         : ${BINDER_PERCENT_X}%"
echo "========================================"
echo

mkdir -p "${BASE_RUN_ROOT}/${RUN_NAME}"

# Global CSV for all runs/cycles (now with binder_sequence)
GLOBAL_CSV="${BASE_RUN_ROOT}/${RUN_NAME}/all_runs_iptm.csv"
echo "run,cycle,iptm,binder_sequence" > "${GLOBAL_CSV}"

for RUN_INDEX in $(seq 1 "${N_RUNS}"); do
  RUN_TAG=$(printf "run_%03d" "${RUN_INDEX}")
  RUN_ROOT="${BASE_RUN_ROOT}/${RUN_NAME}/${RUN_TAG}"

  echo "########################################"
  echo ">>> Starting ${RUN_TAG} (${RUN_NAME})"
  echo "    Run root: ${RUN_ROOT}"
  echo "########################################"
  echo

  mkdir -p "${RUN_ROOT}"

  METRICS_CSV="${RUN_ROOT}/boltz_iptm_per_cycle.csv"
  echo "cycle,iptm,confidence_json,structure_path" > "${METRICS_CSV}"

  echo "Generating random binder sequence for ${RUN_TAG}..."
  INITIAL_BINDER_SEQ="$(generate_random_binder_seq "${BINDER_MIN_LEN}" "${BINDER_MAX_LEN}" "${BINDER_PERCENT_X}")"
  echo "Initial binder length: ${#INITIAL_BINDER_SEQ}"
  echo "Initial binder (first 60 aa): ${INITIAL_BINDER_SEQ:0:60}..."
  echo

  CURRENT_SEQ="${INITIAL_BINDER_SEQ}"

  for CYCLE in $(seq 1 "${N_CYCLES}"); do
    CYCLE_TAG=$(printf "cycle_%02d" "${CYCLE}")
    CYCLE_DIR="${RUN_ROOT}/${CYCLE_TAG}"
    BOLTZ_OUT="${CYCLE_DIR}/boltz"
    LIGAND_OUT="${CYCLE_DIR}/ligandmpnn"
    CYCLE_YAML="${CYCLE_DIR}/boltz_input.yaml"

    echo "==============================="
    echo ">>> ${RUN_TAG} - ${CYCLE_TAG}"
    echo "==============================="

    mkdir -p "${BOLTZ_OUT}" "${LIGAND_OUT}"

    # Decide whether to inject target MSA path
    MSA_PATH_TO_USE=""
    if [[ -n "${FIRST_TARGET_MSA_PATH}" ]]; then
      if ! { [[ "${RUN_INDEX}" -eq 1 ]] && [[ "${CYCLE}" -eq 1 ]]; }; then
        MSA_PATH_TO_USE="${FIRST_TARGET_MSA_PATH}"
      fi
    fi

    echo "Writing YAML for ${CYCLE_TAG} with binder sequence for chain A."
    [[ -n "${CURRENT_SEQ}" ]] || die "CURRENT_SEQ is empty on cycle ${CYCLE}"
    make_yaml_with_binder_sequence "${TEMPLATE_YAML}" "${CYCLE_YAML}" "${CURRENT_SEQ}" "${MSA_PATH_TO_USE}"

    # --- Boltz prediction ---
    echo "Running Boltz prediction for ${RUN_TAG} ${CYCLE_TAG}..."
    # shellcheck source=/dev/null
    source "${BOLTZ_VENV}/bin/activate"

    ${BOLTZ_CLI} predict "${CYCLE_YAML}" \
      --out_dir "${BOLTZ_OUT}" \
      "${BOLTZ_EXTRA_FLAGS[@]}"

    deactivate

    # Structure file (CIF or PDB)
    STRUCT_PATH="$(find "${BOLTZ_OUT}" -type f \( -name '*.pdb' -o -name '*.cif' \) | sort | head -n 1 || true)"
    [[ -n "${STRUCT_PATH}" ]] || die "No structure (PDB/CIF) found in ${BOLTZ_OUT}"
    echo "Boltz structure for ${RUN_TAG} ${CYCLE_TAG}: ${STRUCT_PATH}"

    # Discover target MSA CSV on first run/cycle
    if [[ -z "${FIRST_TARGET_MSA_PATH}" && "${RUN_INDEX}" -eq 1 && "${CYCLE}" -eq 1 ]]; then
      FIRST_TARGET_MSA_PATH="$(find "${BOLTZ_OUT}" -type f -name '*.csv' -path '*/msa/*' | head -n 1 || true)"
      if [[ -n "${FIRST_TARGET_MSA_PATH}" ]]; then
        echo "Discovered target MSA CSV (to reuse later): ${FIRST_TARGET_MSA_PATH}"
      else
        echo "WARNING: Could not find MSA CSV in ${BOLTZ_OUT}; target MSAs will not be reused."
      fi
    fi

    # Confidence JSON -> iPTM
    CONF_JSON="$(find "${BOLTZ_OUT}" -type f -name 'confidence_*_model_0.json' | sort | head -n 1 || true)"
    if [[ -n "${CONF_JSON}" ]]; then
      IPTM_VALUE="$(get_iptm_from_conf_json "${CONF_JSON}")"
    else
      IPTM_VALUE="nan"
    fi
    echo "iPTM for ${RUN_TAG} ${CYCLE_TAG}: ${IPTM_VALUE}"

    echo "${CYCLE},${IPTM_VALUE},${CONF_JSON},${STRUCT_PATH}" >> "${METRICS_CSV}"
    echo "${RUN_INDEX},${CYCLE},${IPTM_VALUE},${CURRENT_SEQ}" >> "${GLOBAL_CSV}"

    # CIF UNK->ALA patch ONLY for cycle_01 of each run
    if [[ "${STRUCT_PATH##*.}" == "cif" && "${CYCLE}" -eq 1 ]]; then
      PATCHED_PATH="${STRUCT_PATH%.cif}_ALA.cif"
      echo "Patching CIF UNK->ALA for cycle_01: ${PATCHED_PATH}"
      patch_cif_unk_to_ala "${STRUCT_PATH}" "${PATCHED_PATH}"
      STRUCT_FOR_LMPNN="${PATCHED_PATH}"
    else
      STRUCT_FOR_LMPNN="${STRUCT_PATH}"
    fi

    # --- LigandMPNN ---
    echo "Running LigandMPNN for ${RUN_TAG} ${CYCLE_TAG}..."

    # Convert CIF -> PDB via gemmi if needed (use Boltz venv python which has gemmi)
    if [[ "${STRUCT_FOR_LMPNN##*.}" == "cif" ]]; then
      LMPNN_INPUT="${STRUCT_FOR_LMPNN%.cif}.pdb"
      "${BOLTZ_VENV}/bin/python" - "$STRUCT_FOR_LMPNN" "$LMPNN_INPUT" << 'PY'
import sys, gemmi
in_path, out_path = sys.argv[1:3]
st = gemmi.read_structure(in_path)

for model in st:
    for chain in list(model):
        if len(chain) == 0:
            model.remove_chain(chain)
for model in st:
    for chain in model:
        for res in list(chain):
            if len(res) == 0:
                chain.remove_residue(res)

st.write_pdb(out_path)
PY
    else
      LMPNN_INPUT="${STRUCT_FOR_LMPNN}"
    fi

    # Run LigandMPNN in its own venv
    # shellcheck source=/dev/null
    source "${LIGAND_VENV}/bin/activate"
    pushd "${LIGANDMPNN_REPO}" >/dev/null
    ${LIGANDMPNN_RUN} \
      --pdb_path "${LMPNN_INPUT}" \
      --out_folder "${LIGAND_OUT}" \
      "${LIGANDMPNN_EXTRA_FLAGS[@]}"
    popd >/dev/null
    deactivate

    SEQS_DIR="${LIGAND_OUT}/seqs"
    [[ -d "${SEQS_DIR}" ]] || die "Expected LigandMPNN seqs dir not found: ${SEQS_DIR}"

    echo "Extracting redesigned binder sequence from ${SEQS_DIR}..."
    FASTA_GLOB=( "${SEQS_DIR}"/*.fa )
    [[ -e "${FASTA_GLOB[0]}" ]] || die "No FASTA (*.fa) found in ${SEQS_DIR}"

    CURRENT_SEQ="$(extract_redesigned_sequence_from_fastas "${SEQS_DIR}"/*.fa)"
    echo "New binder length: ${#CURRENT_SEQ}"
    echo "Binder (first 60 aa): ${CURRENT_SEQ:0:60}..."
    echo "Completed ${RUN_TAG} ${CYCLE_TAG}."
    echo
  done

  echo ">>> Finished ${RUN_TAG} (${RUN_NAME})"
  echo "Final binder length: ${#CURRENT_SEQ}"
  echo "Final binder (first 120 aa): ${CURRENT_SEQ:0:120}..."
  echo

  PLOT_PNG="${RUN_ROOT}/iptm_vs_cycle.png"
  echo "Plotting iPTM vs cycle for ${RUN_TAG} -> ${PLOT_PNG}"
  plot_iptm_vs_cycle "${METRICS_CSV}" "${PLOT_PNG}"
done

GLOBAL_PLOT="${BASE_RUN_ROOT}/${RUN_NAME}/iptm_vs_run_all_cycles.png"
echo "Plotting global iPTM vs run -> ${GLOBAL_PLOT}"
plot_iptm_vs_run "${GLOBAL_CSV}" "${GLOBAL_PLOT}"

echo "All ${N_RUNS} runs complete for experiment '${RUN_NAME}'."
echo "Global CSV:   ${GLOBAL_CSV}"
echo "Global plot:  ${GLOBAL_PLOT}"
