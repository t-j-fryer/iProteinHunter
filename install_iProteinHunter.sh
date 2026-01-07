#!/usr/bin/env bash
set -euo pipefail

########################################
# Config
########################################

IPROTEINHUNTER_ROOT="${IPROTEINHUNTER_ROOT:-$HOME/iProteinHunter}"

PYTHON_BIN="python3.11"

BOLTZ_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_boltz"
LIGAND_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_ligandmpnn"

SRC_DIR="${IPROTEINHUNTER_ROOT}/src"
LIGANDMPNN_REPO="${SRC_DIR}/LigandMPNN"

RUNNER="${IPROTEINHUNTER_ROOT}/iProteinHunter_run.sh"

########################################
# Preflight
########################################

echo "==> iProteinHunter root: ${IPROTEINHUNTER_ROOT}"

command -v git >/dev/null 2>&1 || {
  echo "ERROR: git not found. Install Xcode Command Line Tools first:"
  echo "  xcode-select --install"
  exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "ERROR: ${PYTHON_BIN} not found."
  echo "Install Python 3.11 (e.g. via python.org or pyenv)."
  exit 1
}

"${PYTHON_BIN}" - <<'PY'
import sys
assert sys.version_info[:2] == (3,11), "Python 3.11 required"
print(f"Python OK: {sys.version.split()[0]}")
PY

mkdir -p "${IPROTEINHUNTER_ROOT}/"{venvs,src,examples,output}

########################################
# Boltz venv
########################################

echo
echo "==> Installing Boltz env..."

if [[ ! -d "${BOLTZ_VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${BOLTZ_VENV}"
fi

source "${BOLTZ_VENV}/bin/activate"

pip install --upgrade pip

# Torch for Apple Silicon (MPS), no CUDA
pip install torch

# Boltz itself
pip install boltz

python - <<'PY'
import torch
print(f"torch: {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
PY

deactivate

########################################
# LigandMPNN venv
########################################

echo
echo "==> Installing LigandMPNN env..."

if [[ ! -d "${LIGAND_VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${LIGAND_VENV}"
fi

source "${LIGAND_VENV}/bin/activate"

pip install --upgrade pip

# Pin torch exactly as LigandMPNN expects (CPU/MPS build, no CUDA)
pip install torch==2.2.1

########################################
# Clone LigandMPNN
########################################

if [[ ! -d "${LIGANDMPNN_REPO}" ]]; then
  git clone https://github.com/dauparas/LigandMPNN.git "${LIGANDMPNN_REPO}"
else
  echo "LigandMPNN repo already exists, updating..."
  git -C "${LIGANDMPNN_REPO}" pull --ff-only
fi

########################################
# Filter LigandMPNN requirements (no CUDA)
########################################

REQ_IN="${LIGANDMPNN_REPO}/requirements.txt"
REQ_OUT="${LIGANDMPNN_REPO}/requirements.macos_nocuda.txt"

echo "==> Creating filtered LigandMPNN requirements (no CUDA): ${REQ_OUT}"

grep -Ev 'cuda|cublas|cudnn|nccl|nvidia|triton' "${REQ_IN}" > "${REQ_OUT}"

pip install -r "${REQ_OUT}"

########################################
# Download LigandMPNN weights
########################################

MODEL_DIR="${LIGANDMPNN_REPO}/model_params"
mkdir -p "${MODEL_DIR}"

echo "==> Downloading LigandMPNN weights..."

python - <<'PY'
import urllib.request, os, pathlib

urls = {
  "proteinmpnn_v_48_020.pt":
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
  "ligandmpnn_v_32_010.pt":
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010.pt",
}

outdir = pathlib.Path(os.environ["MODEL_DIR"])
for name, url in urls.items():
    path = outdir / name
    if path.exists():
        print(f"✓ {name} already present")
        continue
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, path)
PY

deactivate

########################################
# Make runner executable
########################################

echo
echo "==> Finalizing install..."

if [[ -f "${RUNNER}" ]]; then
  chmod +x "${RUNNER}"
  echo "✓ Made iProteinHunter_run.sh executable"
else
  echo "⚠️  Warning: ${RUNNER} not found"
fi

########################################
# Done
########################################

echo
echo "✅ iProteinHunter installation complete."
echo
echo "Run with:"
echo "  ${RUNNER} --help"
echo
echo "Defaults:"
echo "  examples/: ${IPROTEINHUNTER_ROOT}/examples"
echo "  output/:   ${IPROTEINHUNTER_ROOT}/output"
echo
