#!/usr/bin/env bash
set -euo pipefail

########################################
# Config
########################################

IPROTEINHUNTER_ROOT="${IPROTEINHUNTER_ROOT:-$HOME/iProteinHunter}"

PYTHON_BIN="python3.11"

BOLTZ_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_boltz"
LIGAND_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_ligandmpnn"
INTELLIFOLD_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_intellifold"
OPENFOLD_VENV="${IPROTEINHUNTER_ROOT}/venvs/iProteinHunter_openfold3_mlx"

SRC_DIR="${IPROTEINHUNTER_ROOT}/src"
LIGANDMPNN_REPO="${SRC_DIR}/LigandMPNN"
INTELLIFOLD_REPO="${SRC_DIR}/IntelliFold"
OPENFOLD_REPO="${SRC_DIR}/openfold-3-mlx"

RUNNER="${IPROTEINHUNTER_ROOT}/iproteinhunter_run.sh"

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
# Clone IntelliFold (upstream)
########################################

if [[ ! -d "${INTELLIFOLD_REPO}" ]]; then
  git clone https://github.com/IntelliGen-AI/IntelliFold.git "${INTELLIFOLD_REPO}"
else
  echo "IntelliFold repo already exists, updating..."
  git -C "${INTELLIFOLD_REPO}" pull --ff-only
fi

########################################
# Clone openfold-3-mlx
########################################

if [[ ! -d "${OPENFOLD_REPO}" ]]; then
  git clone https://github.com/latent-spacecraft/openfold-3-mlx.git "${OPENFOLD_REPO}"
else
  echo "openfold-3-mlx repo already exists, updating..."
  if git -C "${OPENFOLD_REPO}" rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    git -C "${OPENFOLD_REPO}" pull --ff-only
  else
    git -C "${OPENFOLD_REPO}" fetch origin
    if git -C "${OPENFOLD_REPO}" show-ref --verify --quiet refs/remotes/origin/main; then
      git -C "${OPENFOLD_REPO}" branch --set-upstream-to=origin/main >/dev/null 2>&1 || true
      git -C "${OPENFOLD_REPO}" pull --ff-only || true
    else
      echo "⚠️  Warning: origin/main not found; skipping pull for openfold-3-mlx"
    fi
  fi
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

python - "${MODEL_DIR}" <<'PY'
import sys, urllib.request, pathlib

urls = {
  "proteinmpnn_v_48_020.pt":
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
  "ligandmpnn_v_32_010_25.pt":
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
}

outdir = pathlib.Path(sys.argv[1])
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
# IntelliFold venv
########################################

echo
echo "==> Installing IntelliFold env..."

if [[ ! -d "${INTELLIFOLD_VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${INTELLIFOLD_VENV}"
fi

source "${INTELLIFOLD_VENV}/bin/activate"

pip install --upgrade pip

# Torch for Apple Silicon (MPS)
pip install torch==2.6.0

# Install IntelliFold (editable) without CUDA-only deps
pip install -e "${INTELLIFOLD_REPO}" --no-deps

# Runtime deps (no CUDA/DeepSpeed)
pip install \
  accelerate==1.1.1 biopython==1.85 click==8.1.8 \
  einops==0.8.0 einx==0.3.0 ihm==2.5 mashumaro==3.14 \
  ml_collections==1.0.0 modelcif==1.2 networkx==3.4.2 \
  numba==0.61.0 numpy==1.24.0 pandas==2.2.3 pyyaml==6.0.2 \
  rdkit==2024.3.2 requests==2.32.3 scipy==1.14.1 \
  torchdiffeq==0.2.5 tqdm==4.67.1 fsspec==2025.3.0

python - <<'PY'
import torch
print(f"torch: {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
PY

deactivate

########################################
# openfold-3-mlx venv
########################################

echo
echo "==> Installing openfold-3-mlx env..."

if [[ ! -d "${OPENFOLD_VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${OPENFOLD_VENV}"
fi

source "${OPENFOLD_VENV}/bin/activate"

pip install --upgrade pip

# Torch for Apple Silicon (MPS)
pip install torch==2.6.0

# Install openfold-3-mlx (editable)
pip install -e "${OPENFOLD_REPO}"

deactivate

########################################
# Make runner executable
########################################

echo
echo "==> Finalizing install..."

if [[ -f "${RUNNER}" ]]; then
  chmod +x "${RUNNER}"
  echo "✓ Made $(basename "${RUNNER}") executable"
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
