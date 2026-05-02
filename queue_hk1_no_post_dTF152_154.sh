#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/thomasfryer/iProteinHunter"
RUN_SCRIPT="${REPO_ROOT}/iproteinhunter_run.sh"
TEMPLATE_YAML="${REPO_ROOT}/examples/SUMO.yaml"
OUT_ROOT="${REPO_ROOT}/output"
QUEUE_LOG="${OUT_ROOT}/queue_hk1_no_post_dTF152_154.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${QUEUE_LOG}"
}

run_campaign() {
  local run_name="$1"
  local predictor="$2"

  mkdir -p "${OUT_ROOT}/${run_name}"
  log "Starting ${run_name} (predictor=${predictor}, post=none, helix_kill=1)"
  caffeinate -dims "${RUN_SCRIPT}" \
    --template-yaml "${TEMPLATE_YAML}" \
    --run-name "${run_name}" \
    --predictor "${predictor}" \
    --post-mode none \
    --post-predictor none \
    --num-runs 50 \
    --num-opt-cycles 5 \
    --binder-min-len 65 \
    --binder-max-len 95 \
    --max-parallel 4 \
    --helix-kill \
    --negative-helix-constant 1 \
    > "${OUT_ROOT}/${run_name}/launch.log" 2>&1
  log "Finished ${run_name}"
}

mkdir -p "${OUT_ROOT}"
log "Queue worker started."
log "Campaign plan: dTF152(boltz) -> dTF153(intellifold) -> dTF154(openfold-3-mlx)"

run_campaign "dTF152" "boltz"
run_campaign "dTF153" "intellifold"
run_campaign "dTF154" "openfold-3-mlx"

log "Queue worker completed all campaigns."
