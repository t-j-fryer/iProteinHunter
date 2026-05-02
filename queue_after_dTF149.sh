#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/thomasfryer/iProteinHunter"
RUN_SCRIPT="${REPO_ROOT}/iproteinhunter_run.sh"
TEMPLATE_YAML="${REPO_ROOT}/examples/SUMO.yaml"
OUT_ROOT="${REPO_ROOT}/output"

WAIT_RUN="dTF149"
FIRST_RUN="dTF150"
SECOND_RUN="dTF151"

QUEUE_LOG="${OUT_ROOT}/queue_after_dTF149.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${QUEUE_LOG}"
}

is_run_active() {
  local run_name="$1"
  if pgrep -f "iproteinhunter_run.sh .*--run-name ${run_name}" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "caffeinate -dims .*--run-name ${run_name}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

wait_for_run_to_finish() {
  local run_name="$1"
  while is_run_active "${run_name}"; do
    log "Waiting for ${run_name} to finish..."
    sleep 60
  done
  log "${run_name} appears finished."
}

run_campaign() {
  local run_name="$1"
  local predictor="$2"
  local post_predictor="$3"

  mkdir -p "${OUT_ROOT}/${run_name}"
  log "Starting ${run_name} (predictor=${predictor}, post=${post_predictor})"
  caffeinate -dims "${RUN_SCRIPT}" \
    --template-yaml "${TEMPLATE_YAML}" \
    --run-name "${run_name}" \
    --predictor "${predictor}" \
    --post-predictor "${post_predictor}" \
    --num-runs 50 \
    --num-opt-cycles 5 \
    --binder-min-len 65 \
    --binder-max-len 95 \
    --max-parallel 4 \
    > "${OUT_ROOT}/${run_name}/launch.log" 2>&1
  log "Finished ${run_name}"
}

mkdir -p "${OUT_ROOT}"
log "Queue worker started."
log "Will wait for ${WAIT_RUN}, then run ${FIRST_RUN}, then ${SECOND_RUN}."

wait_for_run_to_finish "${WAIT_RUN}"
run_campaign "${FIRST_RUN}" "intellifold" "boltz,openfold-3-mlx"
run_campaign "${SECOND_RUN}" "openfold-3-mlx" "boltz,intellifold"

log "Queue worker completed all queued campaigns."
