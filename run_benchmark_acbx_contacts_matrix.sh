#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/thomasfryer/iProteinHunter"
VAR_DIR="$REPO/examples/aCbx_bind_contacts_benchmark_variants"
LOG_DIR="$1"
MATRIX_CSV="$LOG_DIR/matrix_runs.csv"

mkdir -p "$LOG_DIR"
echo "yaml_path,run_name,status" > "$MATRIX_CSV"

for yaml in "$VAR_DIR"/*.yaml; do
  base="$(basename "$yaml" .yaml)"
  run_name="${base}"
  echo "[$(date '+%F %T')] START $run_name"

  if caffeinate -dims "$REPO/iProteinHunter_run.sh" \
      --predictor boltz \
      --post-mode none \
      --num-runs 12 \
      --max-parallel 6 \
      --binder-min-len 80 \
      --binder-max-len 80 \
      --template-yaml "$yaml" \
      --run-name "$run_name" \
      > "$LOG_DIR/${run_name}.log" 2>&1; then
    echo "$yaml,$run_name,ok" >> "$MATRIX_CSV"
    echo "[$(date '+%F %T')] DONE $run_name"
  else
    echo "$yaml,$run_name,failed" >> "$MATRIX_CSV"
    echo "[$(date '+%F %T')] FAIL $run_name (see $LOG_DIR/${run_name}.log)"
  fi
done

echo "[$(date '+%F %T')] Matrix complete"
