#!/usr/bin/env bash
set -euo pipefail

REPO='/Users/thomasfryer/iProteinHunter'
VAR_DIR="$REPO/examples/aCbx_bind_contacts_benchmark_variants_v2"
LOG_DIR="$1"
MATRIX_CSV="$LOG_DIR/matrix_runs.csv"

mkdir -p "$LOG_DIR"
echo "yaml_path,run_name,status,output_root" > "$MATRIX_CSV"

for yaml in $(ls -1 "$VAR_DIR"/*.yaml | sort); do
  base="$(basename "$yaml" .yaml)"
  run_name="bench_contacts_v2_${base}"
  out_root="$REPO/output/${run_name}"

  echo "[$(date '+%F %T')] START $run_name"
  if "$REPO/iProteinHunter_run.sh" \
      --predictor boltz \
      --post-mode none \
      --num-runs 12 \
      --max-parallel 6 \
      --binder-min-len 80 \
      --binder-max-len 80 \
      --template-yaml "$yaml" \
      --run-name "$run_name" \
      > "$LOG_DIR/${run_name}.log" 2>&1; then
    echo "$yaml,$run_name,ok,$out_root" >> "$MATRIX_CSV"
    echo "[$(date '+%F %T')] DONE $run_name"
  else
    echo "$yaml,$run_name,failed,$out_root" >> "$MATRIX_CSV"
    echo "[$(date '+%F %T')] FAIL $run_name (see $LOG_DIR/${run_name}.log)"
  fi
done

echo "[$(date '+%F %T')] Matrix complete"
