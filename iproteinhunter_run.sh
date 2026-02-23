#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible shim. The canonical runner is now iproteinhunter_run.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/iproteinhunter_run.sh" "$@"
