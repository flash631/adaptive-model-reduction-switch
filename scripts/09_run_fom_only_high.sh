#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/09_run_fom_only_high_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

if ! command -v blockMesh >/dev/null 2>&1; then
  log "ERROR: OpenFOAM not sourced (blockMesh not found)."
  exit 1
fi

VENV_ACTIVATE="/home/kalel1938/venvs/myenv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  log "Activated venv: ${VENV_ACTIVATE}"
fi

CONFIG="${ROOT_DIR}/rom/configs/high.yaml"
log "Using config: ${CONFIG}"

log "Running FOM-only baseline (high)..."
python "${ROOT_DIR}/rom/python/fom_only_run.py" --config "${CONFIG}" --force 2>&1 | tee -a "${LOG}"

log "Done. Log saved to: ${LOG}"

