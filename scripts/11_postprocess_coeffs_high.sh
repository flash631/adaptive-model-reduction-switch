#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/11_postprocess_coeffs_high_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

if ! command -v foamPostProcess >/dev/null 2>&1 && ! command -v postProcess >/dev/null 2>&1; then
  log "ERROR: OpenFOAM not sourced (foamPostProcess/postProcess not found)."
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

log "Post-processing force coefficients over reconstructed ROM fields (adaptive case)..."
python "${ROOT_DIR}/rom/python/postprocess_force_coeffs.py" \
  --config "${CONFIG}" \
  --case adaptive \
  --force 2>&1 | tee -a "${LOG}"

log "Done. Log saved to: ${LOG}"
