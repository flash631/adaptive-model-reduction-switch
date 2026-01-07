#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/03_build_rom_low_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

VENV_ACTIVATE="/home/kalel1938/venvs/myenv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  log "Activated venv: ${VENV_ACTIVATE}"
fi

CONFIG="${ROOT_DIR}/rom/configs/low.yaml"
log "Using config: ${CONFIG}"

log "Building POD + OpInf ROM and writing ROM-only forecast case..."
python "${ROOT_DIR}/rom/python/build_rom_low.py" --config "${CONFIG}" 2>&1 | tee -a "${LOG}"

log "Done. Log saved to: ${LOG}"

