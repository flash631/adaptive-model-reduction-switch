#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/00_env_check_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

log "Repo root: ${ROOT_DIR}"
log "PWD: $(pwd)"
log "User: $(id -un)  Host: $(hostname)"

log "Checking OpenFOAM environment..."
if command -v foamVersion >/dev/null 2>&1; then
  if foamVersion_out="$(foamVersion 2>&1)"; then
    log "foamVersion: ${foamVersion_out}"
  else
    log "foamVersion failed:"
    log "${foamVersion_out}"
  fi
else
  log "foamVersion: NOT FOUND (some OpenFOAM packages don't provide it as a command)."
  if [[ -n "${WM_PROJECT_VERSION:-}" ]]; then
    log "WM_PROJECT_VERSION: ${WM_PROJECT_VERSION}"
  fi
  if [[ -n "${WM_PROJECT:-}" ]]; then
    log "WM_PROJECT: ${WM_PROJECT}"
  fi
fi

log "Checking key OpenFOAM executables (command -v)..."
for exe in blockMesh icoFoam pimpleFoam paraFoam foamToVTK; do
  if command -v "${exe}" >/dev/null 2>&1; then
    log "${exe}: $(command -v "${exe}")"
  else
    log "${exe}: NOT FOUND"
  fi
done

log "Checking Python (prefer venv)..."
VENV_ACTIVATE="/home/kalel1938/venvs/myenv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  log "Activated venv: ${VENV_ACTIVATE}"
else
  log "Venv not found at ${VENV_ACTIVATE} (continuing with system python)."
fi

if command -v python >/dev/null 2>&1; then
  log "python: $(command -v python)"
  log "python --version: $(python --version 2>&1)"
else
  log "python: NOT FOUND"
fi

log "Done. Log saved to: ${LOG}"
