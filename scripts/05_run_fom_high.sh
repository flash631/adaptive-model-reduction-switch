#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/05_run_fom_high_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

CONFIG="${ROOT_DIR}/rom/configs/high.yaml"

log "Using config: ${CONFIG}"

if ! command -v blockMesh >/dev/null 2>&1; then
  log "ERROR: blockMesh not found. Source OpenFOAM-13 bashrc first."
  exit 1
fi

VENV_ACTIVATE="/home/kalel1938/venvs/myenv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  log "Activated venv: ${VENV_ACTIVATE}"
fi

BASE_DIR="${ROOT_DIR}/cases/base"
HIGH_DIR="${ROOT_DIR}/cases/high"
RESULTS_DIR="${ROOT_DIR}/results/high"
mkdir -p "${RESULTS_DIR}"

log "Syncing cases/base -> cases/high ..."
rm -rf "${HIGH_DIR}"
mkdir -p "${HIGH_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${BASE_DIR}/" "${HIGH_DIR}/"
else
  cp -a "${BASE_DIR}/." "${HIGH_DIR}/"
fi

log "Applying high-fidelity settings (mesh/time/writeFormat)..."
python "${ROOT_DIR}/rom/python/case_setup_low.py" --config "${CONFIG}" | tee -a "${LOG}"

clean_case() {
  local casedir="$1"
  log "Cleaning case: ${casedir}"
  rm -rf "${casedir}/VTK" "${casedir}/postProcessing" "${casedir}/processor"* 2>/dev/null || true
  find "${casedir}" -maxdepth 1 -type d -regex '.*/[0-9].*' ! -name "0" -exec rm -rf {} + 2>/dev/null || true
}

clean_case "${HIGH_DIR}"

log "Running blockMesh..."
(cd "${HIGH_DIR}" && blockMesh) 2>&1 | tee -a "${LOG}"

log "Post-mesh case setup (functionObjects: forces if cylinder patch detected)..."
python "${ROOT_DIR}/rom/python/case_setup_low.py" --config "${CONFIG}" --post-mesh | tee -a "${LOG}"

control_dict="${HIGH_DIR}/system/controlDict"
solver_name="$(grep -E '^[[:space:]]*solver[[:space:]]+' "${control_dict}" 2>/dev/null | head -n 1 | sed -E 's/^[[:space:]]*solver[[:space:]]+([^;[:space:]]+).*/\1/' || true)"
app_name="$(grep -E '^[[:space:]]*application[[:space:]]+' "${control_dict}" 2>/dev/null | head -n 1 | sed -E 's/^[[:space:]]*application[[:space:]]+([^;[:space:]]+).*/\1/' || true)"

run_cmd=()
if [[ -n "${solver_name}" ]]; then
  run_cmd=(foamRun -solver "${solver_name}")
  log "Runner: foamRun  Solver: ${solver_name}"
elif [[ -n "${app_name}" ]]; then
  run_cmd=("${app_name}")
  log "Application: ${app_name}"
else
  run_cmd=(icoFoam)
  log "No solver/application found in controlDict; falling back to icoFoam"
fi

log "Running solver..."
(cd "${HIGH_DIR}" && "${run_cmd[@]}") 2>&1 | tee -a "${LOG}"

if command -v foamToVTK >/dev/null 2>&1; then
  log "Running foamToVTK (all times via -time '0:') ..."
  set +e
  (cd "${HIGH_DIR}" && foamToVTK -time '0:' -fields '(U p)') 2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "WARNING: foamToVTK failed with exit code ${rc}; continuing (snapshots will use OpenFOAM ASCII parsing)."
  fi
else
  log "foamToVTK not found; skipping VTK export."
fi

log "Exporting snapshots to results/high/snapshots ..."
python "${ROOT_DIR}/rom/python/snapshot_export.py" --config "${CONFIG}" --prefer-vtk 2>&1 | tee -a "${LOG}"

log "Done. Log saved to: ${LOG}"
