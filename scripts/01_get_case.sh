#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/01_get_case_${ts}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

TUTORIALS_ROOT_DEFAULT="/home/kalel1938/OpenFOAM/kalel1938-13/run/tutorials"
if [[ -n "${FOAM_TUTORIALS:-}" ]] && [[ -d "${FOAM_TUTORIALS}" ]]; then
  TUTORIALS_ROOT_DEFAULT="${FOAM_TUTORIALS}"
fi
TUTORIALS_ROOT="${TUTORIALS_ROOT:-${TUTORIALS_ROOT_DEFAULT}}"

BASE_CASE_DIR="${ROOT_DIR}/cases/base"
MARKER_FILE="${BASE_CASE_DIR}/.case_source"

log "Tutorials root: ${TUTORIALS_ROOT}"
if [[ ! -d "${TUTORIALS_ROOT}" ]]; then
  log "ERROR: Tutorials root not found. Set TUTORIALS_ROOT or check your OpenFOAM install."
  exit 1
fi

extract_application() {
  local controlDict="$1"
  # Matches: application  icoFoam;
  grep -E '^[[:space:]]*application[[:space:]]+' "${controlDict}" 2>/dev/null \
    | head -n 1 \
    | sed -E 's/^[[:space:]]*application[[:space:]]+([^;[:space:]]+).*/\1/'
}

score_case() {
  local d="$1"
  local controlDict="${d}/system/controlDict"
  local app
  app="$(extract_application "${controlDict}")"

  local score=0
  local path_lc
  path_lc="$(echo "${d}" | tr '[:upper:]' '[:lower:]')"

  # Hard excludes: multiRegion / CHT / VoF / etc.
  if [[ "${path_lc}" == *"multiregion"* ]] || [[ -f "${d}/system/regionProperties" ]] || [[ -f "${d}/constant/regionProperties" ]]; then
    echo "-9999"
    return
  fi
  if [[ "${path_lc}" == *"cht"* ]] || [[ "${path_lc}" == *"vof"* ]] || [[ "${path_lc}" == *"multiphase"* ]]; then
    echo "-9999"
    return
  fi

  # Prefer incompressible, single-region, small/quick cases.
  [[ "${path_lc}" == *"/incompressible/"* ]] && score=$((score + 200))
  [[ -f "${d}/system/blockMeshDict" ]] && score=$((score + 50))
  [[ -f "${d}/system/snappyHexMeshDict" ]] && score=$((score - 50))

  case "${app}" in
    icoFoam) score=$((score + 200)) ;;
    pisoFoam) score=$((score + 150)) ;;
    pimpleFoam) score=$((score + 100)) ;;
    potentialFoam) score=$((score - 500)) ;;
    simpleFoam) score=$((score - 500)) ;;
    *) score=$((score - 100)) ;;
  esac

  # Require typical unsteady incompressible fields for our workflow.
  if [[ ! -f "${d}/0/U" ]] || [[ ! -f "${d}/0/p" ]]; then
    # Some tutorials use 0.org; treat as acceptable.
    if [[ ! -f "${d}/0.org/U" ]] || [[ ! -f "${d}/0.org/p" ]]; then
      echo "-9999"
      return
    fi
  fi

  [[ "${path_lc}" == *"flowaroundcylinder"* ]] && score=$((score + 80))
  [[ "${path_lc}" == *"cylinder2d"* ]] && score=$((score + 60))
  [[ "${path_lc}" == *"cylinder"* ]] && score=$((score + 20))

  echo "${score}"
}

pick_best_case() {
  local mode="$1" # cylinder|cavity
  local best_score="-100000"
  local best_dir=""

  # Find candidate tutorial case folders by name, then score.
  local find_expr=()
  case "${mode}" in
    cylinder)
      find_expr=( \( -iname "*flowaroundcylinder*" -o -iname "*cylinder2d*" -o -iname "*cylinder*" \) )
      ;;
    cavity)
      find_expr=( -iname "*cavity*" )
      ;;
    *)
      find_expr=()
      ;;
  esac

  while IFS= read -r d; do
    [[ -f "${d}/system/controlDict" ]] || continue
    local sc
    sc="$(score_case "${d}")"
    if [[ "${sc}" -gt "${best_score}" ]]; then
      best_score="${sc}"
      best_dir="${d}"
    fi
  done < <(find "${TUTORIALS_ROOT}" -type d "${find_expr[@]}" 2>/dev/null || true)

  echo "${best_dir}"
}

select_case_dir=""

log "Searching for a simple cylinder tutorial case (single-region, incompressible, blockMesh-preferred)..."
select_case_dir="$(pick_best_case cylinder || true)"

if [[ -z "${select_case_dir}" ]]; then
  log "No suitable cylinder case found; falling back to cavity..."
  select_case_dir="$(pick_best_case cavity || true)"
fi

if [[ -z "${select_case_dir}" ]]; then
  log "ERROR: Could not find a cylinder or cavity tutorial case containing system/controlDict."
  exit 1
fi

log "Selected tutorial case: ${select_case_dir}"

if [[ -d "${BASE_CASE_DIR}" ]] && [[ -n "$(ls -A "${BASE_CASE_DIR}" 2>/dev/null || true)" ]]; then
  if [[ -f "${MARKER_FILE}" ]] && grep -Fxq "${select_case_dir}" "${MARKER_FILE}"; then
    log "cases/base already populated from this source; nothing to do."
    log "Done. Log saved to: ${LOG}"
    exit 0
  fi

  if [[ "${FORCE}" -ne 1 ]]; then
    log "ERROR: ${BASE_CASE_DIR} is not empty. Re-run with --force to overwrite."
    exit 1
  fi

  log "--force set; overwriting ${BASE_CASE_DIR}"
  rm -rf "${BASE_CASE_DIR}"
fi

mkdir -p "${BASE_CASE_DIR}"
log "Copying tutorial into cases/base..."

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${select_case_dir}/" "${BASE_CASE_DIR}/"
else
  cp -a "${select_case_dir}/." "${BASE_CASE_DIR}/"
fi

echo "${select_case_dir}" > "${MARKER_FILE}"
log "Wrote source marker: ${MARKER_FILE}"
log "Done. Log saved to: ${LOG}"
