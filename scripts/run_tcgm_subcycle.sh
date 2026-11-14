#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="tcgm_subcycle"
INPUT_FILENAME="tcgm_subcycle/tcgm_subcycle.athinput"
OUTPUT_FILENAME="tcgm_subcycle"
RST_FILENAME="wind_outflow_rad_bondi.00001.rst"

# Repo root defaults to the parent of this script's directory (scripts/); override with
# ATHENAK_ROOT=... . Executable, outputs and logs live in the build directory BUILD_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATHENAK_ROOT="${ATHENAK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-${ATHENAK_ROOT}/build_tcgm_subcycle/src}"

INPUT_PATH="${ATHENAK_ROOT}/inputs/hydro/${INPUT_FILENAME}"
OUTPUT_PATH="${BUILD_DIR}/${OUTPUT_FILENAME}"
RESTART_PATH="${OUTPUT_PATH}/rst/${RST_FILENAME}"
EXE_PATH="${BUILD_DIR}/athena"
# Cycles between diagnostic lines in the .out log; each line reports dt and cool_nsub
# (number of cooling sub-cycles in that step). Overrides <time>/ndiag of the input file.
NDIAG=100

echo "ATHENAK_DIR       = ${ATHENAK_ROOT}"
echo "BUILD_DIR         = ${BUILD_DIR}"
echo "EXECUTABLE_PATH   = ${EXE_PATH}"
echo "INPUT_PATH        = ${INPUT_PATH}"
echo "OUTPUT_PATH       = ${OUTPUT_PATH}"
echo "NDIAG             = ${NDIAG}"



# Sanity checks
if [ ! -f "${EXE_PATH}" ]; then
    echo "Error: athena binary not found at ${EXE_PATH}"
    exit 1
fi
if [ ! -f "${INPUT_PATH}" ]; then
    echo "Error: input file not found at ${INPUT_PATH}"
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"
cd "${BUILD_DIR}"

# Pin to GPU 0 explicitly
export CUDA_VISIBLE_DEVICES=0

# Fresh run
nohup ${EXE_PATH} \
    -i "${INPUT_PATH}" \
    -d "${OUTPUT_PATH}" \
    time/ndiag=${NDIAG} \
    > "${BUILD_DIR}/${JOB_NAME}.out" \
    2> "${BUILD_DIR}/${JOB_NAME}.err" \
    < /dev/null &

echo "Started fresh run with PID $!"
echo "Monitor GPU:  nvidia-smi dmon -s u -d 5"
echo "Follow log:   tail -f ${BUILD_DIR}/${JOB_NAME}.out"
echo "Follow err:   tail -f ${BUILD_DIR}/${JOB_NAME}.err"
echo "Sub-cycles:   grep -o 'cycle=[0-9]* .*cool_nsub=[0-9]*' ${BUILD_DIR}/${JOB_NAME}.out"

# --- Restart run (uncomment when needed) ---
# nohup ${EXE_PATH} \
#     -r "${RESTART_PATH}" \
#     -d "${OUTPUT_PATH}" \
#     time/ndiag=${NDIAG} \
#     > "${BUILD_DIR}/${JOB_NAME}_restart.out" \
#     2> "${BUILD_DIR}/${JOB_NAME}_restart.err" \
#     < /dev/null &
# echo "Started restart run with PID $!"
# echo "Monitor GPU:  nvidia-smi dmon -s u -d 5"
# echo "Follow log:   tail -f ${BUILD_DIR}/${JOB_NAME}_restart.out"
# echo "Follow err:   tail -f ${BUILD_DIR}/${JOB_NAME}_restart.err"