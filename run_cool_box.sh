#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="cbox"
INPUT_FILENAME="cooling_box/cbox.athinput"
OUTPUT_FILENAME="cbox_02Myr"
RST_FILENAME="cbox.00001.rst"

ATHENAK_ROOT="/home/meemik/Meemik/athenak_cool"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${ATHENAK_ROOT}/inputs/hydro/${INPUT_FILENAME}"
OUTPUT_PATH="${SCRIPT_DIR}/${OUTPUT_FILENAME}"
RESTART_PATH="${OUTPUT_PATH}/rst/${RST_FILENAME}"
EXE_PATH="${SCRIPT_DIR}/athena"

echo "ATHENAK_DIR       = ${ATHENAK_ROOT}"
echo "SCRRIPT_DIR       = ${SCRIPT_DIR}"
echo "EXECUTABLE_PATH   = ${EXE_PATH}"
echo "INPUT_PATH        = ${INPUT_PATH}"
echo "OUTPUT_PATH       = ${OUTPUT_PATH}"



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
cd "${SCRIPT_DIR}"

# Pin to GPU 0 explicitly
export CUDA_VISIBLE_DEVICES=0

# Fresh run
nohup ${EXE_PATH} \
    -i "${INPUT_PATH}" \
    -d "${OUTPUT_PATH}" \
    > "${SCRIPT_DIR}/${JOB_NAME}.out" \
    2> "${SCRIPT_DIR}/${JOB_NAME}.err" \
    < /dev/null &

echo "Started fresh run with PID $!"
echo "Monitor GPU:  nvidia-smi dmon -s u -d 5"
echo "Follow log:   tail -f ${SCRIPT_DIR}/${JOB_NAME}.out"
echo "Follow err:   tail -f ${SCRIPT_DIR}/${JOB_NAME}.err"

# --- Restart run (uncomment when needed) ---
# nohup ${EXE_PATH} \
#     -r "${RESTART_PATH}" \
#     -d "${OUTPUT_PATH}" \
#     > "${SCRIPT_DIR}/${JOB_NAME}_restart.out" \
#     2> "${SCRIPT_DIR}/${JOB_NAME}_restart.err" \
#     < /dev/null &
# echo "Started restart run with PID $!"
# echo "Monitor GPU:  nvidia-smi dmon -s u -d 5"
# echo "Follow log:   tail -f ${SCRIPT_DIR}/${JOB_NAME}_restart.out"
# echo "Follow err:   tail -f ${SCRIPT_DIR}/${JOB_NAME}_restart.err"