#!/usr/bin/env bash
#
# make_build_file.sh  —  configure an AthenaK build on PARAM Rudra
#                        (A100 80GB / CUDA 13.0 / HPC-X MPI)
#
# Usage: ./make_build_file.sh <pgen_name> [mpi|nompi]
#
# Run this on a GPU node, not the login node:
#   srun --pty --time=01:00:00 --partition=gpu-debug --cpus-per-task=4 --gres=gpu:1 bash
#   source ./load_module.sh
#   ./make_build_file.sh <pgen_name>
#

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pgen_name> [mpi|nompi]"
    echo "Example: $0 blast_wave"
    exit 1
fi

PGEN_NAME="$1"
MPI_MODE="${2:-mpi}"
BUILD_DIR_NAME="build_${PGEN_NAME}"

case "$MPI_MODE" in
    mpi)   ENABLE_MPI=ON ;;
    nompi) ENABLE_MPI=OFF ;;
    *) echo "Error: second argument must be 'mpi' or 'nompi'"; exit 1 ;;
esac

ATHENAK_ROOT="${ATHENAK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -d "$ATHENAK_ROOT" ]; then
    echo "Error: ATHENAK_ROOT=$ATHENAK_ROOT does not exist."
    exit 1
fi
cd "$ATHENAK_ROOT"

# ---- environment: CUDA 13.0 + CUDA-aware HPC-X MPI -------------------------
if ! command -v nvcc >/dev/null 2>&1 || ! command -v mpicc >/dev/null 2>&1; then
    echo "Error: nvcc/mpicc not found. Load the environment first:"
    echo "  source $ATHENAK_ROOT/load_module.sh"
    exit 1
fi

case "$(hostname)" in
    login*) echo "Warning: you are on a login node; compile on a GPU node (srun ... --partition=gpu-debug)." ;;
esac

# ---- Kokkos checks (CUDA 13 patches) ---------------------------------------
NVCC_WRAPPER="$ATHENAK_ROOT/kokkos/bin/nvcc_wrapper"
if [ ! -f "$NVCC_WRAPPER" ]; then
    echo "Error: nvcc_wrapper not found at $NVCC_WRAPPER"
    echo "Run: git submodule update --init"
    exit 1
fi
if grep -q 'default_arch="sm_70"' "$NVCC_WRAPPER"; then
    echo "Error: nvcc_wrapper still defaults to sm_70 (unsupported by CUDA 13)."
    echo "Fix:   sed -i 's/default_arch=\"sm_70\"/default_arch=\"sm_80\"/' $NVCC_WRAPPER"
    exit 1
fi
KK_INST="$ATHENAK_ROOT/kokkos/core/src/Cuda/Kokkos_Cuda_Instance.hpp"
if [ "$(grep -c 'cudaMemLocation location' "$KK_INST" || true)" -lt 2 ]; then
    echo "Warning: Kokkos CUDA-13 API patch (cudaMemLocation) not applied to"
    echo "         $KK_INST — compilation will likely fail. See the Rudra setup guide, step 5c."
fi

export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++

# ---- system-header workaround (some GPU nodes lack /usr/include headers) ----
LOCAL_INC="$HOME/local_include"
EXTRA_INC=""
EXTRA_CMAKE=()
if [ ! -f /usr/include/features.h ] || [ ! -f /usr/include/c++/11/type_traits ]; then
    if [ ! -d "$LOCAL_INC" ]; then
        echo "Error: system headers missing on $(hostname) and $LOCAL_INC does not exist."
        echo "On the login node run: mkdir -p $LOCAL_INC && cp -r /usr/include/. $LOCAL_INC/"
        exit 1
    fi
    echo "System headers missing on this node; using $LOCAL_INC"
    EXTRA_INC="-isystem $LOCAL_INC/c++/11 -isystem $LOCAL_INC/c++/11/x86_64-redhat-linux -idirafter $LOCAL_INC"
    EXTRA_CMAKE=(
        -DCMAKE_INCLUDE_PATH="$LOCAL_INC"
        -DLIBDL_INCLUDE_DIRS="$LOCAL_INC"
    )
fi

# ---- configure --------------------------------------------------------------
rm -rf "$BUILD_DIR_NAME"

if [ "$ENABLE_MPI" = ON ]; then C_COMPILER=mpicc; else C_COMPILER=gcc; fi

cmake -S . -B "$BUILD_DIR_NAME" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
    -DCMAKE_C_FLAGS="$EXTRA_INC" \
    -DCMAKE_CXX_FLAGS="$EXTRA_INC" \
    "${EXTRA_CMAKE[@]}" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DAthena_ENABLE_MPI="$ENABLE_MPI" \
    -DPROBLEM="$PGEN_NAME"

echo ""
echo "CMake configuration complete (MPI=$ENABLE_MPI)."
echo "To build run (in the same shell where load_module.sh was sourced):"
echo "  cmake --build $ATHENAK_ROOT/$BUILD_DIR_NAME -j\${SLURM_CPUS_PER_TASK:-4}"
echo "Binary will be in: $ATHENAK_ROOT/$BUILD_DIR_NAME/src/athena"
