#!/usr/bin/env bash
#
# make_build_file.sh
# Usage: ./make_build_file.sh <pgen_name>
#

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pgen_name>"
    echo "Example: $0 blast_wave"
    exit 1
fi

PGEN_NAME="$1"
BUILD_DIR_NAME="build_${PGEN_NAME}"

ATHENAK_ROOT="${ATHENAK_ROOT:-$HOME/Meemik/athenak_MR}"
if [ ! -d "$ATHENAK_ROOT" ]; then
    echo "Error: ATHENAK_ROOT=$ATHENAK_ROOT does not exist."
    exit 1
fi
cd "$ATHENAK_ROOT"

: "${CUDA_HOME:=/usr/local/cuda-12.6}"

NVCC_WRAPPER="$ATHENAK_ROOT/kokkos/bin/nvcc_wrapper"
if [ ! -f "$NVCC_WRAPPER" ]; then
    echo "Error: nvcc_wrapper not found at $NVCC_WRAPPER"
    echo "Run: git submodule update --init"
    exit 1
fi

export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export CUDA_HOME="$CUDA_HOME"

rm -rf "$BUILD_DIR_NAME"
mkdir "$BUILD_DIR_NAME"

cmake -B "$BUILD_DIR_NAME" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DAthena_ENABLE_MPI=OFF \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DKokkos_ARCH_ADA89=ON \
    -DHDF5_ROOT=/usr \
    -DHDF5_INCLUDE_DIR=/usr/include/hdf5/serial \
    -DHDF5_LIBRARY=/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    -DPROBLEM="$PGEN_NAME"

echo ""
echo "CMake configuration complete."
echo "To build run:"
echo "  cmake --build $ATHENAK_ROOT/$BUILD_DIR_NAME -j\$(nproc)"
echo "Binary will be in: $ATHENAK_ROOT/$BUILD_DIR_NAME/src"