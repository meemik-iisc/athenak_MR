#!/usr/bin/env bash
#
# load_module.sh  —  set up the AthenaK build/run environment on PARAM Rudra
#
# Must be SOURCED (not executed) so the modules stay loaded in your shell:
#   source ./load_module.sh
# Then:
#   ./make_build_file.sh <pgen_name>
#   cmake --build build_<pgen_name> -j${SLURM_CPUS_PER_TASK:-4}
#

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Error: this script must be sourced:  source $0"
    exit 1
fi

# make the 'module' command available in non-login shells
if ! type module >/dev/null 2>&1 && [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module purge
module load nvhpc-hpcx-cuda13/25.11     # CUDA 13.0 + CUDA-aware HPC-X MPI

export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++
export OMP_NUM_THREADS=1
export UCX_LOG_LEVEL=fatal

if command -v nvcc >/dev/null 2>&1 && command -v mpicc >/dev/null 2>&1; then
    echo "Environment loaded:"
    echo "  nvcc : $(command -v nvcc) ($(nvcc --version | grep -o 'release [0-9.]*'))"
    echo "  mpicc: $(command -v mpicc)"
else
    echo "Error: nvcc or mpicc not found after loading nvhpc-hpcx-cuda13/25.11"
    return 1
fi

case "$(hostname)" in
    login*) echo "Note: you are on a login node; compile on a GPU node:"
            echo "  srun --pty --time=01:00:00 --partition=gpu-debug --cpus-per-task=4 --gres=gpu:1 bash" ;;
esac
