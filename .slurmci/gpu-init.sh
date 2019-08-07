#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --gres=gpu:1

set -euo pipefail
set -x #echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot/gpu"
export CUDA_PATH="/lib64"
export OPENBLAS_NUM_THREADS=1

module load cmake/3.10.2 openmpi/4.0.1 cuda/10.0

julia --color=no --project=env/gpu -e 'using Pkg; pkg"instantiate"; pkg"build"; pkg"precompile"'
