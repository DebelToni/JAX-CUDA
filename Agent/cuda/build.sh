#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
"$NVCC" -O3 -std=c++17 --use_fast_math -arch=sm_86 smollm_cuda.cu -lcublas -o smollm_cuda
"$NVCC" -O3 -std=c++17 --use_fast_math -arch=sm_86 smollm_cuda_half.cu -lcublas -o smollm_cuda_half
