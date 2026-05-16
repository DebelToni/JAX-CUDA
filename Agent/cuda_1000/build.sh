#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-sm_120}
"$NVCC" -O3 -std=c++17 --use_fast_math -maxrregcount=64 -arch=${ARCH} smollm_1000.cu -lcublas -o smollm_1000
