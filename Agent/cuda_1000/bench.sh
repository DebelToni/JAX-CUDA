#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/opt/venv/bin/python Agent/cuda_1000/export_smollm_binary.py
ARCH=${ARCH:-sm_120} bash Agent/cuda_1000/build.sh
./Agent/cuda_1000/smollm_1000 Agent/cuda_1000/smollm135m_f32.bin --graph
