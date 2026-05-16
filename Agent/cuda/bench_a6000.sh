#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/opt/venv/bin/python}
PROMPT='Technical note: In transformer language models, rotary position embeddings work by rotating query and key vectors in pairs. This helps attention layers represent relative positions because'

echo '== GPU =='
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo '== Reference/JAX baseline =='
(cd Reference && "$PY" run_inference.py --prompt "$PROMPT" --steps 100 --verbose)

echo '== Export CUDA binary checkpoint =='
"$PY" Agent/cuda/export_smollm_binary.py

echo '== Build CUDA runner =='
bash Agent/cuda/build.sh

echo '== C++/CUDA runner (normal device-token chain) =='
./Agent/cuda/smollm_cuda Agent/cuda/smollm135m_f32.bin

echo '== C++/CUDA runner (CUDA Graph captured 100-token decode) =='
./Agent/cuda/smollm_cuda Agent/cuda/smollm135m_f32.bin --graph

echo '== C++/CUDA FP16 custom-GEMV runner (CUDA Graph captured 100-token decode) =='
./Agent/cuda/smollm_cuda_half Agent/cuda/smollm135m_f32.bin --graph
