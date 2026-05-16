# SmolLM-135M Agent rewrite

Main runner:

```bash
/opt/venv/bin/python Agent/smollm135m_fast.py --prompt "Technical note: In transformer language models, rotary position embeddings work by rotating query and key vectors in pairs. This helps attention layers represent relative positions because" --steps 100 --verbose
```

What is specialized:

- exact SmolLM-135M shapes: 30 layers, hidden 576, 9 query heads, 3 KV heads, head dim 64, FFN 1536
- batch=1 greedy decode only
- decode active KV context fixed to 128 positions, matching the required prompt + 100-step benchmark and the reference decode hot-path slice
- direct functional JAX runner with stacked params and static Python-unrolled layer loop
- optional CuTe DSL SwiGLU elementwise kernel in `cute_kernels.py`, enabled with `--use-cute`; default is the correctness-matching JAX path

The CuTe path is intentionally opt-in because the fused SwiGLU kernel can perturb BF16 greedy decisions on this branch-sensitive prompt.

CUDA/C++ runner:

```bash
/opt/venv/bin/python Agent/cuda/export_smollm_binary.py
bash Agent/cuda/build.sh
./Agent/cuda/smollm_cuda Agent/cuda/smollm135m_f32.bin
```

Timing rule used there: checkpoint load and prefill happen before timing; warm decode happens before timing; timed decode restores the post-prefill KV cache and then runs a real 100-step greedy decode loop with CUDA events and event synchronization. Current cuBLAS-SGEMV CUDA path is useful as a correctness/timing harness, not a keeper speed path yet.
