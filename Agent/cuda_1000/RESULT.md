# SmolLM CUDA 1000+ TPS run

New experiment folder: `Agent/cuda_1000/`.

Remote GPU used: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, driver `570.195.03`, JAX `0.10.0`.

Same-GPU reference baseline:

```text
Reference/run_inference.py: decode_only=0.1187s tok/s=842.56
```

Best optimized C++/CUDA run:

```text
./Agent/cuda_1000/smollm_1000 Agent/cuda_1000/smollm135m_f32.bin --graph
[cuda-half-perf] prompt=29 gen=100 decode_only=0.0834001s tok/s=1199.04 graph=1
```

Confirmation rerun:

```text
[cuda-half-perf] prompt=29 gen=100 decode_only=0.0875447s tok/s=1142.27 graph=1
```

Main tricks:

- separate new folder; did not continue modifying `Agent/cuda/` for this attempt
- FP16 weights/activations/KV cache
- hardcoded SmolLM-135M shapes and fixed 128-token decode window
- CUDA Graph capture for the whole 100-token decode
- warp-per-output custom GEMV kernels
- fused residual add into GEMV output kernels
- fused RoPE + KV cache write
- half2 attention dot products
