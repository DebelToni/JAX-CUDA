# A6000-only status

Current valid A6000-only run, not 1000 TPS yet.

GPU:

```text
NVIDIA RTX A6000, driver 550.127.08
```

Best observed valid run on this A6000 attempt:

```text
[cuda-half-perf] prompt=29 gen=100 decode_only=0.110762s tok/s=902.837 graph=1
```

Notes:

- Blackwell result is invalid for A6000 comparison and is not counted.
- Current valid source is `Agent/cuda_1000/smollm_1000.cu`.
- Main current tricks: FP16 weights/cache/activations, CUDA Graph decode, warp GEMV, half2 attention dot, 64-thread attention blocks, fused logits partial argmax.
- Pod left running for more iteration: `rcrr8mvfznsce2` / `gpu-box-1`.
