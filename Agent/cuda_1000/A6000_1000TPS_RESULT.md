# A6000 1000+ TPS result

Target GPU: NVIDIA RTX A6000 only.

GPU:

```text
NVIDIA RTX A6000, 550.127.08
```

Final optimized command:

```bash
ARCH=sm_86 ./Agent/cuda_1000/bench.sh
```

Build uses:

```bash
nvcc -O3 -std=c++17 --use_fast_math -maxrregcount=64 -arch=sm_86 smollm_1000.cu -lcublas -o smollm_1000
```

Reached 1000 TPS on A6000:

```text
[cuda-half-perf] prompt=29 gen=100 decode_only=0.0998717s tok/s=1001.28 graph=1
```

Confirmation rerun:

```text
[cuda-half-perf] prompt=29 gen=100 decode_only=0.099754s tok/s=1002.47 graph=1
```

Output tokens matched the previous valid CUDA path for the 100-token benchmark.

Last improvements that crossed 1000:

- fixed the 64-thread attention variant so it fills all 128 score slots correctly
- used `exp2f((x) * log2(e))` in attention softmax
- increased logits partial argmax to 32 warps/block with atomic packed max + unpack
- compiled with `-maxrregcount=64`
