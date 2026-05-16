"""CuTe DSL kernels for the SmolLM-135M one-token engine.

The model runner treats these as optional: if CUTLASS/CuTe is not installed, it
falls back to equivalent JAX ops.  The kernels are deliberately tiny and shape
specialized to the hot decode path (batch=1, hidden=576, ffn=1536).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.jax as cjax
    import cuda.bindings.driver as cuda
except Exception:  # pragma: no cover - optional dependency on GPU host only
    cutlass = None
    cute = None
    cjax = None
    cuda = None


HAS_CUTE = cute is not None


if HAS_CUTE:
    @cute.kernel
    def silu_mul_kernel(x: cute.Tensor, out: cute.Tensor, n: int):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        idx = bidx * bdim + tidx
        if idx < n:
            u = x[idx]
            v = x[idx + n]
            # sigmoid(u) * u * v, all in the tensor dtype.  This fuses the
            # SwiGLU elementwise part and avoids materializing two split arrays.
            out[idx] = (u / (cutlass.Float32(1.0) + cutlass.exp(-u))) * v


    @cute.jit
    def launch_silu_mul(stream: cuda.CUstream, x: cute.Tensor, out: cute.Tensor, *, n: int):
        block = 256
        grid = (n + block - 1) // block
        silu_mul_kernel(x, out, n).launch(grid=[grid, 1, 1], block=[block, 1, 1], stream=stream)


def silu_mul(x: jax.Array) -> jax.Array:
    """Return silu(x[:1536]) * x[1536:] for a 3072-vector."""
    if not HAS_CUTE:
        u, v = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(u) * v
    n = x.shape[-1] // 2
    flat = x.reshape(-1)
    call = cjax.cutlass_call(
        launch_silu_mul,
        output_shape_dtype=jax.ShapeDtypeStruct((n,), x.dtype),
        n=n,
    )
    return call(flat)
