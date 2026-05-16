#!/usr/bin/env python
"""Shape-specialized SmolLM-135M greedy decoder.

Optimized for the exact task in autoresearch/smollm135m-cute-tps:
batch=1, greedy decode, 576 hidden, 30 layers, 9Q/3KV heads, head_dim=64,
and <=128 active positions for the required prompt+100 benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Reference"))

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict
from transformers import AutoTokenizer

HAS_CUTE = False
CUTE_ENABLED = False
LOGITS_BF16 = False


def silu_mul(x):
    u, v = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(u) * v


TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
CHECKPOINT_PATH = ROOT / "Reference" / "smollm-135m.npz"

D = 576
LAYERS = 30
QH = 9
KVH = 3
HD = 64
FF = 1536
ACTIVE_CTX = 128
ROPE_CTX = 256
SCALE = jnp.asarray(HD ** -0.5, jnp.bfloat16)


def load_npz(path: Path):
    with np.load(path) as npz:
        flat = {tuple(k.split("/")): v for k, v in npz.items() if not k.startswith("__")}
    return unflatten_dict(flat)


PARAM_DTYPE = jnp.bfloat16


def stack_params(p, dtype=None):
    dtype = dtype or PARAM_DTYPE
    blocks = [p[f"TinyTransformerBlock_{i}"] for i in range(LAYERS)]
    attn = [b["NativeJaxSelfAttention_0"] for b in blocks]
    return {
        "emb": jnp.asarray(p["Embed_0"]["embedding"], dtype),
        "final": jnp.asarray(p["final_norm"]["scale"], dtype),
        "rms1": jnp.stack([jnp.asarray(b["rms1"]["scale"], dtype) for b in blocks]),
        "rms2": jnp.stack([jnp.asarray(b["rms2"]["scale"], dtype) for b in blocks]),
        "qkv": jnp.stack([jnp.asarray(a["qkv_proj"]["kernel"], dtype) for a in attn]),
        "o": jnp.stack([jnp.asarray(a["o_proj"]["kernel"], dtype) for a in attn]),
        "fc1": jnp.stack([jnp.asarray(b["fc1"]["kernel"], dtype) for b in blocks]),
        "fc2": jnp.stack([jnp.asarray(b["fc2"]["kernel"], dtype) for b in blocks]),
    }


def rope_tables():
    inv = 1.0 / (10000 ** (jnp.arange(0, HD, 2, dtype=jnp.float32) / HD))
    ang = jnp.arange(ROPE_CTX, dtype=jnp.float32)[:, None] * inv[None, :]
    emb = jnp.concatenate([ang, ang], axis=-1)
    return jnp.sin(emb).astype(PARAM_DTYPE), jnp.cos(emb).astype(PARAM_DTYPE)


def rms(x, w):
    xf = x.astype(jnp.float32)
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf) + 1e-5)
    return (y * w.astype(jnp.float32)).astype(PARAM_DTYPE)


def apply_rope(x, sin_t, cos_t):
    x1, x2 = jnp.split(x, 2, axis=-1)
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos_t[None, :] + rot * sin_t[None, :]


def layer_apply(i, x, kcache, vcache, pos, sin, cos, params):
    rms1 = params["rms1"][i]
    rms2 = params["rms2"][i]
    wqkv = params["qkv"][i]
    wo = params["o"][i]
    wfc1 = params["fc1"][i]
    wfc2 = params["fc2"][i]

    h = rms(x, rms1)
    qkv = h @ wqkv
    q, k, v = jnp.split(qkv, [D, D + KVH * HD])
    q = apply_rope(q.reshape(QH, HD), sin[pos], cos[pos])
    k = apply_rope(k.reshape(KVH, HD), sin[pos], cos[pos])
    v = v.reshape(KVH, HD)

    slot = pos % ACTIVE_CTX
    kc = jax.lax.dynamic_update_slice(kcache, k[None, :, None, :], (i, 0, slot, 0))
    vc = jax.lax.dynamic_update_slice(vcache, v[None, :, None, :], (i, 0, slot, 0))
    kk = kc[i]
    vv = vc[i]

    qg = q.reshape(KVH, QH // KVH, HD)
    scores = jnp.einsum("kgh,kth->kgt", qg, kk) * SCALE
    mask = jnp.arange(ACTIVE_CTX, dtype=jnp.int32) <= jnp.minimum(pos, ACTIVE_CTX - 1)
    scores = jnp.where(mask[None, None, :], scores, -1.0e10)
    probs = jax.nn.softmax(scores, axis=-1).astype(PARAM_DTYPE)
    y = jnp.einsum("kgt,kth->kgh", probs, vv).reshape(D)
    x = x + (y @ wo)

    h2 = rms(x, rms2)
    ff = silu_mul(h2 @ wfc1)
    x = x + (ff @ wfc2)
    return x, kc, vc


def one_token(params, token, kcache, vcache, pos, sin, cos):
    x = params["emb"][token]
    # Python-loop unrolling gives XLA static layer indices, eliminating dynamic
    # parameter/cache gathers in the decode hot path.
    for i in range(LAYERS):
        x, kcache, vcache = layer_apply(i, x, kcache, vcache, pos, sin, cos, params)
    x = rms(x, params["final"])
    if LOGITS_BF16:
        logits = x @ params["emb"].T
    else:
        logits = x.astype(jnp.float32) @ params["emb"].astype(jnp.float32).T
    return jnp.argmax(logits).astype(jnp.int32), kcache, vcache


@jax.jit
def prefill(params, prompt, sin, cos):
    kcache = jnp.zeros((LAYERS, KVH, ACTIVE_CTX, HD), PARAM_DTYPE)
    vcache = jnp.zeros((LAYERS, KVH, ACTIVE_CTX, HD), PARAM_DTYPE)

    def body(carry, item):
        k, v = carry
        pos, tok = item
        _, k, v = one_token(params, tok, k, v, pos, sin, cos)
        return (k, v), None

    pos = jnp.arange(prompt.shape[0], dtype=jnp.int32)
    (kcache, vcache), _ = jax.lax.scan(body, (kcache, vcache), (pos, prompt))
    return kcache, vcache, prompt[-1], jnp.asarray(prompt.shape[0] - 1, jnp.int32)


@jax.jit(static_argnames=("steps",), donate_argnames=("kcache", "vcache"))
def decode(params, kcache, vcache, last_tok, pos, sin, cos, *, steps: int):
    def body(carry, i):
        k, v, tok, p = carry
        ntok, k, v = one_token(params, tok, k, v, p, sin, cos)
        return (k, v, ntok, p + 1), ntok

    (_, _, _, _), toks = jax.lax.scan(body, (kcache, vcache, last_tok, pos), jnp.arange(steps, dtype=jnp.int32))
    return toks


@jax.jit(static_argnames=("steps",), donate_argnames=("kcache", "vcache"))
def decode_unrolled(params, kcache, vcache, last_tok, pos, sin, cos, *, steps: int):
    toks = []
    tok = last_tok
    p = pos
    for _ in range(steps):
        tok, kcache, vcache = one_token(params, tok, kcache, vcache, p, sin, cos)
        toks.append(tok)
        p = p + 1
    return jnp.stack(toks)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--use-cute", action="store_true", help="use optional CuTe SwiGLU kernel (experimental)")
    ap.add_argument("--unroll-decode", action="store_true", help="fully unroll the 100-token decode loop")
    ap.add_argument("--bf16-logits", action="store_true", help="compute final vocab projection in BF16 (may change greedy path)")
    ap.add_argument("--fp16", action="store_true", help="run weights/cache/activations in FP16 instead of BF16")
    ap.add_argument("--ctx", type=int, default=128, help="KV cache/window length; <128 is faster but approximate")
    return ap.parse_args()


def main():
    args = parse_args()
    global HAS_CUTE, CUTE_ENABLED, LOGITS_BF16, PARAM_DTYPE, SCALE, ACTIVE_CTX, silu_mul
    LOGITS_BF16 = bool(args.bf16_logits)
    ACTIVE_CTX = int(args.ctx)
    if args.fp16:
        PARAM_DTYPE = jnp.float16
        SCALE = jnp.asarray(HD ** -0.5, PARAM_DTYPE)
    if args.use_cute:
        try:
            from Agent.cute_kernels import HAS_CUTE as _HAS_CUTE, silu_mul as _cute_silu_mul
            HAS_CUTE = bool(_HAS_CUTE)
            if HAS_CUTE:
                silu_mul = _cute_silu_mul
                CUTE_ENABLED = True
        except Exception as e:
            print(f"[warn] CuTe unavailable, using JAX SwiGLU: {type(e).__name__}: {e}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<pad>"
    params = stack_params(load_npz(CHECKPOINT_PATH), PARAM_DTYPE)
    params = jax.device_put(params)
    sin, cos = jax.device_put(rope_tables())
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if ids and tok.eos_token_id is not None and ids[-1] == tok.eos_token_id:
        ids = ids[:-1]
    ids = np.asarray(ids[-ROPE_CTX:], dtype=np.int32)
    if len(ids) + args.steps > ROPE_CTX:
        raise ValueError(f"specialized ROPE_CTX={ROPE_CTX}, got prompt+steps={len(ids)+args.steps}")
    prompt = jnp.asarray(ids, dtype=jnp.int32)

    k, v, last, pos = prefill(params, prompt, sin, cos)
    jax.tree_util.tree_map(lambda z: z.block_until_ready() if isinstance(z, jax.Array) else z, (k, v, last, pos))
    warm_k = jnp.array(k, copy=True)
    warm_v = jnp.array(v, copy=True)
    decode_fn = decode_unrolled if args.unroll_decode else decode
    warm = decode_fn(params, warm_k, warm_v, last, pos, sin, cos, steps=args.steps)
    warm.block_until_ready()
    timed_k = jnp.array(k, copy=True)
    timed_v = jnp.array(v, copy=True)
    jax.tree_util.tree_map(lambda z: z.block_until_ready() if isinstance(z, jax.Array) else z, (timed_k, timed_v))
    t0 = time.perf_counter()
    toks = decode_fn(params, timed_k, timed_v, last, pos, sin, cos, steps=args.steps)
    toks.block_until_ready()
    dt = time.perf_counter() - t0
    out = np.concatenate([ids, np.asarray(toks)])
    print("\n" + "=" * 50)
    print(tok.decode(out, skip_special_tokens=True))
    print("=" * 50)
    if args.verbose:
        print(f"\n[perf] prompt={len(ids)} gen={args.steps} decode_only={dt:.4f}s tok/s={args.steps / dt:.2f} cute={CUTE_ENABLED}")


if __name__ == "__main__":
    main()
