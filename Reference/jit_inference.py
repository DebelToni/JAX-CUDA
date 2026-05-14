import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")

import jax
import jax.numpy as jnp
from GiantGPT import GiantGPT


def init_inference_state(model: GiantGPT, key_params, key_dropout,
                         batch_size: int, *, pad_token_id: int = 0):
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy, deterministic=True, cur_index=0,
    )
    return variables["params"], {k: v for k, v in variables.items() if k != "params"}


def _apply_with_cache(model, params, nonparam, tokens_1, cur_idx):
    variables = {"params": params, **nonparam}
    logits, new_vars = model.apply(
        variables, tokens_1, deterministic=True,
        cur_index=cur_idx, mutable=["cache"],
    )
    return logits, {**nonparam, "cache": new_vars["cache"]}


def make_prefill_and_decode_fns(model: GiantGPT):
    @jax.jit
    def prefill(params, nonparam, prompt_tokens):
        B, Lp = prompt_tokens.shape
        t0 = jnp.array(0, jnp.int32)

        def step(carry, tok_t_2d):
            np_, t = carry
            _, np_ = _apply_with_cache(model, params, np_, tok_t_2d, t)
            return (np_, t + 1), None

        if Lp > 0:
            xs = jnp.expand_dims(jnp.swapaxes(prompt_tokens, 0, 1), -1)
            (nonparam, t), _ = jax.lax.scan(step, init=(nonparam, t0), xs=xs)
            last_tok = prompt_tokens[:, -1:]
        else:
            nonparam, t = nonparam, t0
            last_tok = jnp.zeros((B, 1), dtype=jnp.int32)

        last_pos = jnp.maximum(t - 1, jnp.array(0, jnp.int32))
        return nonparam, last_pos, last_tok

    @partial(jax.jit, static_argnames=("steps",))
    def decode(params, nonparam, last_tok, t, *, steps: int):
        B = last_tok.shape[0]
        out = jnp.zeros((B, steps), dtype=jnp.int32)

        def body(carry, i):
            np_, t_, tok_prev, out_ = carry
            logits, np_ = _apply_with_cache(model, params, np_, tok_prev, t_)
            next_tok = jnp.argmax(logits[:, -1, :], axis=-1)
            out_ = jax.lax.dynamic_update_slice(out_, next_tok[:, None], (0, i))
            return (np_, t_ + 1, next_tok[:, None], out_), None

        (nonparam, _, _, out), _ = jax.lax.scan(
            body, init=(nonparam, t, last_tok, out),
            xs=jnp.arange(steps, dtype=jnp.int32),
        )
        return out, nonparam

    return prefill, decode
