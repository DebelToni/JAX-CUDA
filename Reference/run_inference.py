#!/usr/bin/env python
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from jit_inference import init_inference_state, make_prefill_and_decode_fns

MODEL_CFG = dict(
    embedding_size=576,
    num_heads=9,
    num_kv_heads=3,
    num_layers=30,
    feed_forward_size=1536,
    rope_dim=64,
    context_length=2048,
)
TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
CHECKPOINT_PATH = "smollm-135m.npz"


def load_npz(path):
    NAME_KEY = "__checkpoint_name__"
    META_PREFIX = "__meta__/"
    with np.load(path) as npz:
        flat = {
            tuple(k.split("/")): v
            for k, v in npz.items()
            if k != NAME_KEY and not k.startswith(META_PREFIX)
        }
    return unflatten_dict(flat)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def parse_args():
    p = argparse.ArgumentParser(description="SmolLM-135M greedy inference")
    p.add_argument("--prompt", default="Once upon")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    ckpt = _resolve(CHECKPOINT_PATH)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"Using checkpoint: {ckpt}")
    params_dict = load_npz(ckpt)
    vocab_size = int(params_dict["Embed_0"]["embedding"].shape[0])
    print(f"Detected vocab_size={vocab_size} from checkpoint")
    if len(tokenizer) != vocab_size:
        raise ValueError(f"Tokenizer/checkpoint vocab mismatch: {len(tokenizer)} != {vocab_size}")

    params = jax.tree_util.tree_map(jnp.asarray, params_dict)

    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=MODEL_CFG["context_length"],
        d_model=MODEL_CFG["embedding_size"],
        n_heads=MODEL_CFG["num_heads"],
        d_ff=MODEL_CFG["feed_forward_size"],
        n_layers=MODEL_CFG["num_layers"],
        num_kv_heads=MODEL_CFG["num_kv_heads"],
        rotary_dim=MODEL_CFG["rope_dim"],
        compute_dtype="bfloat16",
    )

    key_params, key_dropout = jax.random.split(jax.random.PRNGKey(0))
    _, nonparam = init_inference_state(model, key_params, key_dropout, batch_size=1,
                                       pad_token_id=tokenizer.pad_token_id)

    params = jax.device_put(params)
    nonparam = jax.device_put(nonparam)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if prompt_ids and tokenizer.eos_token_id is not None and prompt_ids[-1] == tokenizer.eos_token_id:
        prompt_ids = prompt_ids[:-1]
    prompt_ids = np.asarray(prompt_ids[-MODEL_CFG["context_length"]:], dtype=np.int32)
    prompt_len = len(prompt_ids)

    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)
    state = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), nonparam)
    prompt = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)

    nonparam, t_cur, last_tok = prefill_fn(params, state, prompt)
    for leaf in jax.tree_util.tree_leaves((nonparam, last_tok)):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()

    warm_state = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), nonparam)
    warm_tokens, _ = decode_fn(params, warm_state, last_tok, t_cur, steps=args.steps)
    warm_tokens.block_until_ready()

    timed_state = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), nonparam)
    t0 = time.perf_counter()
    tokens_new, _ = decode_fn(params, timed_state, last_tok, t_cur, steps=args.steps)
    tokens_new.block_until_ready()
    decode_t = time.perf_counter() - t0

    tokens_new = np.asarray(tokens_new[0])
    full = np.concatenate([prompt_ids, tokens_new])
    text = tokenizer.decode(full, skip_special_tokens=True)

    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)

    if args.verbose:
        print(f"\n[perf] prompt={prompt_len} gen={args.steps} "
              f"decode_only={decode_t:.4f}s tok/s={args.steps / decode_t:.2f}")


if __name__ == "__main__":
    main()
