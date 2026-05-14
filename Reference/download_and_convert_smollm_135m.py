#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file


EXPECTED = dict(
    vocab_size=49152,
    hidden_size=576,
    num_attention_heads=9,
    num_key_value_heads=3,
    num_hidden_layers=30,
    intermediate_size=1536,
    max_position_embeddings=2048,
)


def _fp32(x):
    return np.asarray(x, dtype=np.float32)


def _load_hf_snapshot(repo: str, cache_dir: str | None):
    root = Path(snapshot_download(
        repo_id=repo,
        cache_dir=cache_dir,
        allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
    ))
    cfg = json.loads((root / "config.json").read_text())
    sd = {}
    for path in sorted(root.glob("*.safetensors")):
        sd.update(load_file(path))
    if not sd:
        raise FileNotFoundError(f"No .safetensors files found in downloaded snapshot: {root}")
    return cfg, sd


def _check_config(cfg):
    for key, expected in EXPECTED.items():
        actual = int(cfg[key])
        if actual != expected:
            raise ValueError(f"Config mismatch for {key}: HF={actual}, expected={expected}")


def convert(repo: str, out: Path, cache_dir: str | None):
    print(f"[download] {repo}")
    cfg, sd = _load_hf_snapshot(repo, cache_dir)
    _check_config(cfg)

    params = {}
    params[("Embed_0", "embedding")] = _fp32(sd["model.embed_tokens.weight"])
    params[("final_norm", "scale")] = _fp32(sd["model.norm.weight"])

    for i in range(EXPECTED["num_hidden_layers"]):
        block = f"TinyTransformerBlock_{i}"
        prefix = f"model.layers.{i}."
        attn = f"{block}/NativeJaxSelfAttention_0"

        params[(block, "rms1", "scale")] = _fp32(sd[prefix + "input_layernorm.weight"])
        params[(block, "rms2", "scale")] = _fp32(sd[prefix + "post_attention_layernorm.weight"])

        q = _fp32(sd[prefix + "self_attn.q_proj.weight"]).T
        k = _fp32(sd[prefix + "self_attn.k_proj.weight"]).T
        v = _fp32(sd[prefix + "self_attn.v_proj.weight"]).T
        params[tuple(attn.split("/")) + ("qkv_proj", "kernel")] = np.concatenate([q, k, v], axis=1)
        params[tuple(attn.split("/")) + ("o_proj", "kernel")] = _fp32(sd[prefix + "self_attn.o_proj.weight"]).T

        gate = _fp32(sd[prefix + "mlp.gate_proj.weight"]).T
        up = _fp32(sd[prefix + "mlp.up_proj.weight"]).T
        down = _fp32(sd[prefix + "mlp.down_proj.weight"]).T
        params[(block, "fc1", "kernel")] = np.concatenate([gate, up], axis=1)
        params[(block, "fc2", "kernel")] = down

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{"/".join(k): np.asarray(v, dtype=np.float32) for k, v in params.items()})
    print(f"[done] wrote {out}")


def main():
    p = argparse.ArgumentParser(description="Download SmolLM-135M from HF and convert to FP32 GIANT/JAX NPZ.")
    p.add_argument("--hf-repo", default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--out", default="smollm-135m.npz")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    convert(args.hf_repo, Path(args.out).expanduser(), args.cache_dir)


if __name__ == "__main__":
    main()
