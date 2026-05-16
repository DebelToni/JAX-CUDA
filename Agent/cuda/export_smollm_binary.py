#!/usr/bin/env python
"""Export the Reference NPZ into a simple flat binary for the CUDA runner."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "Reference" / "smollm-135m.npz"
OUT = Path(__file__).resolve().parent / "smollm135m_f32.bin"
PROMPT = "Technical note: In transformer language models, rotary position embeddings work by rotating query and key vectors in pairs. This helps attention layers represent relative positions because"

D = 576
LAYERS = 30


def arr(npz, name):
    return np.asarray(npz[name], dtype=np.float32, order="C")


def write_array(f, a):
    a = np.asarray(a, dtype=np.float32, order="C")
    f.write(struct.pack("<Q", a.size))
    f.write(a.tobytes(order="C"))


def main():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer", use_fast=True)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    if ids and tok.eos_token_id is not None and ids[-1] == tok.eos_token_id:
        ids = ids[:-1]
    with np.load(NPZ) as z, open(OUT, "wb") as f:
        f.write(b"SMOLCUDA1")
        f.write(struct.pack("<7i", D, 49152, LAYERS, 9, 3, 64, 1536))
        f.write(struct.pack("<i", len(ids)))
        f.write(np.asarray(ids, dtype=np.int32).tobytes())
        write_array(f, arr(z, "Embed_0/embedding"))
        write_array(f, arr(z, "final_norm/scale"))
        for i in range(LAYERS):
            b = f"TinyTransformerBlock_{i}"
            a = f"{b}/NativeJaxSelfAttention_0"
            for name in [
                f"{b}/rms1/scale",
                f"{b}/rms2/scale",
                f"{a}/qkv_proj/kernel",
                f"{a}/o_proj/kernel",
                f"{b}/fc1/kernel",
                f"{b}/fc2/kernel",
            ]:
                write_array(f, arr(z, name))
    print(OUT)


if __name__ == "__main__":
    main()
