import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm


IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())

def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


def _apply_partial_rope(x, sin, cos, rot_dim):
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x1, x2 = jnp.split(x_rot, 2, axis=-1)
    x_rot = (x_rot * cos) + (jnp.concatenate((-x2, x1), axis=-1) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


class NativeJaxSelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    context_length: int = 2048
    dropout_rate: float = 0.0
    num_kv: int = 1
    rotary_dim: int | None = None

    def setup(self):
        self.head_dim = self.qkv_features // self.num_heads
        self._rotary_dim = self.rotary_dim if self.rotary_dim is not None else self.head_dim
        assert self._rotary_dim % 2 == 0, "rotary_dim must be even"

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(total_out, use_bias=False, name="qkv_proj",
                                 dtype=self.dtype, param_dtype=self.param_dtype)
        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj",
                               dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self._rope_sin, self._rope_cos = _build_rope_cache(
            self.context_length, self._rotary_dim, self.dtype)

    @nn.compact
    def __call__(self, x, *, deterministic: bool, cur_index: jnp.ndarray | int | None = None):
        b, l, _ = x.shape
        impl = "xla"
        hd = self.head_dim
        q_size = self.num_heads * hd
        kv_size = self.num_kv * hd

        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q.reshape(b, l, self.num_heads, hd)
        k = k.reshape(b, l, self.num_kv, hd)
        v = v.reshape(b, l, self.num_kv, hd)
        kv_indices = None
        if self.num_kv != self.num_heads:
            kv_indices = jnp.arange(self.num_heads) // max(1, self.num_heads // self.num_kv)

        cur_index = jnp.asarray(cur_index, jnp.int32)
        if cur_index.ndim == 0:
            sin = jnp.broadcast_to(
                jax.lax.dynamic_slice(self._rope_sin, (0, cur_index, 0, 0), (1, l, 1, self._rotary_dim)),
                (b, l, 1, self._rotary_dim))
            cos = jnp.broadcast_to(
                jax.lax.dynamic_slice(self._rope_cos, (0, cur_index, 0, 0), (1, l, 1, self._rotary_dim)),
                (b, l, 1, self._rotary_dim))
        else:
            positions = cur_index[:, None] + jnp.arange(l, dtype=jnp.int32)[None, :]
            sin = jnp.take(self._rope_sin[0], positions, axis=0)
            cos = jnp.take(self._rope_cos[0], positions, axis=0)
        q = _apply_partial_rope(q, sin, cos, self._rotary_dim)
        k = _apply_partial_rope(k, sin, cos, self._rotary_dim)

        cache_shape = (b, self.num_kv, self.context_length, hd)
        cached_k = self.variable("cache", "k", jnp.zeros, cache_shape, self.dtype)
        cached_v = self.variable("cache", "v", jnp.zeros, cache_shape, self.dtype)

        k_to_cache = jnp.swapaxes(k, 1, 2)
        v_to_cache = jnp.swapaxes(v, 1, 2)

        if cur_index.ndim == 0:
            if l == 1:
                new_k = jax.lax.dynamic_update_slice(cached_k.value, k_to_cache, (0, 0, cur_index, 0))
                new_v = jax.lax.dynamic_update_slice(cached_v.value, v_to_cache, (0, 0, cur_index, 0))
            else:
                new_k = cached_k.value.at[:, :, cur_index:cur_index + l, :].set(k_to_cache)
                new_v = cached_v.value.at[:, :, cur_index:cur_index + l, :].set(v_to_cache)
            cur_max = cur_index + (l - 1)
            attn_bias = jnp.where(jnp.arange(self.context_length) <= cur_max, 0.0, -1e10)
            attn_bias = attn_bias[None, None, None, :].astype(self.dtype)
        else:
            pos = cur_index[:, None] + jnp.arange(l, dtype=jnp.int32)[None, :]
            batch_idx = jnp.arange(b)[:, None]
            new_k = cached_k.value.at[batch_idx, :, pos, :].set(jnp.transpose(k_to_cache, (0, 2, 1, 3)))
            new_v = cached_v.value.at[batch_idx, :, pos, :].set(jnp.transpose(v_to_cache, (0, 2, 1, 3)))
            cur_max = cur_index + (l - 1)
            attn_bias = jnp.where(jnp.arange(self.context_length)[None, :] <= cur_max[:, None], 0.0, -1e10)
            attn_bias = attn_bias[:, None, None, :].astype(self.dtype)

        cached_k.value = new_k
        cached_v.value = new_v

        k_full = jnp.swapaxes(cached_k.value, 1, 2)
        v_full = jnp.swapaxes(cached_v.value, 1, 2)
        if kv_indices is not None and l == 1:
            k_full = k_full[:, :128, :, :]
            v_full = v_full[:, :128, :, :]
            group = self.num_heads // self.num_kv
            q_grouped = q.reshape(b, self.num_kv, group, hd)
            scale = jnp.asarray(hd, self.dtype) ** jnp.asarray(-0.5, self.dtype)
            scores = jnp.einsum("bkgh,btkh->bkgt", q_grouped, k_full) * scale
            scores = scores + attn_bias[:, :, 0, :128]
            weights = jax.nn.softmax(scores, axis=-1)
            y = jnp.einsum("bkgt,btkh->bkgh", weights, v_full).reshape(b, l, self.num_heads, hd)
        else:
            if kv_indices is not None:
                k_full = jnp.take(k_full, kv_indices, axis=2)
                v_full = jnp.take(v_full, kv_indices, axis=2)

            y = jax.nn.dot_product_attention(q, k_full, v_full, bias=attn_bias,
                                              is_causal=False, implementation=impl)
        y = y.reshape(b, l, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    context_length: int = 2048
    dropout_rate: float = 0.0
    num_kv_heads: int | None = None
    rotary_dim: int | None = None

    @nn.compact
    def __call__(self, x, *, deterministic: bool, cur_index: jnp.ndarray | int | None = None):
        num_kv = self.num_kv_heads if self.num_kv_heads is not None else self.n_heads

        residual = x
        x = RMSNorm(name="rms1", dtype=self.dtype, epsilon=1e-5)(x)
        x = NativeJaxSelfAttention(
            num_heads=self.n_heads, num_kv=num_kv,
            qkv_features=self.d_model, context_length=self.context_length,
            dropout_rate=self.dropout_rate, dtype=self.dtype,
            param_dtype=self.param_dtype, rotary_dim=self.rotary_dim,
        )(x, deterministic=deterministic, cur_index=cur_index)
        x = residual + x

        residual = x
        x = RMSNorm(name="rms2", dtype=self.dtype, epsilon=1e-5)(x)
        h = nn.Dense(self.d_ff * 2, name="fc1", dtype=self.dtype,
                     param_dtype=self.param_dtype, use_bias=False)(x)
        u, v = jnp.split(h, 2, axis=-1)
        x = nn.silu(u) * v
        x = nn.Dense(self.d_model, name="fc2", dtype=self.dtype,
                     param_dtype=self.param_dtype, use_bias=False)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return residual + x
