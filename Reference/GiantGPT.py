import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
from Transformer_block import TinyTransformerBlock

class GiantGPT(nn.Module):
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    param_dtype: str = "float32"
    compute_dtype: str = "float32"
    dropout_rate: float = 0.0
    num_kv_heads: int | None = None
    rotary_dim: int | None = None

    @nn.compact
    def __call__(self, tokens, *, deterministic=True, cur_index=None):
        pd = jnp.dtype(self.param_dtype)
        cd = jnp.dtype(self.compute_dtype)

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=cd, param_dtype=pd,
        )
        x = embed(tokens)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
                context_length=self.context_length, dropout_rate=self.dropout_rate,
                num_kv_heads=self.num_kv_heads, rotary_dim=self.rotary_dim,
                dtype=cd, param_dtype=pd,
            )(x, deterministic=deterministic, cur_index=cur_index)

        x = RMSNorm(name="final_norm", dtype=cd, epsilon=1e-5)(x)
        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embed.embedding.astype(jnp.float32))
        return logits
