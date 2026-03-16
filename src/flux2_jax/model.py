"""
Flux 2 Klein 4B flow model — pure JAX/functional port.

All ops are pure functions operating on a params pytree.
No Flax/nn.Module overhead — just jax.jit the forward pass.

Klein 4B config:
  hidden_size=3072, num_heads=24, head_dim=128
  depth=5 (double blocks), depth_single=20 (single blocks)
  mlp_ratio=3.0, mlp_hidden_dim=9216
  in_channels=128, context_in_dim=7680
  axes_dim=[32,32,32,32], theta=2000
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@dataclass
class Klein4BParams:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


# ── Utility functions ────────────────────────────────────────────────────────

def linear(x, w, b=None):
    """x @ w.T + b. Weights stored as [out, in] (PyTorch convention)."""
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def silu(x):
    return jax.nn.silu(x)


def silu_gate(x):
    """SiLU-gated activation: silu(x1) * x2 where [x1,x2] = split(x)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return silu(x1) * x2


def layer_norm(x, eps=1e-6):
    """LayerNorm without learned parameters (elementwise_affine=False)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def rms_norm(x, scale, eps=1e-6):
    """RMSNorm: x * rsqrt(mean(x^2)) * scale."""
    x_f32 = x.astype(jnp.float32)
    rrms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return (x_f32 * rrms).astype(x.dtype) * scale


# ── Positional encoding ─────────────────────────────────────────────────────

def rope(pos, dim, theta):
    """Compute RoPE rotation matrices for one axis."""
    assert dim % 2 == 0
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta ** scale)
    # pos: [..., N], omega: [D/2]
    out = jnp.einsum("...n,d->...nd", pos, omega)
    cos_out = jnp.cos(out)
    sin_out = jnp.sin(out)
    # Build 2x2 rotation matrix per position
    # [cos, -sin; sin, cos]
    rot = jnp.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
    rot = rot.reshape(*rot.shape[:-1], 2, 2)
    return rot.astype(jnp.float32)


def embed_nd(ids, axes_dim, theta):
    """N-dimensional RoPE embedding."""
    embs = []
    for i, d in enumerate(axes_dim):
        embs.append(rope(ids[..., i], d, theta))
    emb = jnp.concatenate(embs, axis=-3)
    return emb[:, None, :, :, :]  # [B, 1, L, D/2, 2, 2]


def apply_rope(xq, xk, freqs_cis):
    """Apply RoPE using complex-number formulation (faster, bit-exact).
    xq, xk: [B, H, L, D]
    freqs_cis: [B, 1, L, D/2, 2, 2]
    """
    # Extract cos, sin from rotation matrices
    cos_f = freqs_cis[..., 0, 0]  # cos theta
    sin_f = freqs_cis[..., 1, 0]  # sin theta
    # Reshape to pairs
    xq_c = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_c = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    # Complex multiply: (a + bi)(cos + i*sin) = (a*cos - b*sin) + (a*sin + b*cos)i
    xq_out = jnp.stack([
        xq_c[..., 0] * cos_f - xq_c[..., 1] * sin_f,
        xq_c[..., 0] * sin_f + xq_c[..., 1] * cos_f,
    ], axis=-1).reshape(*xq.shape)
    xk_out = jnp.stack([
        xk_c[..., 0] * cos_f - xk_c[..., 1] * sin_f,
        xk_c[..., 0] * sin_f + xk_c[..., 1] * cos_f,
    ], axis=-1).reshape(*xk.shape)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


# ── Timestep embedding ──────────────────────────────────────────────────────

def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    """Sinusoidal timestep embedding. t: [B]."""
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding.astype(t.dtype)


# ── MLPEmbedder ──────────────────────────────────────────────────────────────

def mlp_embedder(x, params):
    """Two-layer MLP: Linear -> SiLU -> Linear."""
    x = linear(x, params['in_layer.weight'])
    x = silu(x)
    x = linear(x, params['out_layer.weight'])
    return x


# ── Modulation ───────────────────────────────────────────────────────────────

def modulation(vec, params, is_double):
    """Compute modulation parameters from conditioning vector."""
    out = linear(silu(vec), params['lin.weight'])
    if out.ndim == 2:
        out = out[:, None, :]
    multiplier = 6 if is_double else 3
    chunks = jnp.split(out, multiplier, axis=-1)
    mod1 = (chunks[0], chunks[1], chunks[2])
    mod2 = (chunks[3], chunks[4], chunks[5]) if is_double else None
    return mod1, mod2


# ── QKNorm ───────────────────────────────────────────────────────────────────

def qk_norm(q, k, v, q_scale, k_scale):
    """Apply RMSNorm to Q and K, cast to V dtype."""
    q = rms_norm(q, q_scale)
    k = rms_norm(k, k_scale)
    return q.astype(v.dtype), k.astype(v.dtype)


# ── Attention ────────────────────────────────────────────────────────────────

def attention(q, k, v):
    """Scaled dot-product attention. q,k,v: [B, H, L, D]."""
    scale = q.shape[-1] ** -0.5
    attn_weights = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    return jnp.einsum("bhij,bhjd->bhid", attn_weights, v)


def causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens):
    """
    Causal attention: [txt, ref, img] layout.
    txt+img attend to all, ref self-attends only.
    """
    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens

    if num_ref_tokens == 0:
        # No ref tokens — simple full attention
        out = attention(q, k, v)
        # [B, H, L, D] -> [B, L, H*D]
        B, H, L, D = out.shape
        return out.transpose(0, 2, 1, 3).reshape(B, L, H * D)

    # Split queries
    q_txt = q[:, :, :ref_start, :]
    q_ref = q[:, :, ref_start:ref_end, :]
    q_img = q[:, :, ref_end:, :]

    # Split keys/values
    k_ref = k[:, :, ref_start:ref_end, :]
    v_ref = v[:, :, ref_start:ref_end, :]

    # txt+img attend to all keys
    q_txt_img = jnp.concatenate([q_txt, q_img], axis=2)
    attn_txt_img = attention(q_txt_img, k, v)
    attn_txt = attn_txt_img[:, :, :ref_start, :]
    attn_img = attn_txt_img[:, :, ref_start:, :]

    # ref only attends to itself
    attn_ref = attention(q_ref, k_ref, v_ref)

    out = jnp.concatenate([attn_txt, attn_ref, attn_img], axis=2)
    B, H, L, D = out.shape
    return out.transpose(0, 2, 1, 3).reshape(B, L, H * D)


# ── DoubleStreamBlock ────────────────────────────────────────────────────────

def double_stream_block(img, txt, pe, pe_ctx, mod_img, mod_txt, params, num_heads, num_ref_tokens):
    """One double-stream block: separate img/txt attention + cross-attend."""
    hidden_size = img.shape[-1]
    head_dim = hidden_size // num_heads
    img_mod1, img_mod2 = mod_img
    txt_mod1, txt_mod2 = mod_txt

    img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
    img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
    txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
    txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

    # Prepare image for attention
    img_modulated = layer_norm(img)
    img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
    img_qkv = linear(img_modulated, params['img_attn.qkv.weight'])
    B, L_img = img_qkv.shape[0], img_qkv.shape[1]
    img_qkv = img_qkv.reshape(B, L_img, 3, num_heads, head_dim)
    img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]
    # [B, L, H, D] -> [B, H, L, D]
    img_q = img_q.transpose(0, 2, 1, 3)
    img_k = img_k.transpose(0, 2, 1, 3)
    img_v = img_v.transpose(0, 2, 1, 3)
    img_q, img_k = qk_norm(img_q, img_k, img_v,
                           params['img_attn.norm.query_norm.scale'],
                           params['img_attn.norm.key_norm.scale'])

    # Prepare text for attention
    txt_modulated = layer_norm(txt)
    txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
    txt_qkv = linear(txt_modulated, params['txt_attn.qkv.weight'])
    L_txt = txt_qkv.shape[1]
    txt_qkv = txt_qkv.reshape(B, L_txt, 3, num_heads, head_dim)
    txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]
    txt_q = txt_q.transpose(0, 2, 1, 3)
    txt_k = txt_k.transpose(0, 2, 1, 3)
    txt_v = txt_v.transpose(0, 2, 1, 3)
    txt_q, txt_k = qk_norm(txt_q, txt_k, txt_v,
                           params['txt_attn.norm.query_norm.scale'],
                           params['txt_attn.norm.key_norm.scale'])

    # Joint attention
    q = jnp.concatenate([txt_q, img_q], axis=2)
    k = jnp.concatenate([txt_k, img_k], axis=2)
    v = jnp.concatenate([txt_v, img_v], axis=2)

    num_txt_tokens = L_txt
    pe_full = jnp.concatenate([pe_ctx, pe], axis=2)
    q, k = apply_rope(q, k, pe_full)

    attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)
    txt_attn = attn[:, :num_txt_tokens, :]
    img_attn = attn[:, num_txt_tokens:, :]

    # Apply residuals — image stream
    img = img + img_mod1_gate * linear(img_attn, params['img_attn.proj.weight'])
    img_normed = layer_norm(img)
    img_mlp_in = (1 + img_mod2_scale) * img_normed + img_mod2_shift
    img_mlp_out = linear(img_mlp_in, params['img_mlp.0.weight'])
    img_mlp_out = silu_gate(img_mlp_out)
    img_mlp_out = linear(img_mlp_out, params['img_mlp.2.weight'])
    img = img + img_mod2_gate * img_mlp_out

    # Apply residuals — text stream
    txt = txt + txt_mod1_gate * linear(txt_attn, params['txt_attn.proj.weight'])
    txt_normed = layer_norm(txt)
    txt_mlp_in = (1 + txt_mod2_scale) * txt_normed + txt_mod2_shift
    txt_mlp_out = linear(txt_mlp_in, params['txt_mlp.0.weight'])
    txt_mlp_out = silu_gate(txt_mlp_out)
    txt_mlp_out = linear(txt_mlp_out, params['txt_mlp.2.weight'])
    txt = txt + txt_mod2_gate * txt_mlp_out

    return img, txt


# ── SingleStreamBlock ────────────────────────────────────────────────────────

def single_stream_block(x, pe, mod, params, num_heads, num_txt_tokens, num_ref_tokens):
    """One single-stream block: fused QKV + MLP."""
    hidden_size = x.shape[-1]
    head_dim = hidden_size // num_heads
    mlp_hidden_dim = int(hidden_size * 3.0)  # mlp_ratio=3.0

    mod_shift, mod_scale, mod_gate = mod

    # adaLN modulation
    x_mod = (1 + mod_scale) * layer_norm(x) + mod_shift

    # Fused linear1: projects to [QKV, MLP_input]
    y = linear(x_mod, params['linear1.weight'])
    qkv_dim = 3 * hidden_size
    qkv, mlp = jnp.split(y, [qkv_dim], axis=-1)

    B, L = qkv.shape[0], qkv.shape[1]
    qkv = qkv.reshape(B, L, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    q, k = qk_norm(q, k, v,
                   params['norm.query_norm.scale'],
                   params['norm.key_norm.scale'])

    q, k = apply_rope(q, k, pe)

    attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)

    # Output: concat attention + gated MLP, project back
    mlp_out = silu_gate(mlp)
    output = linear(jnp.concatenate([attn, mlp_out], axis=-1), params['linear2.weight'])
    return x + mod_gate * output


# ── LastLayer ────────────────────────────────────────────────────────────────

def last_layer(x, vec, params):
    """Final adaLN modulation + linear projection."""
    mod = linear(silu(vec), params['adaLN_modulation.1.weight'])
    shift, scale = jnp.split(mod, 2, axis=-1)
    if shift.ndim == 2:
        shift = shift[:, None, :]
        scale = scale[:, None, :]
    x = (1 + scale) * layer_norm(x) + shift
    x = linear(x, params['linear.weight'])
    return x


# ── Full model forward ──────────────────────────────────────────────────────

def flux2_forward(params, x, x_ids, timesteps, ctx, ctx_ids, guidance, config):
    """
    Full Flux 2 forward pass.

    params: nested dict of JAX arrays
    x: [B, L_img, in_channels] — image latents
    x_ids: [B, L_img, 4] — image position ids
    timesteps: [B] — timestep values
    ctx: [B, L_txt, context_in_dim] — text embeddings
    ctx_ids: [B, L_txt, 4] — text position ids
    guidance: [B] — guidance values
    config: Klein4BParams
    """
    num_heads = config.num_heads
    num_txt_tokens = ctx.shape[1]

    # Timestep + guidance embedding
    t_emb = timestep_embedding(timesteps, 256)
    vec = mlp_embedder(t_emb, params['time_in'])
    if config.use_guidance_embed:
        g_emb = timestep_embedding(guidance, 256)
        vec = vec + mlp_embedder(g_emb, params['guidance_in'])

    # Compute all modulations upfront
    double_mod_img = modulation(vec, params['double_stream_modulation_img'], is_double=True)
    double_mod_txt = modulation(vec, params['double_stream_modulation_txt'], is_double=True)
    single_mod, _ = modulation(vec, params['single_stream_modulation'], is_double=False)

    # Input projections
    img = linear(x, params['img_in.weight'])
    txt = linear(ctx, params['txt_in.weight'])

    # Positional embeddings
    pe_x = embed_nd(x_ids, config.axes_dim, config.theta)
    pe_ctx = embed_nd(ctx_ids, config.axes_dim, config.theta)

    # Double stream blocks
    for i in range(config.depth):
        img, txt = double_stream_block(
            img, txt, pe_x, pe_ctx,
            double_mod_img, double_mod_txt,
            params[f'double_blocks.{i}'],
            num_heads, num_ref_tokens=0,
        )

    # Merge streams
    img = jnp.concatenate([txt, img], axis=1)
    pe = jnp.concatenate([pe_ctx, pe_x], axis=2)

    # Single stream blocks
    for i in range(config.depth_single_blocks):
        img = single_stream_block(
            img, pe, single_mod,
            params[f'single_blocks.{i}'],
            num_heads, num_txt_tokens, num_ref_tokens=0,
        )

    # Extract image tokens
    img = img[:, num_txt_tokens:, :]

    # Final layer
    img = last_layer(img, vec, params['final_layer'])
    return img


# ── Scan-based forward (optimized for TPU compilation) ───────────────────────

def flux2_forward_scan(params, x, x_ids, timesteps, ctx, ctx_ids, guidance, config):
    """
    Same as flux2_forward but uses jax.lax.scan over identical blocks.
    Reduces compilation time and HLO program size.
    """
    num_heads = config.num_heads
    num_txt_tokens = ctx.shape[1]

    # Timestep + guidance embedding
    t_emb = timestep_embedding(timesteps, 256)
    vec = mlp_embedder(t_emb, params['time_in'])
    if config.use_guidance_embed:
        g_emb = timestep_embedding(guidance, 256)
        vec = vec + mlp_embedder(g_emb, params['guidance_in'])

    # Compute all modulations upfront
    double_mod_img = modulation(vec, params['double_stream_modulation_img'], is_double=True)
    double_mod_txt = modulation(vec, params['double_stream_modulation_txt'], is_double=True)
    single_mod, _ = modulation(vec, params['single_stream_modulation'], is_double=False)

    # Input projections
    img = linear(x, params['img_in.weight'])
    txt = linear(ctx, params['txt_in.weight'])

    # Positional embeddings
    pe_x = embed_nd(x_ids, config.axes_dim, config.theta)
    pe_ctx = embed_nd(ctx_ids, config.axes_dim, config.theta)

    # ── Scan over double blocks ──
    def double_block_scan_fn(carry, block_params):
        img, txt = carry
        img, txt = double_stream_block(
            img, txt, pe_x, pe_ctx,
            double_mod_img, double_mod_txt,
            block_params, num_heads, num_ref_tokens=0,
        )
        return (img, txt), None

    # Stack double block params: each leaf becomes [depth, ...]
    double_block_params = _stack_block_params(params, 'double_blocks', config.depth)
    (img, txt), _ = jax.lax.scan(double_block_scan_fn, (img, txt), double_block_params, length=config.depth)

    # Merge streams
    img = jnp.concatenate([txt, img], axis=1)
    pe = jnp.concatenate([pe_ctx, pe_x], axis=2)

    # ── Scan over single blocks ──
    def single_block_scan_fn(carry, block_params):
        img = carry
        img = single_stream_block(
            img, pe, single_mod,
            block_params, num_heads, num_txt_tokens, num_ref_tokens=0,
        )
        return img, None

    single_block_params = _stack_block_params(params, 'single_blocks', config.depth_single_blocks)
    img, _ = jax.lax.scan(single_block_scan_fn, img, single_block_params, length=config.depth_single_blocks)

    # Extract image tokens
    img = img[:, num_txt_tokens:, :]

    # Final layer
    img = last_layer(img, vec, params['final_layer'])
    return img


def _stack_block_params(params, prefix, num_blocks):
    """Stack parameters from block.0, block.1, ... into a single pytree with leading dim."""
    block_params_list = [params[f'{prefix}.{i}'] for i in range(num_blocks)]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *block_params_list)
