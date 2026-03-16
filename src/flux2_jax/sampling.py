"""
Flux 2 sampling / denoising loop — JAX port.
"""

import math

import jax
import jax.numpy as jnp

from .model import flux2_forward, flux2_forward_scan, Klein4BParams


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def generalized_time_snr_shift(t, mu, sigma):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = [i / num_steps for i in range(num_steps, -1, -1)]  # linspace 1->0
    timesteps = [generalized_time_snr_shift(t, mu, 1.0) if t > 0 else 0.0 for t in timesteps]
    return timesteps


def denoise(params, config, img, img_ids, txt, txt_ids, timesteps, guidance, use_scan=True):
    """
    Multi-step denoising loop.

    params: model params pytree
    config: Klein4BParams
    img: [B, L_img, C] initial noisy latents
    img_ids: [B, L_img, 4] position ids
    txt: [B, L_txt, D_txt] text embeddings
    txt_ids: [B, L_txt, 4] text position ids
    timesteps: list of float, from ~1 to 0
    guidance: float
    use_scan: use scan-based forward (faster compilation)
    """
    forward_fn = flux2_forward_scan if use_scan else flux2_forward

    guidance_vec = jnp.full((img.shape[0],), guidance, dtype=img.dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = forward_fn(
            params, img, img_ids, t_vec, txt, txt_ids, guidance_vec, config,
        )
        img = img + (t_prev - t_curr) * pred

    return img


def denoise_jitted(params, config, img, img_ids, txt, txt_ids, timesteps, guidance, use_scan=True):
    """
    JIT-compiled single denoising step for benchmarking.
    Returns a function that does one step (for profiling per-step latency).
    """
    forward_fn = flux2_forward_scan if use_scan else flux2_forward

    @jax.jit
    def step(img, t_curr, t_prev, guidance_vec):
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = forward_fn(
            params, img, img_ids, t_vec, txt, txt_ids, guidance_vec, config,
        )
        return img + (t_prev - t_curr) * pred

    return step


# ── Preprocessing (JAX versions) ────────────────────────────────────────────

def prc_img_jax(x, t_coord=None):
    """
    Process image latents into flat tokens + position IDs.
    x: [C, H, W] single image latent
    Returns: tokens [H*W, C], ids [H*W, 4]
    """
    C, H, W = x.shape
    t_range = jnp.array([0]) if t_coord is None else t_coord
    h_range = jnp.arange(H)
    w_range = jnp.arange(W)
    l_range = jnp.array([0])

    # Cartesian product
    t_grid, h_grid, w_grid, l_grid = jnp.meshgrid(t_range, h_range, w_range, l_range, indexing='ij')
    ids = jnp.stack([t_grid.ravel(), h_grid.ravel(), w_grid.ravel(), l_grid.ravel()], axis=-1)

    # Flatten spatial dims
    tokens = x.reshape(C, -1).T  # [H*W, C]
    return tokens, ids


def prc_txt_jax(x, t_coord=None):
    """
    Process text embeddings into tokens + position IDs.
    x: [L, D] text embeddings
    Returns: tokens [L, D], ids [L, 4]
    """
    L, D = x.shape
    t_range = jnp.array([0]) if t_coord is None else t_coord
    h_range = jnp.array([0])
    w_range = jnp.array([0])
    l_range = jnp.arange(L)

    t_grid, h_grid, w_grid, l_grid = jnp.meshgrid(t_range, h_range, w_range, l_range, indexing='ij')
    ids = jnp.stack([t_grid.ravel(), h_grid.ravel(), w_grid.ravel(), l_grid.ravel()], axis=-1)

    return x, ids
