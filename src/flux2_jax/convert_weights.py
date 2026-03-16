"""
Convert Flux 2 Klein 4B PyTorch safetensors weights to JAX arrays.

The PyTorch model stores weights as nn.Linear.weight with shape [out, in].
We keep the same convention in JAX (our linear() does x @ w.T).
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
from safetensors import safe_open


def convert_safetensors_to_jax(weight_path: str, dtype=jnp.bfloat16) -> dict:
    """
    Load PyTorch safetensors and convert to a nested JAX params dict.

    Returns a flat dict mapping PyTorch key names to JAX arrays.
    The keys use dots as separators (e.g., 'double_blocks.0.img_attn.qkv.weight').
    """
    params = {}
    with safe_open(weight_path, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            params[key] = jnp.array(tensor, dtype=dtype)
    return params


def nest_params(flat_params: dict) -> dict:
    """
    Convert flat dot-separated keys into a nested dict structure.

    'double_blocks.0.img_attn.qkv.weight' ->
    {'double_blocks.0': {'img_attn.qkv.weight': ...}}

    We use a specific nesting strategy that matches our model functions:
    - Top-level: time_in, img_in, txt_in, double/single_stream_modulation_*,
                 double_blocks.N, single_blocks.N, final_layer
    - Within blocks: img_attn.qkv.weight, img_attn.proj.weight, img_attn.norm.*,
                     img_mlp.0.weight, img_mlp.2.weight, etc.
    """
    nested = {}

    for key, value in flat_params.items():
        parts = key.split('.')

        # Determine the top-level group
        if parts[0] in ('time_in', 'guidance_in'):
            group = parts[0]
            subkey = '.'.join(parts[1:])
        elif parts[0] in ('img_in', 'txt_in'):
            group = parts[0]
            subkey = '.'.join(parts[1:])
            # img_in.weight -> group='img_in', subkey='weight'
            nested.setdefault(f'{group}.{subkey}', value)
            continue
        elif parts[0] == 'pe_embedder':
            continue  # EmbedND has no learned params
        elif parts[0] in ('double_stream_modulation_img', 'double_stream_modulation_txt',
                          'single_stream_modulation'):
            group = parts[0]
            subkey = '.'.join(parts[1:])
        elif parts[0] == 'double_blocks':
            group = f'double_blocks.{parts[1]}'
            subkey = '.'.join(parts[2:])
        elif parts[0] == 'single_blocks':
            group = f'single_blocks.{parts[1]}'
            subkey = '.'.join(parts[2:])
        elif parts[0] == 'final_layer':
            group = 'final_layer'
            subkey = '.'.join(parts[1:])
        else:
            # Fallback: put at top level
            nested[key] = value
            continue

        if group not in nested:
            nested[group] = {}
        nested[group][subkey] = value

    return nested


def download_and_convert(
    repo_id: str = "black-forest-labs/FLUX.2-klein-4B",
    filename: str = "flux-2-klein-4b.safetensors",
    output_dir: str = "weights",
    hf_token: str | None = None,
    dtype=jnp.bfloat16,
) -> dict:
    """Download weights from HuggingFace and convert to JAX."""
    import huggingface_hub

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {repo_id}/{filename}...")
    weight_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=hf_token,
    )
    print(f"Downloaded to {weight_path}")

    print("Converting to JAX arrays...")
    flat_params = convert_safetensors_to_jax(weight_path, dtype=dtype)
    params = nest_params(flat_params)

    # Print summary
    total_params = sum(v.size for v in jax.tree.leaves(params))
    print(f"Total parameters: {total_params:,} ({total_params * 2 / 1e9:.2f} GB in bf16)")

    return params


def save_params_msgpack(params: dict, path: str):
    """Save JAX params as msgpack (for fast loading on TPU)."""
    from flax.serialization import to_bytes
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(to_bytes(params))
    print(f"Saved params to {path}")


def load_params_msgpack(path: str) -> dict:
    """Load JAX params from msgpack."""
    from flax.serialization import from_bytes
    with open(path, 'rb') as f:
        return from_bytes(None, f.read())


if __name__ == "__main__":
    import jax

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and len(sys.argv) > 1:
        hf_token = sys.argv[1]

    params = download_and_convert(hf_token=hf_token)

    # Verify structure
    print("\nTop-level keys:")
    for k in sorted(params.keys()):
        if isinstance(params[k], dict):
            print(f"  {k}: {list(params[k].keys())[:5]}...")
        else:
            print(f"  {k}: {params[k].shape}")
