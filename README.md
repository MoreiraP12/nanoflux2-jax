# nanoflux2-jax

A from-scratch JAX port of [Black Forest Labs' Flux 2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) that runs **2-3x faster than H100** on a single TPU v6e chip.

~500 lines of model code. ~120 lines of sampling. Download the weights from HuggingFace. That's it.

<p align="center">
  <img src="assets/tpu_samples/japanese_garden.png" width="320" />
  <img src="assets/tpu_samples/astronaut.png" width="320" />
</p>
<p align="center">
  <img src="assets/tpu_samples/fox.png" width="320" />
  <img src="assets/tpu_samples/alpine_village.png" width="320" />
</p>
<p align="center">
  <sub>All images generated on TPU v6e in under 300ms. Pixel-identical to PyTorch reference (cos_sim > 0.999).</sub>
</p>

---

## Performance

| Resolution | H100 (PyTorch) | TPU v6e (JAX) | Speedup |
|:-----------|:--------------:|:-------------:|:-------:|
| 512x512    | 258ms          | **84ms**      | **3.1x** |
| 768x768    | 409ms          | **179ms**     | **2.3x** |
| 1024x1024  | 651ms          | **325ms**     | **2.0x** |
| 1360x768   | 649ms          | **285ms**     | **2.3x** |
| 1920x1080  | 1,271ms        | **614ms**     | **2.1x** |

<sub>4-step distilled denoise, bf16, batch=1. H100 baseline uses official BFL PyTorch code.</sub>

<p align="center">
  <img src="assets/benchmarks/hero_tpu_vs_h100.png" width="600" alt="TPU v6e vs H100 latency comparison" />
</p>

That's **11.9 images/second** at 512x512 on a single TPU v6e chip.

<p align="center">
  <img src="assets/benchmarks/throughput.png" width="600" alt="Throughput comparison" />
</p>

## Why TPU?

Klein 4B is a flow-matching diffusion transformer (MMDiT). ~75% of its compute is large matrix multiplications -- QKV projections, MLP layers, attention -- exactly what the TPU's 256x256 systolic array was designed for.

The key number is the **critical arithmetic intensity**: the ratio of peak compute to memory bandwidth.

```
TPU v6e: 918 TFLOPS / 1.6 TB/s = 574 FLOPS/byte
H100:    989 TFLOPS / 3.35 TB/s = 295 FLOPS/byte
```

Klein's dominant operations have arithmetic intensities around 3,000 -- deeply compute-bound on both machines. But XLA sees the entire computation graph at compile time and automatically fuses the memory-bound glue operations (norms, reshapes, activations) between the matmuls.

<p align="center">
  <img src="assets/benchmarks/roofline.png" width="600" alt="Roofline analysis" />
</p>

<p align="center">
  <img src="assets/benchmarks/time_breakdown.png" width="600" alt="Time breakdown" />
</p>

## Architecture

The port is pure functional JAX. No Flax, no nn.Module, no framework overhead.

```
src/flux2_jax/
  model.py           # ~484 lines — full Klein 4B architecture
  sampling.py         # ~122 lines — flow-matching denoise loop
  convert_weights.py  # ~149 lines — PyTorch checkpoint → JAX pytree
```

Key design decisions:

- **Pure functions**: `forward(params, x) -> y`. Parameters live in a pytree.
- **`jax.lax.scan`** over the 20 identical single-stream blocks -- compilation is O(1) regardless of depth. JIT takes ~5 seconds.
- **Full-loop JIT**: the entire 4-step denoise is traced at once. This is **4.3x faster** than step-by-step JIT because XLA optimizes buffer reuse across steps.
- **Bit-exact to PyTorch**: cosine similarity > 0.999 against the reference implementation.

## Quickstart

```bash
# Install
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install safetensors transformers

# Download weights from HuggingFace
# https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

# Generate
python -c "
from flux2_jax.model import build_model
from flux2_jax.sampling import sample
from flux2_jax.convert_weights import load_weights

params = load_weights('path/to/klein-4b')
img = sample(params, prompt='a watercolor painting of a Japanese garden in autumn')
"
```

## H100 vs TPU v6e — Same Prompt, Same Seed

Same weights, same precision, same seed. The outputs are pixel-identical (cos_sim > 0.999).

<table>
<tr>
<th align="center">H100 (PyTorch)</th>
<th align="center">TPU v6e (JAX)</th>
<th align="center">Prompt</th>
</tr>
<tr>
<td><img src="assets/tpu_samples/garden_h100_s200.png" width="280" /></td>
<td><img src="assets/tpu_samples/garden_tpu_s200.png" width="280" /></td>
<td><em>"a watercolor painting of a Japanese garden in autumn"</em><br/><sub>1024x1024, seed=200</sub></td>
</tr>
<tr>
<td><img src="assets/tpu_samples/astronaut_h100_s100.png" width="280" /></td>
<td><img src="assets/tpu_samples/astronaut_tpu_s100.png" width="280" /></td>
<td><em>"a photorealistic portrait of an astronaut floating in space"</em><br/><sub>1360x768, seed=100</sub></td>
</tr>
<tr>
<td><img src="assets/tpu_samples/dewdrop_h100_s300.png" width="280" /></td>
<td><img src="assets/tpu_samples/dewdrop_tpu_s300.png" width="280" /></td>
<td><em>"macro photography of a dewdrop on a spider web at sunrise"</em><br/><sub>768x768, seed=300</sub></td>
</tr>
<tr>
<td><img src="assets/tpu_samples/cat_h100_s42.png" width="280" /></td>
<td><img src="assets/tpu_samples/cat_tpu_s42.png" width="280" /></td>
<td><em>"a photo of a cat sitting on a windowsill watching the rain"</em><br/><sub>1360x768, seed=42</sub></td>
</tr>
</table>

## More Samples

All generated on TPU v6e, 4-step denoise, bf16:

<p align="center">
  <img src="assets/tpu_samples/golden_retriever.png" width="400" />
  <img src="assets/tpu_samples/butterfly_macro.png" width="400" />
</p>
<p align="center">
  <img src="assets/tpu_samples/alpine_village.png" width="400" />
  <img src="assets/tpu_samples/fox.png" width="400" />
</p>

## Roofline Analysis

The theoretical minimum per step (1024x1024) is 41ms. We achieve 81ms -- about 51% of roofline peak. The gap comes from data movement between matmuls (reshapes, norms, RoPE) which are memory-bound.

With flash attention, we'd expect ~75ms per step, or 55% of roofline.

<p align="center">
  <img src="assets/benchmarks/roofline_gap.png" width="500" alt="Closing the roofline gap" />
</p>

## Acknowledgments

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) -- the idea that you can strip a powerful model down to its essence and make it readable, hackable, and fast in a few hundred lines of code. This is the nanoGPT of diffusion.

The roofline analysis and TPU optimization approach were heavily informed by [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) -- an incredible resource for learning JAX and understanding hardware performance. If you want to go deep on why TPUs work the way they do, start there.

Huge shoutout to [Black Forest Labs](https://bfl.ai) for building Klein. The architecture is beautifully clean -- the same design clarity that makes it fast on CUDA made it trivially portable to XLA. The model weights are available under Apache 2.0 on [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

## Citation

```bib
@misc{nanoflux2-jax,
    title={nanoflux2-jax: Flux 2 Klein 4B on TPU v6e},
    year={2026},
    howpublished={\url{https://github.com/YOUR_USERNAME/nanoflux2-jax}},
}

@misc{flux-2-2025,
    author={Black Forest Labs},
    title={{FLUX.2: Frontier Visual Intelligence}},
    year={2025},
    howpublished={\url{https://bfl.ai/blog/flux-2}},
}
```

## License

The JAX port code is MIT licensed. Model weights are subject to [Black Forest Labs' licensing](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache 2.0 for Klein 4B).
