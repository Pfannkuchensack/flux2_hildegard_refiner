# FLUX.2 Klein Hildegard Refiner (InvokeAI custom nodes)

Port of the [42lux ComfyUI Hildegard-Refiner](https://github.com/42lux/ComfyUI-42lux-Hildegard-Refiner)
reference-latent scheme to InvokeAI, for tile-based refinement with FLUX.2 Klein.

## Nodes

### FLUX2 Ref Denoise (Hildegard) — `flux2_ref_denoise`

A fork of the stock `FLUX2 Denoise` node. Instead of the built-in
`kontext_conditioning` reference path (which clamps multi-reference images to
1024² and uses a fixed positioning scheme), this node exposes three explicit
reference slots that map 1:1 to the Hildegard three-latent scheme:

| Slot                 | Hildegard meaning                                   |
|----------------------|-----------------------------------------------------|
| **Tile Reference**   | VAE-encoded crop of the upscaled tile               |
| **Position Reference** | 3×3 neighbourhood / position map                  |
| **Global Reference** | Thumbnail of the whole upscaled image               |

Each slot has its own **weight** (resemblance ↔ creativity). References are
encoded at full resolution (no pixel clamp), packed, BN-normalized, and
concatenated as FLUX.2 reference tokens with `T = 10, 20, 30`.

The Hildegard **LoRA is mandatory** — load it onto the transformer (FLUX LoRA
loader) and include its trigger phrase in the prompt. Without it the model does
not understand these reference latents.

#### Typical wiring (single-tile de-risk experiment)

```
FLUX2 Main Model ─┐
FLUX LoRA (Hildegard) ─→ Transformer ─┐
Qwen3 prompt (RFNTILE. …) ────────────→ positive_text_conditioning
                                        FLUX2 Ref Denoise
  tile crop image  ──────────────────→ tile_reference
  global thumbnail ──────────────────→ global_reference
  FLUX2 VAE ─────────────────────────→ vae
                                        ↓ latents
                                        FLUX2 VAE Decode → Save
```

Leave `latents` unconnected for pure noise→refine (Hildegard's default), or feed
upscaled tile latents + `denoising_start` for an img2img-style pass.

## Status

Prototype / de-risking. The reference **positioning** mirrors Invoke's native
`Flux2RefImageExtension` (diffusers-aligned). If the LoRA misbehaves, the
positioning is the first knob to revisit.
