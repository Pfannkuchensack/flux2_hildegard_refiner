"""FLUX.2 Klein Reference Denoise Invocation (Hildegard Refiner).

A fork of the stock ``Flux2DenoiseInvocation`` that replaces the built-in
``Flux2RefImageExtension`` reference path with an explicit, controllable one:

  * Three named reference slots (tile / position / global), matching the
    42lux Hildegard-Refiner three-latent scheme.
  * Per-reference weight ("resemblance vs. creativity" knob).
  * No 1024x1024 multi-reference pixel clamp - the carefully sized Hildegard
    references (position map ~1536px, global thumbnail, tile crop) are encoded
    at full resolution.

The reference latents are concatenated and handed to the backend ``denoise()``
function via its ``img_cond_seq`` / ``img_cond_seq_ids`` parameters - exactly
the same injection point the stock node uses, so the rest of the FLUX.2 Klein
denoise loop (LoRA patching, BN normalization, scheduler) is byte-for-byte the
stock behaviour.

This node is for de-risking: does the Hildegard LoRA respond sensibly when we
feed it reference latents with our own positioning? Start here, then port the
Plan / Split / Combine pixel nodes around it.
"""

from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    ImageField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.latent_noise import validate_noise_tensor_shape
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.ref_image_extension import generate_img_ids_flux2_with_offset
from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids_flux2,
    get_noise_flux2,
    get_schedule_flux2,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_ref_denoise",
    title="FLUX2 Ref Denoise (Hildegard)",
    tags=["image", "flux", "flux2", "klein", "denoise", "refiner", "tile", "hildegard"],
    category="latents",
    version="0.1.0",
    classification=Classification.Prototype,
)
class Flux2RefDenoiseInvocation(BaseInvocation):
    """FLUX.2 Klein denoise with explicit reference latents (Hildegard Refiner).

    Like FLUX2 Denoise, but reference images are supplied via three named slots
    (tile / position / global) with per-reference weights and no pixel clamp.
    """

    # --- Standard FLUX.2 denoise inputs (identical to Flux2DenoiseInvocation) ---
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    positive_text_conditioning: FluxConditioningField = InputField(
        description=FieldDescriptions.positive_cond,
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[FluxConditioningField] = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    guidance: float = InputField(
        default=4.0,
        ge=0,
        le=20,
        description="Guidance strength. Inert for current FLUX.2 Klein variants; kept for compatibility.",
    )
    cfg_scale: float = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=4,
        description="Number of diffusion steps. Use 4 for distilled models, 28+ for base models.",
    )
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process.",
        ui_choice_labels=FLUX_SCHEDULER_LABELS,
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    vae: VAEField = InputField(
        description="FLUX.2 VAE model (required for BN statistics and reference encoding).",
        input=Input.Connection,
    )

    # --- Hildegard reference slots (replace kontext_conditioning) ---
    tile_reference: Optional[ImageField] = InputField(
        default=None,
        description="Tile reference image (VAE-encoded crop of the upscaled tile).",
        input=Input.Connection,
        title="Tile Reference",
    )
    position_reference: Optional[ImageField] = InputField(
        default=None,
        description="Position reference image (3x3 neighbourhood / position map).",
        input=Input.Connection,
        title="Position Reference",
    )
    global_reference: Optional[ImageField] = InputField(
        default=None,
        description="Global reference image (thumbnail of the whole upscaled image).",
        input=Input.Connection,
        title="Global Reference",
    )
    tile_weight: float = InputField(default=1.0, ge=0.0, le=4.0, description="Weight for the tile reference.")
    position_weight: float = InputField(default=1.0, ge=0.0, le=4.0, description="Weight for the position reference.")
    global_weight: float = InputField(default=1.0, ge=0.0, le=4.0, description="Weight for the global reference.")

    # ------------------------------------------------------------------ #
    # BN statistics helpers (copied verbatim from Flux2DenoiseInvocation) #
    # ------------------------------------------------------------------ #
    def _get_bn_stats(self, context: InvocationContext) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        with context.models.load(self.vae.vae).model_on_device() as (_, vae):
            vae.eval()
            bn_layer = None
            if hasattr(vae, "bn"):
                bn_layer = vae.bn
            elif hasattr(vae, "batch_norm"):
                bn_layer = vae.batch_norm
            elif hasattr(vae, "encoder") and hasattr(vae.encoder, "bn"):
                bn_layer = vae.encoder.bn

            if bn_layer is None:
                return None
            if bn_layer.running_mean is None or bn_layer.running_var is None:
                return None

            bn_mean = bn_layer.running_mean.clone()
            bn_var = bn_layer.running_var.clone()
            bn_eps = bn_layer.eps if hasattr(bn_layer, "eps") else 1e-4
            bn_std = torch.sqrt(bn_var + bn_eps)
        return bn_mean, bn_std

    def _bn_normalize(self, x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return (x - bn_mean) / bn_std

    def _bn_denormalize(self, x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return x * bn_std + bn_mean

    # ------------------------------------------------------------------ #
    # Reference latent construction (replaces Flux2RefImageExtension)    #
    # ------------------------------------------------------------------ #
    def _prepare_refs(
        self,
        context: InvocationContext,
        device: torch.device,
        dtype: torch.dtype,
        bn_mean: Optional[torch.Tensor],
        bn_std: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode the reference slots into packed, BN-normalized latents + 4D position IDs.

        Reference order is fixed: tile -> position -> global. Each provided slot
        gets T = 10, 20, 30 ... (matching diffusers Flux2Pipeline / Invoke's native
        scheme) and is spatially tiled onto a shared canvas to keep coordinates
        distinct. Unlike the stock extension there is NO max-pixel clamp.
        """
        slots: list[Tuple[ImageField, float]] = []
        if self.tile_reference is not None:
            slots.append((self.tile_reference, self.tile_weight))
        if self.position_reference is not None:
            slots.append((self.position_reference, self.position_weight))
        if self.global_reference is not None:
            slots.append((self.global_reference, self.global_weight))

        if not slots:
            return None, None

        all_latents: list[torch.Tensor] = []
        all_ids: list[torch.Tensor] = []
        canvas_h = 0
        canvas_w = 0

        to_tensor = tv_transforms.ToTensor()
        vae_info = context.models.load(self.vae.vae)
        with vae_info.model_on_device() as (_, vae):
            vae.eval()
            vae_dtype = next(iter(vae.parameters())).dtype

            for idx, (image_field, weight) in enumerate(slots):
                image = context.images.get_pil(image_field.image_name).convert("RGB")

                # Pixel -> [-1, 1] tensor (NO max-pixel clamp).
                image_tensor = to_tensor(image) * 2.0 - 1.0
                image_tensor = image_tensor.unsqueeze(0).to(
                    device=TorchDevice.choose_torch_device(), dtype=vae_dtype
                )

                latent_dist = vae.encode(image_tensor, return_dict=False)[0]
                if hasattr(latent_dist, "mode"):
                    lat = latent_dist.mode()  # deterministic encode
                elif hasattr(latent_dist, "sample"):
                    lat = latent_dist.sample()
                else:
                    lat = latent_dist

                batch_size, _, latent_height, latent_width = lat.shape

                # Pad to be compatible with patch_size=2.
                pad_h = (2 - latent_height % 2) % 2
                pad_w = (2 - latent_width % 2) % 2
                if pad_h > 0 or pad_w > 0:
                    lat = F.pad(lat, (0, pad_w, 0, pad_h), mode="circular")
                    _, _, latent_height, latent_width = lat.shape

                packed = pack_flux2(lat).to(device, dtype)  # (B, seq, 128)

                if bn_mean is not None and bn_std is not None:
                    packed = self._bn_normalize(packed, bn_mean, bn_std)

                # Per-reference weight: scale the (normalized) reference tokens to
                # modulate how strongly this reference influences attention.
                if weight != 1.0:
                    packed = packed * weight

                # Spatial canvas tiling (mirrors Flux2RefImageExtension so multiple
                # references occupy distinct H/W coordinates).
                h_offset = 0
                w_offset = 0
                if idx > 0:
                    potential_h_vertical = canvas_h + latent_height
                    potential_w_horizontal = canvas_w + latent_width
                    if potential_h_vertical > potential_w_horizontal:
                        w_offset = canvas_w
                        canvas_w = canvas_w + latent_width
                        canvas_h = max(canvas_h, latent_height)
                    else:
                        h_offset = canvas_h
                        canvas_h = canvas_h + latent_height
                        canvas_w = max(canvas_w, latent_width)
                else:
                    canvas_h = latent_height
                    canvas_w = latent_width

                # T = 10, 20, 30 ... separates reference images from the generated
                # image (which uses T = 0).
                t_offset = 10 + 10 * idx
                ids = generate_img_ids_flux2_with_offset(
                    latent_height=latent_height,
                    latent_width=latent_width,
                    batch_size=batch_size,
                    device=device,
                    idx_offset=t_offset,
                    h_offset=h_offset,
                    w_offset=w_offset,
                )

                all_latents.append(packed)
                all_ids.append(ids)
                context.util.signal_progress(f"Encoded reference {idx + 1}/{len(slots)}")

            TorchDevice.empty_cache()

        return torch.cat(all_latents, dim=1), torch.cat(all_ids, dim=1)

    # ------------------------------------------------------------------ #
    # Invoke entrypoint                                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        bn_stats = self._get_bn_stats(context)
        bn_mean, bn_std = bn_stats if bn_stats is not None else (None, None)

        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        should_ignore_noise = init_latents is not None and not self.add_noise and self.denoise_mask is None
        noise: Optional[torch.Tensor]
        if should_ignore_noise:
            noise = None
            b, _c, latent_h, latent_w = init_latents.shape
        else:
            noise = self._prepare_noise_tensor(context, inference_dtype, device)
            b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Positive conditioning (Qwen3 embeds stored in t5_embeds).
        pos_cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(pos_cond_data.conditionings) == 1
        pos_flux_conditioning = pos_cond_data.conditionings[0]
        assert isinstance(pos_flux_conditioning, FLUXConditioningInfo)
        pos_flux_conditioning = pos_flux_conditioning.to(dtype=inference_dtype, device=device)
        txt = pos_flux_conditioning.t5_embeds

        seq_len = txt.shape[1]
        txt_ids = torch.zeros(1, seq_len, 4, device=device, dtype=torch.long)
        txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=torch.long)

        neg_txt = None
        neg_txt_ids = None
        if self.negative_text_conditioning is not None:
            neg_cond_data = context.conditioning.load(self.negative_text_conditioning.conditioning_name)
            assert len(neg_cond_data.conditionings) == 1
            neg_flux_conditioning = neg_cond_data.conditionings[0]
            assert isinstance(neg_flux_conditioning, FLUXConditioningInfo)
            neg_flux_conditioning = neg_flux_conditioning.to(dtype=inference_dtype, device=device)
            neg_txt = neg_flux_conditioning.t5_embeds
            neg_seq_len = neg_txt.shape[1]
            neg_txt_ids = torch.zeros(1, neg_seq_len, 4, device=device, dtype=torch.long)
            neg_txt_ids[..., 3] = torch.arange(neg_seq_len, device=device, dtype=torch.long)

        transformer_config = context.models.get_config(self.transformer.transformer)
        assert transformer_config.base == BaseModelType.Flux2 and transformer_config.type == ModelType.Main

        image_seq_len = packed_h * packed_w
        timesteps = get_schedule_flux2(num_steps=self.num_steps, image_seq_len=image_seq_len)
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.num_steps)
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        if init_latents is not None:
            if self.add_noise:
                assert noise is not None
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            assert noise is not None
            x = noise

        if len(timesteps) <= 1:
            return x

        img_ids = generate_img_ids_flux2(h=latent_h, w=latent_w, batch_size=b, device=device)

        inpaint_mask = self._prep_inpaint_mask(context, x)

        init_latents_packed = pack_flux2(init_latents) if init_latents is not None else None
        inpaint_mask_packed = pack_flux2(inpaint_mask) if inpaint_mask is not None else None
        noise_packed = pack_flux2(noise) if noise is not None else None
        x = pack_flux2(x)

        if bn_mean is not None and bn_std is not None:
            if init_latents_packed is not None:
                init_latents_packed = self._bn_normalize(init_latents_packed, bn_mean, bn_std)
                if noise_packed is not None:
                    noise_packed = self._bn_normalize(noise_packed, bn_mean, bn_std)
            if init_latents is not None:
                x = self._bn_normalize(x, bn_mean, bn_std)

        assert packed_h * packed_w == x.shape[1]

        inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None
        if inpaint_mask_packed is not None:
            assert init_latents_packed is not None
            assert noise_packed is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents_packed,
                inpaint_mask=inpaint_mask_packed,
                noise=noise_packed,
            )

        num_steps = len(timesteps) - 1
        cfg_scale_list = [self.cfg_scale] * num_steps

        is_inpainting = self.denoise_mask is not None or self.denoising_start > 1e-5

        scheduler = None
        if self.scheduler in FLUX_SCHEDULER_MAP and not is_inpainting:
            scheduler_class = FLUX_SCHEDULER_MAP[self.scheduler]
            if self.scheduler == "heun":
                scheduler = scheduler_class(num_train_timesteps=1000, shift=3.0)
            else:
                scheduler = scheduler_class(
                    num_train_timesteps=1000,
                    shift=3.0,
                    use_dynamic_shifting=True,
                    base_shift=0.5,
                    max_shift=1.15,
                    base_image_seq_len=256,
                    max_image_seq_len=4096,
                    time_shift_type="exponential",
                )

        # Build reference latents with our own positioning (no clamp, per-ref weight).
        img_cond_seq, img_cond_seq_ids = self._prepare_refs(context, device, inference_dtype, bn_mean, bn_std)

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            config = transformer_config

            if config.format in [ModelFormat.Diffusers]:
                model_is_quantized = False
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                model_is_quantized = False

            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Match reference batch size to the generated-image batch size.
            if img_cond_seq is not None and img_cond_seq.shape[0] != x.shape[0]:
                img_cond_seq = img_cond_seq.repeat(x.shape[0], 1, 1)
                assert img_cond_seq_ids is not None
                img_cond_seq_ids = img_cond_seq_ids.repeat(x.shape[0], 1, 1)

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                guidance=self.guidance,
                cfg_scale=cfg_scale_list,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                scheduler=scheduler,
                mu=mu,
                inpaint_extension=inpaint_extension,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

        if bn_mean is not None and bn_std is not None:
            x = self._bn_denormalize(x, bn_mean, bn_std)

        x = unpack_flux2(x.float(), self.height, self.width)
        return x

    # ------------------------------------------------------------------ #
    # Helpers copied from Flux2DenoiseInvocation                          #
    # ------------------------------------------------------------------ #
    def _prepare_noise_tensor(
        self, context: InvocationContext, inference_dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.noise is not None:
            noise = context.tensors.load(self.noise.latents_name).to(device=device, dtype=inference_dtype)
            validate_noise_tensor_shape(noise, "FLUX.2", self.width, self.height)
            return noise
        return get_noise_flux2(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> Optional[torch.Tensor]:
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = 1.0 - mask
        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        return mask.expand_as(latents)

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            converted = convert_bfl_lora_patch_to_diffusers(lora_info.model)
            yield (converted, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.flux2_step_callback(state)

        return step_callback
