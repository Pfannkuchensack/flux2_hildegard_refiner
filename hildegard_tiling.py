"""Hildegard tiling nodes for InvokeAI (Plan + References Split).

These are the pixel-space helpers around ``flux2_ref_denoise``:

  * **Hildegard Plan** - upscales the source image by a scale factor and computes
    a tile grid (reusing Invoke's ``calc_tiles_min_overlap``). Outputs the
    upscaled image plus ``list[Tile]`` so it plugs straight into Invoke's
    Iterate -> ... -> Collect -> Merge Tiles to Image pattern.

  * **Hildegard References** - for one tile, produces the three reference images
    the refiner needs:
        - tile reference   : the tile crop (optional high-frequency reduction)
        - position reference: a 3x3 neighbourhood map (current tile centered,
          off-canvas cells gray, cyan marker around the center)
        - global reference  : a thumbnail of the whole upscaled image
    It also passes the ``Tile`` through so the refined result can be paired with
    Invoke's ``Pair Tile with Image`` and merged with ``Merge Tiles to Image``.

Combine is intentionally NOT a custom node - Invoke's stock "Merge Tiles to
Image" already does feathered seam/linear blending over ``list[TileWithImage]``.
"""

from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, Input, InputField, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import Tile


def _snap(value: int, multiple: int = 16) -> int:
    """Round ``value`` to the nearest multiple of ``multiple`` (>= multiple)."""
    return max(multiple, int(round(value / multiple)) * multiple)


# --------------------------------------------------------------------------- #
# Hildegard Plan                                                              #
# --------------------------------------------------------------------------- #
@invocation_output("hildegard_plan_output")
class HildegardPlanOutput(BaseInvocationOutput):
    upscaled_image: ImageField = OutputField(description="The source image upscaled to the planned grid size.")
    tiles: list[Tile] = OutputField(description="The tile grid covering the upscaled image (Iterate over this).")
    width: int = OutputField(description="Width of the upscaled image.")
    height: int = OutputField(description="Height of the upscaled image.")


@invocation(
    "hildegard_plan",
    title="Hildegard Plan",
    tags=["image", "tile", "upscale", "refiner", "hildegard", "flux2"],
    category="tiles",
    version="0.1.0",
    classification=Classification.Prototype,
)
class HildegardPlanInvocation(BaseInvocation):
    """Upscale the source image and compute a tile grid for the Hildegard refiner."""

    image: ImageField = InputField(description="The source image to refine.")
    tile_size: int = InputField(
        default=1024,
        ge=256,
        multiple_of=16,
        description="Tile size in pixels (square). The refiner generates each tile at this resolution.",
    )
    min_overlap: int = InputField(
        default=128,
        ge=0,
        description="Minimum overlap between adjacent tiles, in pixels.",
    )
    scale_factor: float = InputField(
        default=2.0,
        ge=1.0,
        le=8.0,
        description="How much to upscale the source before refining.",
    )

    def invoke(self, context: InvocationContext) -> HildegardPlanOutput:
        image = context.images.get_pil(self.image.image_name).convert("RGB")
        src_w, src_h = image.size

        # Target size: source * scale_factor, snapped to multiples of 16, at least one tile.
        target_w = max(self.tile_size, _snap(int(src_w * self.scale_factor)))
        target_h = max(self.tile_size, _snap(int(src_h * self.scale_factor)))

        upscaled = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        upscaled_dto = context.images.save(image=upscaled)

        # min_overlap must be < tile_size for calc_tiles_min_overlap.
        min_overlap = min(self.min_overlap, self.tile_size - 16)
        tiles = calc_tiles_min_overlap(
            image_height=target_h,
            image_width=target_w,
            tile_height=self.tile_size,
            tile_width=self.tile_size,
            min_overlap=min_overlap,
        )

        context.util.signal_progress(
            f"Planned {len(tiles)} tile(s) for {target_w}x{target_h} (from {src_w}x{src_h})"
        )

        return HildegardPlanOutput(
            upscaled_image=ImageField(image_name=upscaled_dto.image_name),
            tiles=tiles,
            width=target_w,
            height=target_h,
        )


# --------------------------------------------------------------------------- #
# Hildegard References (Split)                                                #
# --------------------------------------------------------------------------- #
@invocation_output("hildegard_references_output")
class HildegardReferencesOutput(BaseInvocationOutput):
    tile_reference: ImageField = OutputField(description="Tile crop (optionally high-frequency reduced).")
    position_reference: ImageField = OutputField(description="3x3 neighbourhood / position map.")
    global_reference: ImageField = OutputField(description="Thumbnail of the whole upscaled image.")
    tile: Tile = OutputField(description="The tile, passed through for pairing/merging.")


@invocation(
    "hildegard_references",
    title="Hildegard References (Split)",
    tags=["image", "tile", "reference", "refiner", "hildegard", "flux2"],
    category="tiles",
    version="0.1.0",
    classification=Classification.Prototype,
)
class HildegardReferencesInvocation(BaseInvocation):
    """Build the tile / position / global reference images for a single tile."""

    upscaled_image: ImageField = InputField(description="The upscaled image (from Hildegard Plan).")
    tile: Tile = InputField(description="The current tile (from Iterate over the Plan's tiles).", input=Input.Connection)

    cell_size: int = InputField(
        default=512,
        ge=64,
        multiple_of=16,
        description="Size of each cell in the 3x3 position map. Map is 3x this. Higher = more tokens.",
    )
    draw_marker: bool = InputField(default=True, description="Draw a cyan marker around the center cell of the map.")
    tile_high_freq_reduce: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reduce high-frequency detail in the tile reference (0 = keep all, 1 = only broad shapes).",
    )
    low_freq_radius: int = InputField(
        default=32,
        ge=1,
        le=256,
        description="Gaussian blur radius (px) separating low/high frequency for the tile reference.",
    )
    global_max_size: int = InputField(
        default=768,
        ge=64,
        description="Long-edge cap for the global thumbnail, in pixels.",
    )

    def _freq_reduce(self, img: Image.Image) -> Image.Image:
        if self.tile_high_freq_reduce <= 0.0:
            return img
        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.low_freq_radius))
        # Image.blend(a, b, alpha) = a*(1-alpha) + b*alpha
        return Image.blend(img, blurred, self.tile_high_freq_reduce)

    def _global_thumb(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(1.0, self.global_max_size / max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)

    def _position_map(self, up_img: Image.Image, tile: Tile) -> Image.Image:
        W, H = up_img.size
        tl, tt = tile.coords.left, tile.coords.top
        tw = tile.coords.right - tile.coords.left
        th = tile.coords.bottom - tile.coords.top
        gray = (128, 128, 128)
        cs = self.cell_size

        pos_map = Image.new("RGB", (cs * 3, cs * 3), gray)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                # Neighbour region in the upscaled image (contiguous offset by tile size).
                rl, rt = tl + dx * tw, tt + dy * th
                rr, rb = rl + tw, rt + th
                # Intersect with the canvas.
                il, it = max(rl, 0), max(rt, 0)
                ir, ib = min(rr, W), min(rb, H)

                cell = Image.new("RGB", (tw, th), gray)
                if ir > il and ib > it:
                    region = up_img.crop((il, it, ir, ib))
                    cell.paste(region, (il - rl, it - rt))
                cell = cell.resize((cs, cs), Image.Resampling.BICUBIC)
                pos_map.paste(cell, ((dx + 1) * cs, (dy + 1) * cs))

        if self.draw_marker:
            draw = ImageDraw.Draw(pos_map)
            draw.rectangle(
                [cs, cs, 2 * cs - 1, 2 * cs - 1],
                outline=(0, 255, 255),
                width=max(2, cs // 64),
            )
        return pos_map

    def _save(self, context: InvocationContext, img: Image.Image) -> ImageField:
        dto = context.images.save(image=img)
        return ImageField(image_name=dto.image_name)

    def invoke(self, context: InvocationContext) -> HildegardReferencesOutput:
        up_img = context.images.get_pil(self.upscaled_image.image_name).convert("RGB")

        # Tile reference: crop + optional frequency reduction.
        tile_crop = up_img.crop(
            (self.tile.coords.left, self.tile.coords.top, self.tile.coords.right, self.tile.coords.bottom)
        )
        tile_ref = self._freq_reduce(tile_crop)

        position_ref = self._position_map(up_img, self.tile)
        global_ref = self._global_thumb(up_img)

        return HildegardReferencesOutput(
            tile_reference=self._save(context, tile_ref),
            position_reference=self._save(context, position_ref),
            global_reference=self._save(context, global_ref),
            tile=self.tile,
        )
