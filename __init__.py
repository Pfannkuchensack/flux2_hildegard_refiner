"""FLUX.2 Klein Hildegard Refiner node pack for InvokeAI.

Port of the 42lux ComfyUI Hildegard-Refiner reference-latent scheme.

Nodes:
  * Flux2RefDenoiseInvocation   - FLUX.2 denoise with explicit tile/position/global references
  * HildegardPlanInvocation     - upscale + tile grid
  * HildegardReferencesInvocation - per-tile tile/position/global reference images

Combine is handled by Invoke's stock "Merge Tiles to Image" node.
"""

from .flux2_ref_denoise import Flux2RefDenoiseInvocation
from .hildegard_tiling import HildegardPlanInvocation, HildegardReferencesInvocation

__all__ = [
    "Flux2RefDenoiseInvocation",
    "HildegardPlanInvocation",
    "HildegardReferencesInvocation",
]
