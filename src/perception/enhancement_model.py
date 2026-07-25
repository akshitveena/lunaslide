"""Self-supervised lunar low-light enhancer inspired by IllumiCurveNet.

This is an independent implementation, not a reproduction of the IJCNN paper.
It learns per-pixel enhancement curves from unpaired low-light imagery using
exposure, spatial-consistency, and illumination-smoothness objectives.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.project(summary))


class DilatedContext(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class CurveEnhancer(nn.Module):
    """Estimates eight bounded curve maps and applies them recursively."""

    def __init__(self, curve_steps: int = 8, channels: int = 32) -> None:
        super().__init__()
        self.curve_steps = curve_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.context = DilatedContext(channels)
        self.attention = SpatialAttention()
        self.curves = nn.Conv2d(channels, curve_steps, 3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("CurveEnhancer expects BCHW grayscale input with one channel.")
        curve_maps = torch.tanh(self.curves(self.attention(self.context(self.encoder(image)))))
        enhanced = image
        for curve in curve_maps.unbind(dim=1):
            curve = curve.unsqueeze(1)
            enhanced = enhanced + curve * enhanced * (1.0 - enhanced)
        return enhanced.clamp(0.0, 1.0), curve_maps


class SelfGuidedEnhancementLoss(nn.Module):
    """No-reference objective for unpaired lunar low-light imagery.

    A faithful single-channel adaptation of the Zero-DCE losses, with the three
    calibration errors of the first version corrected:

    * **Exposure** only penalises *under*-exposure (``relu(target - mean)``), and
      pools with overlap.  The old symmetric ``|mean - target|`` dragged bright
      detail *down* to mid-grey and, being non-overlapping, stamped a 16-pixel
      grid into the output.
    * **Spatial consistency** uses the Zero-DCE region form — it preserves each
      4x4 region's contrast against its four neighbours — rather than merely
      matching raw gradient magnitudes, which had produced an embossed look.
    * **Curve smoothness** (total variation on the curve maps) is weighted to
      matter.  The old ``5.0`` left the maps ~40x too jagged versus the ~200
      scale the design needs, which is what actually caused the etching.

    Weights are constructor arguments so they can be tuned without code edits.
    """

    _NEIGHBOUR_KERNEL = None  # lazily built 4-neighbour difference filter

    def __init__(
        self,
        target_exposure: float = 0.6,
        *,
        w_exposure: float = 1.0,
        w_spatial: float = 1.0,
        w_smoothness: float = 200.0,
        exposure_patch: int = 16,
        region_patch: int = 4,
    ) -> None:
        super().__init__()
        self.target_exposure = target_exposure
        self.w_exposure = w_exposure
        self.w_spatial = w_spatial
        self.w_smoothness = w_smoothness
        self.exposure_patch = exposure_patch
        self.region_patch = region_patch

    @staticmethod
    def _gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]

    def _spatial_consistency(self, source: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """Preserve local contrast: region-to-neighbour differences must match.

        Averages each ``region_patch`` block, then compares how much a block
        differs from each of its four neighbours in the enhanced image versus
        the source.  Enhancement may change absolute brightness freely but must
        keep the *relationships* between neighbouring regions, which is what
        stops it from flattening or inventing structure.
        """
        pool = nn.functional.avg_pool2d
        s = pool(source, self.region_patch)
        e = pool(enhanced, self.region_patch)
        loss = enhanced.new_zeros(())
        for shift, axis in ((1, 2), (-1, 2), (1, 3), (-1, 3)):
            ds = s - torch.roll(s, shifts=shift, dims=axis)
            de = e - torch.roll(e, shifts=shift, dims=axis)
            loss = loss + (de - ds).pow(2).mean()
        return loss

    def forward(
        self, source: torch.Tensor, enhanced: torch.Tensor, curve_maps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Exposure: lift dark regions toward the target, never darken bright
        # ones. Overlapping pooling (stride = patch/2) removes the block grid.
        stride = max(1, self.exposure_patch // 2)
        local_mean = nn.functional.avg_pool2d(enhanced, self.exposure_patch, stride)
        exposure = torch.relu(self.target_exposure - local_mean).mean()

        spatial = self._spatial_consistency(source, enhanced)

        # Illumination smoothness: total variation on the curve maps, squared as
        # in Zero-DCE's L_tvA. Strong weight keeps the maps smooth so the applied
        # enhancement does not etch high-frequency artefacts into the surface.
        cx, cy = self._gradient(curve_maps)
        smoothness = cx.pow(2).mean() + cy.pow(2).mean()

        total = (
            self.w_exposure * exposure
            + self.w_spatial * spatial
            + self.w_smoothness * smoothness
        )
        return {"total": total, "exposure": exposure, "spatial": spatial, "smoothness": smoothness}
