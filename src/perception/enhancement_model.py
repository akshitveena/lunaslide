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
    """No-reference objective for unpaired lunar low-light imagery."""

    def __init__(self, target_exposure: float = 0.55) -> None:
        super().__init__()
        self.target_exposure = target_exposure

    @staticmethod
    def _gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]

    def forward(
        self, source: torch.Tensor, enhanced: torch.Tensor, curve_maps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Local exposure avoids a bright global mean hiding dark regions.
        local_mean = nn.functional.avg_pool2d(enhanced, kernel_size=16, stride=16)
        exposure = (local_mean - self.target_exposure).abs().mean()
        sx, sy = self._gradient(source)
        ex, ey = self._gradient(enhanced)
        spatial = (ex.abs() - sx.abs()).abs().mean() + (ey.abs() - sy.abs()).abs().mean()
        cx, cy = self._gradient(curve_maps)
        smoothness = cx.abs().mean() + cy.abs().mean()
        identity = (enhanced - source).abs().mean()
        total = 10.0 * exposure + spatial + 5.0 * smoothness + 0.05 * identity
        return {"total": total, "exposure": exposure, "spatial": spatial, "smoothness": smoothness}
