"""Trainable glyph and right-page features for the frozen SDXL U-Net."""

from contextlib import contextmanager

import torch
from torch import nn


class BookConditionEncoder(nn.Module):
    """Map [B, 2, H, W] masks in [0, 1] to a residual at U-Net input resolution."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.projection = nn.Conv2d(64, out_channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim != 4 or masks.shape[1] != 2 or not masks.is_floating_point():
            raise ValueError("Expected floating point [batch, 2, height, width] glyph/page masks.")
        if masks.shape[-2] % 8 or masks.shape[-1] % 8:
            raise ValueError("Mask height and width must be divisible by eight.")
        return self.projection(self.features(masks))


@contextmanager
def condition_unet(unet: nn.Module, residual: torch.Tensor):
    """Apply a precomputed residual for one SDXL call and then restore its conv_in.

    If inference batches the conditional and unconditional CFG branches, the
    caller must explicitly prepare the corresponding residual for each branch.
    """
    def add_residual(_layer, _inputs, features):
        if residual.shape != features.shape:
            raise ValueError(f"Book condition {residual.shape} differs from UNet input features {features.shape}.")
        return features + residual.to(features)

    hook = unet.conv_in.register_forward_hook(add_residual)
    try:
        yield
    finally:
        hook.remove()
