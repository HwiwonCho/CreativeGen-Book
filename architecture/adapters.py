"""Attention LoRA and checkpoint loading for the book, illustration and background generators."""
import json
import math
from pathlib import Path

from .approved_base import MODEL_ID, REVISION


def attach_adapter(unet, rank):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class AdapterLinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.base = linear.requires_grad_(False)
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=linear.weight.device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=linear.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        @property
        def weight(self):
            return self.base.weight

        def forward(self, value):
            native = self.base(value)
            with torch.autocast(device_type=value.device.type, enabled=False):
                update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
            return native + update.to(native.dtype)

    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    unet.requires_grad_(False)
    targets = [(name, mod) for name, mod in unet.named_modules()
               if isinstance(mod, nn.Linear) and name.endswith((".attn1.to_q", ".attn1.to_v", ".attn2.to_q", ".attn2.to_v"))]
    if not targets:
        raise ValueError("Expected native UNet attention projections")
    for name, mod in targets:
        parent, key = name.rsplit(".", 1)
        setattr(unet.get_submodule(parent), key, AdapterLinear(mod))
    return {name: p for name, p in unet.named_parameters() if p.requires_grad}


def load_adapter(unet, run):
    """Attach trained weights to a fresh base UNet, for evaluation or inference."""
    import torch
    from safetensors.torch import load_file
    run = Path(run)
    settings = json.loads((run / "settings.json").read_text())
    if settings["base_id"] != MODEL_ID or settings["base_revision"] != REVISION:
        raise ValueError("Adapter base model differs")
    params = attach_adapter(unet, settings["rank"])
    weights = load_file(str(run / "adapter.safetensors"))
    if set(params) != set(weights):
        raise ValueError("Adapter parameter names differ")
    with torch.no_grad():
        for name, p in params.items():
            value = weights[name]
            if p.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
            p.copy_(value)
    return unet.eval()


def load_book_conditioner(unet, run):
    from safetensors.torch import load_file
    from .book_conditioner import BookConditionEncoder
    cond = BookConditionEncoder(unet.conv_in.out_channels)
    cond.load_state_dict(load_file(str(Path(run)/"conditioner.safetensors")), strict=True)
    return cond.to(device=unet.conv_in.weight.device).eval()
