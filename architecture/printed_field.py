"""A learned grayscale ink field and optical character metrics for the SDXL prior."""
import torch
from torch import nn
from torch.nn import functional as F


def character_ids(text):
    if not text or not all(32 <= ord(c) < 127 for c in text):
        raise ValueError("Expected one printable English line")
    return [ord(c) - 32 for c in text]


class PrintedField(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(95, 64)
        self.strokes = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 32 * 24))
        self.metrics = nn.Sequential(nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 4))
        self.register_buffer("paper", torch.ones(3))
        self.register_buffer("ink", torch.zeros(3))

    def letters(self, ids):
        features = self.embedding(ids)
        logits = self.strokes(features).reshape(*ids.shape, 32, 24)
        return logits, self.optical_metrics(features)

    def optical_metrics(self, features):
        raw = self.metrics(features)
        advance, width, top, height = raw.unbind(-1)
        metrics = torch.stack((F.softplus(advance), F.softplus(width), top, F.softplus(height)), -1)
        return metrics

    def coverage(self, text, height, width):
        ids = torch.tensor(character_ids(text), device=self.paper.device)
        logits, metrics = self.letters(ids)
        advance, size, top, extent = metrics.unbind(1)
        em = height / 1.4
        margin = .2 * em
        start = advance.cumsum(0) - advance
        fit = ((width - 2 * margin) / ((start + size).max() * em)).clamp(max=1)
        x = (torch.arange(width, device=ids.device) + .5 - margin) / (em * fit)
        y = (torch.arange(height, device=ids.device) + .5 - margin) / em
        index = torch.searchsorted(start.contiguous(), x, right=True).sub(1).clamp(0, len(ids) - 1)
        u = (x - start[index]) / size[index].clamp_min(.015)
        v = (y[:, None] - top[index]) / extent[index].clamp_min(.015)
        patches = F.pad(logits.sigmoid() * (ids != 0)[:, None, None], (1, 1, 1, 1))
        count, ph, pw = patches.shape
        atlas = patches.permute(1, 0, 2).reshape(1, 1, ph, count * pw)
        gx = 2 * (index * pw + (1 + u * (pw - 2)).clamp(.5, pw - .5)) / (count * pw) - 1
        gy = 2 * (1 + v * (ph - 2)).clamp(.5, ph - .5) / ph - 1
        grid = torch.stack((gx.expand_as(gy), gy), -1)
        return F.grid_sample(atlas, grid[None], align_corners=False)

    def forward(self, text, height, width):
        coverage = self.coverage(text, height, width)
        colors = self.paper[None, :, None, None] + coverage * (self.ink - self.paper)[None, :, None, None]
        return 2 * colors - 1
