"""Local print response fitted to scanned ink on paper."""
import torch
from torch import nn


class InkResponse(nn.Module):
    def __init__(self):
        super().__init__()
        self.neighbors = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.color = nn.Conv2d(9, 6, 1, bias=False)

    def forward(self, paper, coverage):
        features = torch.cat((coverage, self.neighbors(coverage).relu()), dim=1)
        slope, offset = self.color(features).chunk(2, dim=1)
        return (paper + paper * slope + offset).clamp(0, 1)


class ScanInkCurve(nn.Module):
    """Interpolate measured RGB reflectance curves using local paper chromaticity."""
    def __init__(self, curves, tones, bandwidth):
        super().__init__()
        self.register_buffer("curves",curves)
        self.register_buffer("tones",tones)
        self.register_buffer("bandwidth",bandwidth)

    def forward(self, paper, coverage):
        chroma = paper / paper.mean(1, keepdim=True).clamp_min(.01)
        position = coverage[:,0].clamp(0,1)*(self.curves.size(1)-1)
        low = position.long().clamp(max=self.curves.size(1)-2)
        fraction = (position-low)[:,None]
        dist = (chroma[:,None]-self.tones[None,:,:,None,None]).square().sum(2)
        mixing = (-dist/self.bandwidth.square()).softmax(1)
        reflectance = torch.zeros_like(paper * coverage)
        for i, curve in enumerate(self.curves):
            a = curve[low].permute(0,3,1,2)
            b = curve[low+1].permute(0,3,1,2)
            reflectance += mixing[:,i:i+1]*torch.lerp(a,b,fraction)
        return torch.where(coverage==0,paper,paper*reflectance)


def load_response(weights):
    if "curves" in weights:
        return ScanInkCurve(**weights)
    network = InkResponse()
    network.load_state_dict(weights)
    return network
