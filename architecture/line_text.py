"""A transcript-conditioned denoising adapter for a single, photographed line of type."""
import math
import torch
from torch import nn
from torch.nn import functional as F


PAD, START, END = 0, 1, 2


def tokenize(texts, device):
    rows = []
    for text in texts:
        if not text or any(not 32 <= ord(c) <= 126 for c in text):
            raise ValueError("The line adapter takes one line of printable ASCII.")
        rows.append([START, *(ord(c)-29 for c in text), END])
    length = max(map(len, rows))
    return torch.tensor([r+[PAD]*(length-len(r)) for r in rows],device=device)


class LineText(nn.Module):
    def __init__(self):
        super().__init__()
        self.characters = nn.Embedding(98,48,padding_idx=PAD)
        self.neighbours = nn.Sequential(nn.Conv1d(48,48,3,padding=1),nn.SiLU(),nn.Conv1d(48,48,3,padding=1))
        self.advance = nn.Linear(48,1)
        self.offset = nn.Linear(48,1)
        self.features = nn.Sequential(nn.Conv2d(66,64,1),nn.SiLU(),nn.Conv2d(64,48,1),nn.SiLU(),
                                      nn.Conv2d(48,8,1))
        self.coverage = nn.Conv2d(8,1,1)
        self.prior = nn.Sequential(nn.Conv2d(512,96,3,padding=1),nn.SiLU(),nn.Conv2d(96,96,3,padding=1),
                                   nn.SiLU(),nn.Conv2d(96,4,3,padding=1))
        self.residual = nn.Sequential(nn.Conv2d(9,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,4,3,padding=1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_uncertainty = nn.Parameter(torch.tensor(-1.2))

    def positions(self, ids):
        letters = self.characters(ids)
        context = letters + self.neighbours(letters.transpose(1,2)).transpose(1,2)
        advance = (F.softplus(self.advance(context).squeeze(-1))+.01)*(ids!=PAD)
        widths = advance/advance.sum(1,keepdim=True)
        centres = widths.cumsum(1)-.5*widths + .30*widths*self.offset(context).squeeze(-1).tanh()
        return context, centres, widths

    def condition(self, ids, height, width, aligned=None):
        context, centres, widths = self.positions(ids)
        reference=centres
        if aligned is not None:
            reference=torch.where(aligned>=0,aligned,centres)
        x=(torch.arange(width,device=ids.device)+.5)/width
        rel=(x[None,:,None]-reference[:,None,:])/widths[:,None,:].clamp_min(1e-5)
        weight=(-14*rel.square()).masked_fill(ids[:,None,:]==PAD,-1e9).softmax(-1)
        letters=(weight@context).transpose(1,2)[:,:,None,:].expand(-1,-1,height,-1)
        dx=(weight*rel.clamp(-2,2)).sum(-1)[:,None,None,:].expand(-1,1,height,-1)
        dy=((torch.arange(height,device=ids.device)+.5)/height-.5)[None,None,:,None].expand_as(dx)
        coords=[dx,dy]
        for f in (1,2,4,8):
            for v in (dx,dy):
                coords.extend(((2*math.pi*f*v).sin(),(2*math.pi*f*v).cos()))
        features=self.features(torch.cat([letters,*coords],1))
        return self.prior(F.pixel_unshuffle(features,8)), self.coverage(features), centres

    def denoise(self, base_clean, prior, sigma):
        s=torch.as_tensor(sigma,device=prior.device).reshape(-1,1,1,1)
        relative=s.square()/(s.square()+self.log_uncertainty.exp().square())
        update=self.residual(torch.cat((prior,base_clean,s.log1p().expand_as(prior[:,:1])),1))
        return torch.lerp(base_clean,prior+update,relative)
