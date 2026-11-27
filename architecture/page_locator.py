"""Locate the paper outline and the corners of two facing leaves in a generated photograph."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(c, k):
    return nn.Sequential(nn.Conv2d(c,k,3,padding=1), nn.GroupNorm(4,k), nn.SiLU(),
                         nn.Conv2d(k,k,3,padding=1), nn.GroupNorm(4,k), nn.SiLU())


class PageLocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c, self.d = conv(3,12), conv(12,24), conv(24,36), conv(36,48)
        self.neck=conv(48,48)
        self.up_c, self.up_b = conv(84,32),conv(56,24)
        self.output=nn.Conv2d(24,7,1)

    def forward(self,x):
        x=F.interpolate(x,(128,192),mode="bilinear",align_corners=False)
        a=self.a(x)
        b=self.b(F.avg_pool2d(a,2))
        c=self.c(F.avg_pool2d(b,2))
        d=self.d(F.avg_pool2d(c,2))
        d=d+F.interpolate(self.neck(F.avg_pool2d(d,2)),size=d.shape[-2:],mode="bilinear",align_corners=False)
        c=self.up_c(torch.cat((c,F.interpolate(d,size=c.shape[-2:],mode="bilinear",align_corners=False)),1))
        return self.output(self.up_b(torch.cat((b,F.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=False)),1)))


def grid(h,w,device):
    y,x=torch.meshgrid((torch.arange(h,device=device)+.5)/h,(torch.arange(w,device=device)+.5)/w,indexing="ij")
    return torch.stack((x,y),-1)


def homography(src,dst):
    """Solve the four correspondences in page order TL, TR, BL, BR."""
    x,y=src.unbind(-1)
    u,v=dst.unbind(-1)
    zeros,ones=torch.zeros_like(x),torch.ones_like(x)
    even=torch.stack((x,y,ones,zeros,zeros,zeros,-u*x,-u*y),-1)
    odd=torch.stack((zeros,zeros,zeros,x,y,ones,-v*x,-v*y),-1)
    a=torch.stack((even,odd),-2).flatten(-3,-2)
    b=torch.stack((u,v),-1).flatten(-2)
    coeff=torch.linalg.solve(a,b.unsqueeze(-1)).squeeze(-1)
    return torch.cat((coeff,ones[...,:1]),-1).reshape(*coeff.shape[:-1],3,3)


def project(matrix,points):
    xy=F.pad(points,(0,1),value=1) @ matrix.transpose(-1,-2)
    return xy[...,:2]/xy[...,2:].clamp_min(1e-6)


def landmarks(logits):
    b,c,h,w=logits.shape
    values=logits[:,1:].flatten(2)
    index=values.argmax(-1)
    xy=grid(h,w,logits.device)
    peak=xy.flatten(0,1)[index]
    local=(xy[None,None]-peak[:,:,None,None]).abs()
    window=(local[...,0]<2.5/w)&(local[...,1]<2.5/h)
    scores=logits[:,1:].masked_fill(~window,-1e4).flatten(2).softmax(-1)
    return scores @ xy.reshape(h*w,2)
