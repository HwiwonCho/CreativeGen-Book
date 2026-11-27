"""Estimate optical response per source, selecting paper conditioning across training books."""
from argparse import ArgumentParser
from itertools import groupby
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

from architecture.approved_base import file_digest
from architecture.ink_response import ScanInkCurve
from architecture.printed_field import PrintedField
from training.ink_response import evaluate, pairs
from training.line_text import load_rows


def fit_curve(data):
    d, target, paper, mask = data
    ref = paper.squeeze(-1).squeeze(-1).float().div(255)
    tone = (ref/ref.mean(1,keepdim=True).clamp_min(.01)).median(0).values
    # A least-squares RGB reflectance curve per book, anchored at bare paper.
    knots = 9
    mask = mask[:,0].bool().numpy()
    index = d[:,0].numpy()[mask].astype(np.float64)*(knots-1)/255
    ix = index.astype(int).clip(max=knots-2)
    f = index-ix
    observed = target.permute(0,2,3,1).numpy()[mask]/255
    white = np.broadcast_to(ref.numpy()[:,None,None,:],(*mask.shape,3))[mask]
    coeffs = []
    for channel in range(3):
        p = white[:,channel]
        a,b = p*(1-f),p*f
        m = np.diag(np.bincount(ix,a*a,minlength=knots)+np.bincount(ix+1,b*b,minlength=knots))
        off = np.diag(np.bincount(ix,a*b,minlength=knots-1),k=1)
        m += off+off.T
        vec = np.bincount(ix,a*observed[:,channel],minlength=knots)+np.bincount(ix+1,b*observed[:,channel],minlength=knots)
        # Minimum-curvature fit stabilizes the rarely observed opaque endpoint.
        second = np.diff(np.eye(knots),n=2,axis=0)
        m += len(ix)*.0002*second.T@second
        solution = np.linalg.solve(m[1:,1:],vec[1:]-m[1:,0])
        coeffs.append(np.minimum.accumulate(np.r_[1,solution].clip(0,1)))
    return torch.tensor(np.stack(coeffs,-1),dtype=torch.float32),tone


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("field","data","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(3)
    field=PrintedField().eval()
    field.load_state_dict(load_file(str(args.field)))
    with torch.no_grad():
        shapes=field.letters(torch.arange(95))[0].sigmoid()
        templates=[Image.fromarray((t*255).round().byte().numpy()) for t in shapes]
    attenuation=(1-field.ink/field.paper.clamp_min(.1)).clamp(0,1)[None,:,None,None]
    splits=load_rows(args)
    groups={}
    for name,rows in groupby(sorted(splits["train"],key=lambda r:r["book"]),key=lambda r:r["book"]):
        groups[name]=pairs(list(rows),args.data,templates,850,271122)
    fits={k:fit_curve(v) for k,v in groups.items()}
    def model(names, bandwidth):
        curves,tones=zip(*(fits[n] for n in names))
        return ScanInkCurve(torch.stack(curves),torch.stack(tones),torch.tensor(bandwidth))
    scores={}
    for width in (.035,.12,10.):
        per_book=[]
        for book,data in groups.items():
            held_out=tuple(k for k in groups if k!=book)
            subset=tuple(t[::5] for t in data)
            per_book.append(evaluate(model(held_out,width),subset,attenuation,"cpu")["learned"]["all"])
        scores[width]=float(np.mean(per_book))
    bandwidth=min(scores,key=scores.get)
    trained=model(tuple(groups),bandwidth)
    args.output.mkdir(exist_ok=True,parents=True)
    checkpoint=args.output/"ink-curve.safetensors"
    save_file(trained.state_dict(),str(checkpoint))
    results={k:evaluate(trained,pairs(splits[k],args.data,templates,n,seed),attenuation,"cpu")
             for k,n,seed in (("validation",1200,271111),("test",1600,271112))}
    report=dict(books=list(groups),crops={k:len(v[0]) for k,v in groups.items()},
                bandwidth=bandwidth,leave_one_book_out=scores,scores=results,
                field_sha256=file_digest(args.field),lines_sha256=file_digest(args.data/"lines.jsonl"),
                weights_sha256=file_digest(checkpoint))
    (args.output/"ink-curve.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
