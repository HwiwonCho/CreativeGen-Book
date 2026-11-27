"""Fit a discriminative page locator on transformed public-domain training spreads."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from architecture.approved_base import file_digest
from architecture.page_locator import PageLocator, grid, homography, landmarks, project


def sources(root,kind,split):
    path=root/kind/("manifest_joint.jsonl" if kind=="book" else "manifest.jsonl")
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines()]
    picked=[r for r in rows if r["split"]==split]
    assert not ({r["group_id"] for r in picked} & {r["group_id"] for r in rows if r["split"]!=split})
    images=[]
    for row in picked:
        source=root/kind/row["image"]
        if file_digest(source)!=row["sha256"]:
            raise ValueError(f"Changed training image {source}")
        with Image.open(source) as im:
            images.append(np.array(im.convert("RGB").resize((192,128),Image.Resampling.LANCZOS)))
    return torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255


def augment(books,backgrounds,b,g):
    dev=books.device
    def rand(*shape):
        return torch.rand(shape,device=dev,generator=g)
    fg=books[torch.randint(len(books),(b,),generator=g,device=dev)]
    bg=backgrounds[torch.randint(len(backgrounds),(b,),generator=g,device=dev)]
    unit=fg.new_tensor([[0,0],[1,0],[0,1],[1,1]]).expand(b,-1,-1)
    shape=(unit-.5)*(rand(b,1,2)*.49+.48)
    angle=(rand(b,1)-.5)*.82
    co,si=angle.cos(),angle.sin()
    rot=torch.cat((co,si*1.5,-si/1.5,co),1).reshape(b,2,2)
    q=shape@rot+(rand(b,4,2)-.5)*.072
    half=q.abs().amax((1,2),keepdim=True)
    q=q*(.49/half).clamp(max=1)
    low,high=-q.amin(1,keepdim=True),1-q.amax(1,keepdim=True)
    q=q+low+(high-low)*rand(b,1,2)
    forward=homography(unit,q)
    original=grid(128,192,dev).reshape(1,-1,2).expand(b,-1,-1)
    local=project(torch.linalg.inv(forward),original).reshape(b,128,192,2)
    inside=((local>=0)&(local<=1)).all(-1).unsqueeze(1).float()
    page=F.grid_sample(fg,local*2-1,align_corners=False,padding_mode="border")
    # Lighting and crease strength vary independently of the projected corners.
    crease=torch.exp(-((local[...,0]-.5)/(.005+.024*rand(b,1,1))).square())[:,None]
    page=page*(.78+.27*rand(b,3,1,1))*(1-crease*.35*rand(b,1,1,1))
    bg=bg*(.45+.75*rand(b,1,1,1))+(.16*rand(b,3,1,1))
    shadow=F.avg_pool2d(inside,9,1,4)*.35*rand(b,1,1,1)
    img=(bg*(1-inside)*(1-shadow)+page*inside).clamp(0,1)
    marker=fg.new_tensor([[0,0],[.5,0],[1,0],[0,1],[.5,1],[1,1]]).expand(b,-1,-1)
    return img, F.avg_pool2d(inside,2), project(forward,marker)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1300)
    p.add_argument("--calibrate",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(11726)
    torch.set_float32_matmul_precision("high")
    train=[sources(a.data,k,"train") for k in ("book","background")]
    val=[sources(a.data,k,"validation") for k in ("book","background")]
    device=train[0].device
    rng=torch.Generator(device=device).manual_seed(11726)
    model=PageLocator().cuda()
    if a.calibrate:
        record=json.loads((a.output/"locator.json").read_text())
        calibrate(model,a.output,val,record)
        return
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    xy=grid(64,96,device)[None,None]
    hist=[]
    best=float("inf")
    for step in range(1,a.steps+1):
        image,mask,point=augment(*train,16,rng)
        logits=model(image)
        d=(xy-point[:,:,None,None])*point.new_tensor([96.,64.])
        target=(-.5*d.square().sum(-1)/1.3**2).flatten(2).softmax(-1)
        heat=-(target*logits[:,1:].flatten(2).log_softmax(-1)).sum(-1).mean()
        loss=heat+F.binary_cross_entropy_with_logits(logits[:,:1],mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step==a.steps*2//3:
            opt.param_groups[0]["lr"]*=.35
        if step%250==0 or step==a.steps:
            g=torch.Generator(device=device).manual_seed(9009)
            errs,ious=[],[]
            with torch.no_grad():
                for _ in range(8):
                    images,m,points=augment(*val,16,g)
                    pred=model(images)
                    errs.append((landmarks(pred)-points).norm(dim=-1).mean(1))
                    obj,truth=pred[:,:1]>0,m>.5
                    ious.append((obj&truth).sum((1,2,3))/(obj|truth).sum((1,2,3)))
            error=float(torch.cat(errs).mean())
            score=dict(step=step,corner_error=error,iou=float(torch.cat(ious).mean()))
            hist.append(score)
            print(score,flush=True)
            if error<best:
                best=error
                save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(a.output/"locator.safetensors"))
    anns=[json.loads(s) for s in (a.data/"book/print_lines_joint.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    margins=[]
    for r in anns:
        if r["split"]!="train" or not r["lines"]:
            continue
        w,h=r["image_size"]
        boxes=np.array([line["box"] for line in r["lines"]])
        margins.append([boxes[:,0].min()/(w/2)-1,boxes[:,2].max()/(w/2)-1,
                        np.median((boxes[:,1]+boxes[:,3])/(2*h))])
    record=dict(validation=hist,train_books=len(train[0]),validation_books=len(val[0]),train_backgrounds=len(train[1]),
                validation_backgrounds=len(val[1]),validation_compositions=128,seed=11726,
                text_region=np.median(margins,axis=0).tolist(),marker_sha256=file_digest(a.output/"locator.safetensors"))
    calibrate(model,a.output,val,record)


@torch.inference_mode()
def calibrate(model,out,val,record):
    from pipeline.page_scene import geometry
    model.load_state_dict(load_file(str(out/"locator.safetensors")))
    g=torch.Generator(device=val[0].device).manual_seed(9020)
    measurements=[]
    for _ in range(8):
        imgs,_,_=augment(*val,16,g)
        for sample in model.eval()(imgs).split(1):
            _,agreement,area=geometry(sample)
            measurements.append([agreement,area])
    agree,area=np.array(measurements).T
    record["bounds"]=dict(min_agreement=float(np.quantile(agree,.01)),min_area=float(np.quantile(area,.01)),
                          max_area=float(np.quantile(area,.99)))
    print("calibration",record["bounds"],flush=True)
    (out/"locator.json").write_text(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
