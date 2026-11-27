"""Fit local ink appearance with book-disjoint scan, validation and test splits."""
from argparse import ArgumentParser
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from architecture.approved_base import file_digest
from architecture.ink_response import InkResponse, load_response
from architecture.printed_field import PrintedField
from training.line_text import load_rows, page


def pairs(rows, root, templates, count, seed):
    rng = random.Random(seed)
    examples = [(r, char) for r in rows for word in r["words"] for char in word["chars"]]
    examples = rng.sample(examples, min(count, len(examples)))
    examples.sort(key=lambda example: example[0]["image"])
    coverage, scans, papers, masks = [], [], [], []
    for row, char in examples:
        x, y, right, bottom = char["box"]
        em = row["box"][3] - row["box"][1]
        scale = 34 / em
        w, h = round((right-x)*scale), round((bottom-y)*scale)
        if not (1 <= w <= 38 and 1 <= h <= 38):
            continue
        scan = page(root / row["image"])
        top, base = row["box"][1], row["box"][3]
        band = max(2, round(em*.15))
        # Estimate bare paper only from above and below the printed line.
        strips = [scan.crop((x, a, right, b)) for a, b in
                  ((max(0, top-band), top), (base, min(scan.height, base+band))) if a < b]
        if not strips:
            continue
        margin = np.concatenate([np.asarray(part).reshape(-1,3) for part in strips])
        paper = np.median(margin, axis=0).astype(np.uint8)
        shape = (40,40)
        crop = scan.crop((x,y,right,bottom)).resize((w,h),Image.Resampling.BILINEAR)
        glyph = templates[ord(char["c"])-32].resize((w,h),Image.Resampling.BILINEAR)
        d = np.zeros(shape, np.uint8)
        target = np.broadcast_to(paper, (*shape,3)).copy()
        inside = np.zeros(shape,np.uint8)
        box = np.s_[1:1+h,1:1+w]
        d[box], target[box], inside[box] = np.asarray(glyph), np.asarray(crop), 1
        coverage.append(d)
        scans.append(target)
        papers.append(paper)
        masks.append(inside)
    return (torch.from_numpy(np.stack(coverage))[:,None],
            torch.from_numpy(np.stack(scans)).permute(0,3,1,2),
            torch.from_numpy(np.stack(papers))[:,:,None,None],
            torch.from_numpy(np.stack(masks))[:,None])


def batch(data, ix, device):
    density, target, paper, mask = (t[ix].to(device) for t in data)
    return density.float()/255, target.float()/255, paper.float()/255, mask.float()


@torch.no_grad()
def evaluate(model, data, attenuation, device):
    totals = torch.zeros(2,3,device=device)
    counts = torch.zeros(3, device=device)
    for start in range(0, len(data[0]), 96):
        d, real, p, m = batch(data, slice(start,start+96),device)
        regions = torch.cat((m,m*(d>.65),m*(d>.1)*(d<=.65)),1)
        counts += regions.sum((0,2,3))
        for i, prediction in enumerate((p*(1-d*attenuation), model(p,d))):
            totals[i] += ((prediction-real).square().mean(1,keepdim=True)*regions).sum((0,2,3))
    values = (totals/counts).sqrt().cpu().tolist()
    return {key: dict(zip(("all","core","edge"),v)) for key,v in zip(("fixed","learned"),values)}


def evaluate_titles(args, splits, templates, attenuation):
    rows=[json.loads(line) for line in args.titles.read_text().splitlines()]
    books=sorted({r["book"] for r in rows})
    if set(books) & {r["book"] for group in splits.values() for r in group}:
        raise ValueError("Title evaluation books overlap the original corpus")
    root=args.titles.parent
    for image,digest in {(r["image"],r["sha256"]) for r in rows}:
        if file_digest(root/image)!=digest:
            raise ValueError(f"Changed title scan: {image}")
    models={k:load_response(load_file(str(args.output/(k+".safetensors")))).to(args.device).eval()
            for k in ("ink-response","ink-curve")}
    report=[]
    for book in books:
        data=pairs([r for r in rows if r["book"]==book],root,templates,1600,271134)
        scores={}
        for name,model in models.items():
            result=evaluate(model,data,attenuation,args.device)
            scores["fixed"],scores[name]=result["fixed"],result["learned"]
        report.append(dict(book=book,glyphs=len(data[0]),scores=scores))
    result=dict(books=report,macro_rmse={k:np.mean([r["scores"][k]["all"] for r in report]) for k in scores})
    (args.output/"ink-title-evaluation.json").write_text(json.dumps(result,indent=2))
    print(result["macro_rmse"])


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data","field","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--steps",type=int,default=600)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--titles",type=Path,help="Evaluate existing checkpoints on annotated external title pages")
    args = parser.parse_args()
    torch.manual_seed(271109)
    torch.set_num_threads(4)
    field = PrintedField().eval()
    field.load_state_dict(load_file(str(args.field)))
    with torch.no_grad():
        pred = field.letters(torch.arange(95))[0].sigmoid()
        templates = [Image.fromarray((g*255).round().byte().numpy()) for g in pred]
    splits = load_rows(args)
    device = args.device
    attenuation = (1-field.ink/field.paper.clamp_min(.1)).clamp(0,1)[None,:,None,None].to(device)
    if args.titles:
        evaluate_titles(args,splits,templates,attenuation)
        return
    datasets = {k:pairs(splits[k],args.data,templates,n,271110+i)
                for i,(k,n) in enumerate((("train",6500),("validation",1200),("test",1600)))}
    print("crops", {k:len(v[0]) for k,v in datasets.items()}, flush=True)
    model = InkResponse().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.006)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint = args.output/"ink-response.safetensors"
    best = float("inf")
    history = []
    for step in range(1,args.steps+1):
        ix = torch.randint(len(datasets["train"][0]),(48,))
        d, real, p, m = batch(datasets["train"],ix,device)
        light = torch.empty(len(d),1,1,1,device=device).uniform_(.68,1.)
        err = model(light*p,d)-light*real
        loss = (err.square()*m).sum()/(3*m.sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            val = evaluate(model,datasets["validation"],attenuation,device)
            history.append(dict(step=step,validation=val))
            if val["learned"]["all"] < best:
                best, best_step = val["learned"]["all"], step
                save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
            print("validation", step, val, flush=True)
    model.load_state_dict(load_file(str(checkpoint)))
    result = dict(step=best_step,validation_history=history,test=evaluate(model,datasets["test"],attenuation,device),
                  books={k:sorted({r["book"] for r in v}) for k,v in splits.items()},
                  crops={k:len(v[0]) for k,v in datasets.items()},field_sha256=file_digest(args.field),
                  lines_sha256=file_digest(args.data/"lines.jsonl"),weights_sha256=file_digest(checkpoint))
    (args.output/"ink-response.json").write_text(json.dumps(result,indent=2))
    print("held out",result["test"],flush=True)


if __name__=="__main__":
    main()
