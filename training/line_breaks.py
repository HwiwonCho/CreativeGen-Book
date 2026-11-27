"""Fit a conditional line-break score and test it on other books."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json
import math
import numpy as np
import torch
from safetensors.torch import load_file

from architecture.approved_base import file_digest
from architecture.printed_field import PrintedField, character_ids


def word_widths(field,words):
    ids=torch.arange(95,device=field.paper.device)
    metrics=field.optical_metrics(field.embedding(ids)).detach().cpu().tolist()
    widths=[]
    ends=[]
    for word in words:
        tokens=character_ids(word)
        widths.append(sum(metrics[t][0] for t in tokens))
        last=metrics[tokens[-1]]
        ends.append(last[1]-last[0])
    return widths,ends,metrics[0][0]


def features(used,n,limit,word,last):
    fraction=used/limit
    return [fraction,fraction**2,float(last)*fraction,float(last)*fraction**2,
            math.log1p(n),float(n==1),float(word[-1:] in (",",";",":",".","!","?")),1.]


def boundaries(words,widths,ends,space,limit,weights):
    n=len(words)
    scores=[0.]+[-math.inf]*n
    previous=[None]*(n+1)
    for stop in range(1,n+1):
        used=ends[stop-1]
        for start in range(stop-1,-1,-1):
            used+=widths[start]
            if used>limit:
                break
            f=features(used,stop-start,limit,words[stop-1],stop==n)
            score=scores[start]+sum(v*w for v,w in zip(f,weights))
            if score>scores[stop]:
                previous[stop],scores[stop]=start,score
            used+=space
    if previous[-1] is None:
        return []
    indexes=[]
    stop=n
    while stop:
        indexes.append(stop)
        stop=previous[stop]
    return list(reversed(indexes))


def examples(rows,field):
    pages=defaultdict(list)
    for row in rows:
        pages[row["split"],row["image"]].append(row)
    result=defaultdict(list)
    for (split,_),page in pages.items():
        page=sorted(page,key=lambda r:r["box"][1])
        if len(page)<5:
            continue
        data=[]
        for row in page:
            words=row["text"].split()
            w,e,s=word_widths(field,words)
            data.append((words,w,e,s))
        widths=[sum(w)+e[-1]+s*(len(w)-1) for _,w,e,s in data]
        for i,(left,right) in enumerate(zip(page,page[1:])):
            height=max(left["box"][3]-left["box"][1],right["box"][3]-right["box"][1])
            gap=right["box"][1]-left["box"][1]
            if not .65*height<gap<1.9*height:
                continue
            limit=float(np.quantile(widths[:i]+widths[i+2:],.94))
            wa,a,ae,space=data[i]
            wb,b,be,_=data[i+1]
            words,w,e=wa+wb,a+b,ae+be
            choices=[]
            for cut in range(1,len(w)):
                first=sum(w[:cut])+space*(cut-1)+e[cut-1]
                second=sum(w[cut:])+space*(len(w)-cut-1)+e[-1]
                if max(first,second)>limit:
                    continue
                f=np.array(features(first,cut,limit,words[cut-1],False))+features(second,len(w)-cut,limit,words[-1],True)
                choices.append((cut,f,abs(first-second)))
            if len(choices)<2 or not any(a[0]==len(wa) for a in choices):
                continue
            result[split].append((choices,len(wa),(words,w,e,space,limit)))
    return result


def main():
    p=ArgumentParser()
    p.add_argument("rows",type=Path); p.add_argument("weights",type=Path); p.add_argument("output",type=Path)
    a=p.parse_args()
    field=PrintedField().eval()
    field.load_state_dict(load_file(str(a.weights)))
    sets=examples([json.loads(s) for s in a.rows.read_text().splitlines()],field)
    samples=sets["train"]
    coef=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([coef],max_iter=80,line_search_fn="strong_wolfe")
    feats=[torch.from_numpy(np.stack([f for _,f,_ in r])) for r,_,_ in samples]
    targets=[next(i for i,t in enumerate(r) if t[0]==true) for r,true,_ in samples]
    centred=torch.cat([mat-mat.mean(0) for mat in feats],0)
    scale=centred.square().mean(0).sqrt()
    scale=torch.where(scale<1e-5,torch.ones_like(scale),scale)
    feats=[(mat-mat.mean(0))/scale for mat in feats]
    def loss():
        opt.zero_grad()
        penalties=[z.logsumexp(0)-z[t] for mat,t in zip(feats,targets) for z in [mat@coef]]
        value=torch.stack(penalties).mean()+.003*coef.square().sum()
        value.backward()
        return value
    opt.step(loss)
    weights=(coef.detach()/scale).tolist()
    # Pairwise ranking identifies relative word-boundary scores. Calibrate the line-count
    # intercept on training passages alone so decoding can also choose the number of lines.
    def accuracy(rows,weights):
        return sum(boundaries(*args,weights)==[cut,len(args[0])] for _,cut,args in rows)
    candidates=[weights[:-1]+[float(b)] for b in range(-18,1)]
    weights=max(candidates,key=lambda w:accuracy(samples,w))
    report={}
    for key,rows in sets.items():
        report[key]=dict(pairs=len(rows),learned=accuracy(rows,weights),
                         balanced=accuracy(rows,[2,-1,0,0,0,0,0,-1000]))
    result=dict(weights=weights,metrics=report,manifest_sha256=file_digest(a.rows),field_sha256=file_digest(a.weights))
    a.output.write_text(json.dumps(result,indent=2))
    print(report)


if __name__=="__main__":
    main()
