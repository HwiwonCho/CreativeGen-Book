"""Data, training and paired evaluation for single-line SDXL generation from a transcript."""
import argparse
from collections import Counter
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image
import torch

from architecture.approved_base import file_digest, verify_base


@lru_cache(maxsize=16)
def page(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def line_crop(row, folder, ink_height, fixed_width=None):
    x0,y0,x1,y1 = row["box"]
    line_height = y1-y0
    margin = max(2,round(.20*line_height))
    image = page(folder/row["image"])
    box = (max(0,x0-margin),max(0,y0-margin),min(image.width,x1+margin),min(image.height,y1+margin))
    width = fixed_width or 8*max(4,round((box[2]-box[0])*ink_height/line_height/8))
    height = 8*max(4,round(1.4*ink_height/8))
    return image.crop(box).resize((width,height),Image.Resampling.LANCZOS), box


def ocr(image):
    image = image.convert("L").resize((image.width*2,image.height*2),Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf,"PNG")
    result = subprocess.run(["tesseract","stdin","stdout","-l","eng","--psm","7"],
                            input=buf.getvalue(),capture_output=True,check=True,timeout=25)
    return " ".join(result.stdout.decode().split())


def distance(a,b):
    costs = list(range(len(b)+1))
    for i,c in enumerate(a):
        prev,costs = costs,[i+1]
        for j,d in enumerate(b):
            costs.append(min(costs[-1]+1, prev[j+1]+1, prev[j]+(c!=d)))
    return costs[-1]


def tensor(image):
    return torch.from_numpy(np.array(image)).permute(2,0,1).float()/127.5-1


def rgb(t):
    return Image.fromarray(((t.detach().float().cpu().permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().numpy())


def load_rows(args):
    rows = [json.loads(s) for s in (args.data/"lines.jsonl").read_text().splitlines()]
    splits = {k:[r for r in rows if r["split"]==k] for k in ("train","validation","test")}
    groups = [set(r["book"] for r in v) for v in splits.values()]
    if any(not g for g in groups) or any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("Book groups must be disjoint.")
    for name, expected in {(r["image"],r["sha256"]) for r in rows}:
        if file_digest(args.data/name)!=expected:
            raise ValueError(f"Changed scan: {name}")
    return splits


@torch.no_grad()
def assess(args):
    from diffusers import AutoencoderKL
    verify_base(args.model)
    splits=load_rows(args)
    vae=AutoencoderKL.from_pretrained(str(args.model/"vae"),variant="fp16",torch_dtype=torch.float32,
                                      local_files_only=True).cuda().eval()
    rows=[]
    rng=random.Random(5411)
    for name in sorted({r["book"] for r in splits["train"]}):
        group=[r for r in splits["train"] if r["book"]==name]
        rows.extend(rng.sample(group,min(4,len(group))))
    records=[]
    scores=[]
    for height in (12,18,24,32,40,48,64):
        for r in rows:
            crop,_=line_crop(r,args.data,height)
            pixels=tensor(crop).cuda()[None]
            recon=rgb(vae.decode(vae.encode(pixels).latent_dist.mode()).sample[0])
            records.append(dict(book=r["book"],image=r["image"],box=r["box"],text=r["text"],height=height,
                                original=ocr(crop),decoded=ocr(recon)))
        at=[v for v in records if v["height"]==height]
        score={"height":height,"n":len(at)}
        for key in ("original","decoded"):
            score[key+"_exact"]=sum(v["text"]==v[key] for v in at)
            score[key+"_cer"]=sum(distance(v["text"],v[key]) for v in at)/sum(len(v["text"]) for v in at)
        scores.append(score)
        print(score,flush=True)
        if score["decoded_exact"]/score["n"]>=.9 and score["original_exact"]-score["decoded_exact"]<=.05*score["n"]:
            break
    eligible=[s for s in scores if s["decoded_exact"]/s["n"]>=.9 and
              s["original_exact"]-s["decoded_exact"]<=.05*s["n"]]
    chosen=(eligible[0] if eligible else min(scores,key=lambda s:(s["decoded_cer"],s["height"])))["height"]
    # Check the actual, fixed-size training strips as well as the original aspect ratio.
    trials=[]
    for width in (max(640,64*round(chosen*24/64)),max(960,64*round(chosen*36/64))):
        trial=[]
        for r in rows:
            crop,_=line_crop(r,args.data,chosen,width)
            pixels=tensor(crop).cuda()[None]
            decoded=rgb(vae.decode(vae.encode(pixels).latent_dist.mode()).sample[0])
            trial.append(dict(text=r["text"],original=ocr(crop),decoded=ocr(decoded)))
        trials.append(dict(width=width,n=len(trial),decoded_exact=sum(a["text"]==a["decoded"] for a in trial),
                           original_exact=sum(a["text"]==a["original"] for a in trial),
                           decoded_cer=sum(distance(a["text"],a["decoded"]) for a in trial)/sum(len(a["text"]) for a in trial)))
        print("fixed strip",trials[-1],flush=True)
        if trials[-1]["decoded_exact"]>=.9*len(trial):
            break
    picked=min(trials,key=lambda r:(-r["decoded_exact"],r["decoded_cer"],r["width"]))
    report=dict(chosen_height=chosen,chosen_width=picked["width"],strip_trials=trials,scores=scores,lines=records,
                counts={k:len(v) for k,v in splits.items()},
                train_characters=dict(Counter("".join(r["text"] for r in splits["train"]))))
    (args.output/"readability.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("assess","train","compare"))
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--book",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=2400)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    {"assess":assess,"train":train,"compare":compare}[args.action](args)


def train(args):
    from safetensors.torch import save_file
    from torch.nn import functional as F
    from architecture.line_text import LineText,tokenize
    torch.manual_seed(1127)
    torch.set_float32_matmul_precision("high")
    splits=load_rows(args)
    if any(r["text"].casefold() in {p.casefold() for p in PROMPTS} for r in splits["train"]):
        raise ValueError("A generation prompt occurs in the training data")
    calibration=json.loads((args.output/"readability.json").read_text())
    height,width=calibration["chosen_height"],calibration["chosen_width"]
    model_hash=verify_base(args.model)
    pipe=load_pipe(args)
    train_rows=splits["train"]
    val_rows=random.Random(1128).sample(splits["validation"],min(36,len(splits["validation"])))
    rows=train_rows+val_rows
    ids=tokenize([r["text"] for r in rows],"cuda")
    centres=torch.full(ids.shape,-1.,device="cuda")
    pixels=[]
    for i,row in enumerate(rows):
        image,box=line_crop(row,args.data,height,width)
        pixels.append(tensor(image))
        measured=[]
        for j,word in enumerate(row["words"]):
            if j:
                measured.append(-1.)
            measured.extend(((ch["box"][0]+ch["box"][2])*.5-box[0])/(box[2]-box[0]) for ch in word["chars"])
        if len(measured)!=len(row["text"]):
            raise ValueError(f"Misaligned OCR record in {row['image']}")
        centres[i,1:1+len(measured)]=torch.tensor(measured,device="cuda")
    pixels=torch.stack(pixels).cuda()
    n,_,h,w=pixels.shape
    with torch.no_grad():
        latents=torch.cat([pipe.vae.encode(b).latent_dist.mode() for b in pixels.split(4)])*pipe.vae.config.scaling_factor
        gray=(pixels.mean(1,keepdim=True)+1)/2
        p05=torch.quantile(gray.flatten(1),.06,dim=1)[:,None,None,None]
        p90=torch.quantile(gray.flatten(1),.9,dim=1)[:,None,None,None]
        ink=((p90-gray)/(p90-p05).clamp_min(.15)).clamp(0,1)
        del pixels,gray
        predictions=[]
        levels=[]
        pipe.scheduler.set_timesteps(24,device="cuda")
        for begin in range(0,n,6):
            z=latents[begin:begin+6]
            strings=[prompt(r["text"]) for r in rows[begin:begin+len(z)]]
            context,pooled=embeddings(pipe,strings)
            hints=dict(text_embeds=pooled,time_ids=pooled.new_tensor([[h,w,0,0,h,w]]).expand(2*len(z),-1))
            guesses=[]
            sigmas=[]
            for lo,hi in ((1,7),(7,15),(15,23)):
                j=random.Random(1127+begin+lo).randrange(lo,hi)
                sigma=float(pipe.scheduler.sigmas[j])
                noisy=z+sigma*torch.randn_like(z)
                eps=pipe.unet(torch.cat((noisy,noisy)).half()/math.sqrt(1+sigma**2),pipe.scheduler.timesteps[j],
                              encoder_hidden_states=context,added_cond_kwargs=hints).sample.float()
                uncond,cond=eps.chunk(2)
                guesses.append(noisy-sigma*(uncond+CFG*(cond-uncond)))
                sigmas.append(sigma)
            predictions.append(torch.stack(guesses,1))
            levels.append(torch.tensor(sigmas,device="cuda").expand(len(z),-1))
            if begin%180==0:
                print("frozen SDXL training features",begin,"/",n,flush=True)
        base=torch.cat(predictions)
        levels=torch.cat(levels)
    del pipe,predictions
    torch.cuda.empty_cache()
    adapter=LineText().cuda()
    opt=torch.optim.AdamW(adapter.parameters(),lr=.001)
    def alignment(pred,ref,tokens):
        valid=ref>=0
        lengths=(tokens!=0).sum(1).float()[:,None]
        return F.smooth_l1_loss((pred*lengths)[valid],(ref*lengths)[valid],beta=.25)
    ntrain=len(train_rows)
    for i in range(300):
        idx=torch.randint(ntrain,(48,),device="cuda")
        _,pos,_=adapter.positions(ids[idx])
        opt.zero_grad(set_to_none=True)
        alignment(pos,centres[idx],ids[idx]).backward()
        opt.step()
    def losses(idx,j,teacher):
        prior,logits,pos=adapter.condition(ids[idx],h,w,centres[idx] if teacher else None)
        predicted=adapter.denoise(base[idx,j],prior,levels[idx,j])
        zone=F.avg_pool2d(ink[idx],8)
        weight=.2+zone
        score=((predicted-latents[idx]).square()*weight).mean()/weight.mean()
        reconstruction=((prior-latents[idx]).square()*weight).mean()/weight.mean()
        shape=F.binary_cross_entropy_with_logits(logits,ink[idx],weight=.4+ink[idx])
        layout=alignment(pos,centres[idx],ids[idx])
        return score+.55*reconstruction+.65*shape+.25*layout,score
    best=float("inf")
    metrics=[]
    for step in range(1,args.steps+1):
        idx=torch.randint(ntrain,(8,),device="cuda")
        j=torch.randint(3,(len(idx),),device="cuda")
        loss,_=losses(idx,j,teacher=(step%4!=0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),2.,error_if_nonfinite=True)
        opt.step()
        if step==int(args.steps*.7):
            opt.param_groups[0]["lr"]*=.4
        if step%300==0 or step==args.steps:
            scores=[]
            with torch.no_grad():
                for idx in torch.arange(ntrain,n,device="cuda").split(8):
                    for level in range(3):
                        _,score=losses(idx,torch.full_like(idx,level),False)
                        scores.extend([float(score)]*len(idx))
            val=sum(scores)/len(scores)
            metrics.append(dict(step=step,loss=float(loss.detach()),validation_clean_mse=val))
            print(metrics[-1],flush=True)
            if val<best:
                best,best_step=val,step
                tmp=args.output/"line.tmp"
                save_file({k:v.cpu().contiguous() for k,v in adapter.state_dict().items()},str(tmp))
                tmp.replace(args.output/"line.safetensors")
    record=dict(steps=args.steps,best_step=best_step,metrics=metrics,model_sha256=model_hash,
                book_sha256=file_digest(args.book/"adapter.safetensors"),manifest_sha256=file_digest(args.data/"lines.jsonl"),
                source_sha256=file_digest(__file__),architecture_sha256=file_digest(Path(__file__).resolve().parents[1]/"architecture/line_text.py"),
                train_lines=ntrain,validation_lines=len(val_rows),height=h,width=w,train_seed=1127,
                checkpoint_sha256=file_digest(args.output/"line.safetensors"))
    (args.output/"training.json").write_text(json.dumps(record,indent=2)+"\n")


CFG=6.
PROMPTS=("A seagull flies just above the surface of the water.",
         "The blue jar is empty.", "The bell rang twice.", "We see the quiet sea.",
         "Small birds sit on the sill.", "A little light filled the room.", "The brown fox jumps.",
         "She folded the letter.", "The rain stopped at noon.", "His book fell open.",
         "A bright kite rose slowly.", "A moonlit path leads home.", "Bring the red ribbon.",
         "Is the old gate locked?", "Please close the glass door.", "The reeds bend east.",
         "It is still too soon.", "Fresh bread cooled on the shelf.", "No boats left at dawn.",
         "They will return in June.")


def prompt(text):
    return "A close-up photograph of a single line printed on a book page. The line reads: "+text


def load_pipe(args):
    from diffusers import StableDiffusionXLPipeline
    from architecture.adapters import load_adapter
    pipe=StableDiffusionXLPipeline.from_pretrained(str(args.model),variant="fp16",torch_dtype=torch.float16,
                  use_safetensors=True,local_files_only=True,add_watermarker=False).to("cuda")
    load_adapter(pipe.unet,args.book)
    pipe.vae.float().eval()
    return pipe


def embeddings(pipe,strings):
    pos,neg,pooled,neg_pooled=pipe.encode_prompt(strings,device="cuda",do_classifier_free_guidance=True)
    return torch.cat((neg,pos)),torch.cat((neg_pooled,pooled))


@torch.no_grad()
def compare(args):
    from safetensors.torch import load_file
    from architecture.line_text import LineText,tokenize
    record=json.loads((args.output/"training.json").read_text())
    if verify_base(args.model)!=record["model_sha256"] or file_digest(args.book/"adapter.safetensors")!=record["book_sha256"]:
        raise ValueError("Comparison checkpoints differ from training")
    if file_digest(args.output/"line.safetensors")!=record["checkpoint_sha256"]:
        raise ValueError("Print checkpoint has changed")
    pipe=load_pipe(args)
    model=LineText().cuda().eval()
    model.load_state_dict(load_file(str(args.output/"line.safetensors")))
    h,w=record["height"],record["width"]
    results=[]
    for i,text in enumerate(PROMPTS):
        seed=241125+i
        init=torch.randn((1,4,h//8,w//8),generator=torch.Generator().manual_seed(seed)).cuda()
        context,emb=embeddings(pipe,[prompt(text)])
        cond=dict(text_embeds=emb,time_ids=emb.new_tensor([[h,w,0,0,h,w]]).expand(2,-1))
        prior,_,_=model.condition(tokenize([text],"cuda"),h,w)
        result=dict(text=text,seed=seed)
        for mode in ("book","character"):
            scheduler=pipe.scheduler
            scheduler.set_timesteps(24,device="cuda")
            z=init*scheduler.init_noise_sigma
            for j,t in enumerate(scheduler.timesteps):
                sigma=float(scheduler.sigmas[j])
                eps=pipe.unet(scheduler.scale_model_input(torch.cat((z,z)),t).half(),t,encoder_hidden_states=context,
                              added_cond_kwargs=cond).sample.float()
                u,c=eps.chunk(2)
                eps=u+CFG*(c-u)
                if mode=="character":
                    clean=model.denoise(z-sigma*eps,prior,sigma)
                    eps=(z-clean)/sigma
                z=scheduler.step(eps,t,z).prev_sample
            image=rgb(pipe.vae.decode(z/pipe.vae.config.scaling_factor).sample[0])
            image.save(args.output/f"{mode}-{i:02}.png")
            result[mode]=ocr(image)
        results.append(result)
        print(i,result,flush=True)
    totals={}
    for mode in ("book","character"):
        totals[mode]=dict(exact=sum(r[mode]==r["text"] for r in results),
                          cer=sum(distance(r[mode],r["text"]) for r in results)/sum(len(r["text"]) for r in results))
    (args.output/"comparison.json").write_text(json.dumps(dict(totals=totals,examples=results),indent=2)+"\n")
    print(totals,flush=True)


if __name__=="__main__":
    main()
