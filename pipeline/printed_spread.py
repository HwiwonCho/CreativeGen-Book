"""Experimental spread sampler with a learned print condition at each Euler step."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from architecture.approved_base import file_digest, verify_base
from architecture.adapters import load_book_conditioner
from architecture.book_conditioner import condition_unet
from architecture.printed_field import PrintedField, character_ids
from training.line_text import distance, embeddings, load_pipe, rgb, tensor


def wrap(field, sentence, line_height, width):
    words = sentence.split()
    ids = torch.arange(95,device=field.paper.device)
    metrics = field.optical_metrics(field.embedding(ids)).detach().cpu().tolist()
    widths = []
    overhang = []
    for word in words:
        tokens = character_ids(word)
        widths.append(sum(metrics[t][0] for t in tokens))
        last = metrics[tokens[-1]]
        overhang.append(last[1]-last[0])
    limit = width*1.4/line_height-.4
    best = [(0,0.,0)] + [(len(words)+1,float("inf"),0)]*len(words)
    for end in range(1,len(words)+1):
        used = overhang[end-1]
        for begin in range(end-1,-1,-1):
            used += widths[begin]
            if used>limit:
                break
            candidate = (best[begin][0]+1, best[begin][1]+(limit-used)**2, begin)
            if candidate<best[end]:
                best[end]=candidate
            used += metrics[0][0]
    if best[-1][0]>len(words):
        return []
    lines=[]
    end=len(words)
    while end:
        begin=best[end][2]
        lines.append(" ".join(words[begin:end]))
        end=begin
    return list(reversed(lines))


def paragraph(field, text, width, available):
    for height in (48, 40, 32):
        lines=wrap(field,text,height,width)
        if lines and len(lines)*height <= available:
            return lines,height
    raise ValueError("Sentence does not fit on this page")


def encode(pipe, x):
    return pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor


def paper_silhouette(blank):
    from scipy.ndimage import binary_fill_holes
    gray=np.asarray(blank.convert("L"))
    hist=np.bincount(gray.flatten(), minlength=256).astype(float)
    mass=hist.cumsum()
    moments=(hist*np.arange(256)).cumsum()
    variance=(moments[-1]*mass-moments*mass[-1])**2 / np.maximum(mass*(mass[-1]-mass),1.)
    threshold=int(variance.argmax())
    inside=binary_fill_holes(gray>threshold)
    return Image.fromarray((inside*255).astype("uint8")).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0))


def diffuse(pipe, source, prior, text, seed, steps, active, weight):
    s = pipe.scheduler
    s.set_timesteps(steps, device="cuda")
    start = steps-active
    s.set_begin_index(start)
    noise = torch.randn(source.shape,generator=torch.Generator().manual_seed(seed)).cuda()
    z = source + float(s.sigmas[start])*noise
    context, pooled = embeddings(pipe,[text])
    h,w = source.size(-2)*8, source.size(-1)*8
    cond=dict(text_embeds=pooled,time_ids=pooled.new_tensor([[h,w,0,0,h,w]]).expand(2,-1))
    for i in range(start,steps):
        t = s.timesteps[i]
        inp = s.scale_model_input(torch.cat((z,z)),t).half()
        u,c = pipe.unet(inp,t,encoder_hidden_states=context,added_cond_kwargs=cond).sample.float().chunk(2)
        eps = torch.lerp(u+6*(c-u),(z-prior)/float(s.sigmas[i]), weight)
        z=s.step(eps,t,z).prev_sample
    return z


def transcribe(image):
    import io
    import subprocess
    buf = io.BytesIO()
    image.resize((image.width*2,image.height*2),Image.Resampling.LANCZOS).save(buf,"PNG")
    result=subprocess.run(["tesseract","stdin","stdout","-l","eng","--psm","6"],input=buf.getvalue(),
                          capture_output=True,check=True,timeout=45)
    return " ".join(result.stdout.decode().split())


@torch.inference_mode()
def run(args):
    manifest=json.loads((args.scene/"manifest.json").read_text())
    text=manifest["plan"]["sentence"]
    base=verify_base(args.model)
    fit=json.loads((args.field/"fit.json").read_text())
    selection=json.loads((args.field/"selection.json").read_text())
    if base!=manifest["plan"]["base_manifest_sha256"] or fit["book_sha256"]!=file_digest(args.book/"adapter.safetensors"):
        raise ValueError("Incompatible book or base model")
    name=args.variant
    weight=selection["weight"]
    if fit["checkpoints"][name]!=file_digest(args.field/f"{name}.safetensors"):
        raise ValueError("Incompatible print field")
    refs={}
    for kind in ("background","illustration"):
        rec=manifest["results"][kind]
        path=args.scene/rec["image"]
        if file_digest(path)!=rec["sha256"]:
            raise ValueError(f"Changed {kind} reference")
        with Image.open(path) as im:
            refs[kind]=im.convert("RGB")
    pipe=load_pipe(args)
    field=PrintedField().cuda().eval()
    field.load_state_dict(load_file(str(args.field/f"{name}.safetensors")))
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    seed=manifest["plan"]["seed"]
    book_seed=manifest["plan"]["stages"]["book"]["seed"]
    bw,bh=896,672
    mask=torch.zeros((2,2,bh,bw),device="cuda")
    mask[:,1,190:580,448:744]=1
    conditioner=load_book_conditioner(pipe.unet,args.book).cuda().eval()
    blank_prompt="Overhead photograph of a flat open book, both facing pages blank and unprinted, bare paper and a central binding."
    with condition_unet(pipe.unet,conditioner(mask)):
        blank_z=pipe(prompt=blank_prompt,negative_prompt="text, writing, illustration, letters, hands",
                     width=bw,height=bh,num_inference_steps=32,guidance_scale=6.,
                     generator=torch.Generator().manual_seed(book_seed),output_type="latent").images.float()
    blank=rgb(pipe.vae.decode(blank_z/pipe.vae.config.scaling_factor).sample[0])
    blank.save(output/"blank-book.png")
    matte=paper_silhouette(blank)
    frame="otsu_generated_book"
    w,h=1440,960
    x0,y0,x1,y1=216,104,1224,856
    rect=(x1-x0,y1-y0)
    def pixels(image, size):
        return tensor(image.resize(size,Image.Resampling.LANCZOS)).cuda()[None]
    bg=encode(pipe,pixels(refs["background"],(w,h)))
    page=encode(pipe,pixels(blank,rect))
    init=bg.clone()
    ph,pw=page.shape[-2:]
    roi=(...,slice(y0//8,y0//8+ph),slice(x0//8,x0//8+pw))
    m=torch.from_numpy(np.array(matte.resize((pw,ph),Image.Resampling.BILINEAR))).cuda()[None,None].float()/255
    init[roi]=torch.lerp(init[roi],page,m)
    ax,ay,aw=272,288,384
    art_roi=(...,slice(ay//8,(ay+aw)//8),slice(ax//8,(ax+aw)//8))
    art=encode(pipe,pixels(refs["illustration"],(aw,aw)))
    init[art_roi]=art
    scene_prompt=("Overhead photograph of an open illustrated book, a single engraving on the left, a blank cream page on the right. "
                  "The setting and the engraving depict: "+text)
    blank_scene=diffuse(pipe,init,init,scene_prompt,seed,24,11,.28)
    blank_rgb=pipe.vae.decode(blank_scene/pipe.vae.config.scaling_factor).sample.clamp(-1,1)
    rgb(blank_rgb[0]).save(output/"blank-scene.png")
    lx,lw=752,424
    lines,line_h=paragraph(field,text,lw,(y1-y0)*.85)
    block_h=len(lines)*line_h
    ly=8*round(((y0+y1-block_h)/2)/8)
    attenuation=(1-field.ink/field.paper.clamp_min(.1)).clamp(0,1)[None,:,None,None]
    alpha=torch.cat([field.coverage(line,line_h,lw) for line in lines],dim=-2)
    # Compute the optical-density change on the generated local paper, in the conditioning branch only.
    density=F.pad(alpha,(lx,w-lx-lw,ly,h-ly-block_h))
    zprior=encode(pipe,(blank_rgb+1)*(1-density*attenuation)-1)
    print_prompt="A photograph of an open illustrated book with real ink printed on the paper. The passage reads: "+text
    final=diffuse(pipe,blank_scene,zprior,print_prompt,seed+1,24,12,weight)
    image=rgb(pipe.vae.decode(final/pipe.vae.config.scaling_factor).sample[0])
    image.save(output/"final.png")
    crop=(lx-8,ly-8,lx+lw+8,ly+block_h+8)
    observed=image.crop(crop)
    observed.save(output/"detail.png")
    read=transcribe(observed)
    blank_read=transcribe(rgb(blank_rgb[0]).crop(crop))
    report=dict(sentence=text,ocr=read,exact=read==text,cer=distance(text,read)/len(text),blank_ocr=blank_read,
                print_lines=lines,seed=seed,book_seed=book_seed,base=base,variant=name,field_sha256=fit["checkpoints"][name],
                references={k:manifest["results"][k]["sha256"] for k in refs},frame=frame,layout=dict(book=[x0,y0,x1,y1],
                art=[ax,ay,aw],text=crop),weight=weight,scene_reverse_steps=11,print_reverse_steps=12,
                post_decode_pixel_edits=0,final_sha256=file_digest(output/"final.png"),code_sha256=file_digest(__file__))
    (output/"result.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def generate_from_sentence(args):
    from architecture import background, illustration
    from pipeline.__main__ import write_json
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/project.json").read_text(encoding="utf-8-sig"))["modules"]
    if not args.sentence.strip() or not (0 <= args.seed < 2**63):
        raise ValueError("Supply a sentence and a nonnegative seed")
    text=" ".join(args.sentence.split())
    params=json.loads((args.field/"fit.json").read_text())
    character_ids(text)
    if set(text).intersection(params["unseen_training_characters"]):
        raise ValueError("The passage contains a character absent from print training")
    verifier=PrintedField().eval()
    verifier.load_state_dict(load_file(str(args.field/f"{args.variant}.safetensors")))
    with torch.inference_mode():
        paragraph(verifier,text,424,639)
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    manifest=out/"manifest.json"
    plan=dict(sentence=text,seed=args.seed,size=768,steps=32,guidance=6.,model=str(args.model.resolve()),
              base_manifest_sha256=verify_base(args.model),stages={"book":{"seed":241126}})
    state=dict(plan=plan,results={},active="starting",status="running")
    try:
        for module in ("background","illustration"):
            directory=root/config[module]["adapter"]
            stage=dict(adapter=str(directory),adapter_sha256=file_digest(directory/"adapter.safetensors"),
                       seed=int.from_bytes(hashlib.sha256(f"{args.seed}/{module}".encode()).digest()[:8],"big")%2**63)
            if module=="background":
                stage.update(background.flatlay_settings(text,background.SCENE_SAMPLER,768))
            else:
                stage["prompt"]=illustration.build_prompt(text)
            plan["stages"][module]=stage
            state["active"]=module
            write_json(manifest,state)
            subprocess.run([sys.executable,"-m","pipeline.generate","--manifest",str(manifest.resolve()),
                            "--stage",module],cwd=root,check=True)
            image=Path(module)/(module+".png")
            state["results"][module]=dict(image=image.as_posix(),sha256=file_digest(out/image))
        state["active"]="spread"
        write_json(manifest,state)
        scene_args=argparse.Namespace(**vars(args))
        scene_args.scene,scene_args.output=out,out/"spread"
        run(scene_args)
        state.update(status="completed",active=None,final=dict(image="spread/final.png",sha256=file_digest(out/"spread/final.png")))
    except Exception as exc:
        state.update(status="failed",error=str(exc))
        raise
    finally:
        write_json(manifest,state)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("model","book","field","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene",type=Path)
    source.add_argument("--sentence")
    parser.add_argument("--seed",type=int,default=241125)
    parser.add_argument("--variant",choices=("reader","control"),default="control")
    args=parser.parse_args()
    if args.scene is None:
        generate_from_sentence(args)
    else:
        run(args)
