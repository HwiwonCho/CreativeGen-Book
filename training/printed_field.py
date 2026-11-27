"""Fit a print condition on real scans and compare it inside SDXL reverse diffusion."""
import argparse
import json
import math
from pathlib import Path
import random
import string

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from architecture.approved_base import file_digest, verify_base
from architecture.printed_field import PrintedField, character_ids
from training.line_reader import make_reader
from training.line_text import CFG, PROMPTS, distance, embeddings, load_pipe, load_rows, ocr, page, prompt, rgb


UNSEEN = ("A green cup rests on the table.", "Five bees circle the pear tree.",
          "The small dog ran into the yard.", "Her coat hung beside the window.",
          "Will the last train stop here?", "We kept a lamp beside the bed.",
          "Cold wind crossed the open field.", "An owl slept among the leaves.")
PAPER_HEIGHT, PAPER_WIDTH, STEPS = 56, 960, 24
CASES = string.ascii_uppercase + string.ascii_lowercase


def collect(rows, data):
    labels, patches, metrics, colors = [], [], [], []
    for row in rows:
        image = page(data / row["image"])
        x0, y0, x1, y1 = row["box"]
        em = y1 - y0
        strip = np.asarray(image.crop((x0,y0,x1,y1)), dtype=np.float32) / 255
        gray = strip.mean(2)
        black, white = np.quantile(gray, [.065, .85])
        colors.append([strip[gray >= white].mean(0), strip[gray <= black].mean(0)])
        chars = []
        for word in row["words"]:
            if chars:
                last = chars[-1][1][2]
                chars.append((" ", [last,y0,word["box"][0],y1]))
            chars.extend((entry["c"], entry["box"]) for entry in word["chars"])
        if "".join(c for c, _ in chars) != row["text"]:
            raise ValueError(f"Misaligned transcript in {row['image']}")
        for i, (char, (left, top, right, bottom)) in enumerate(chars):
            if right <= left or bottom <= top:
                continue
            following = chars[i + 1][1][0] if i + 1 < len(chars) else right
            advance = (following - left) / em
            if not 0 < advance < 3:
                continue
            if char == " ":
                coverage = np.zeros((32, 24), np.float32)
            else:
                crop = image.crop((left, top, right, bottom)).convert("L").resize((24, 32), Image.Resampling.BILINEAR)
                coverage = ((white - np.asarray(crop) / 255) / max(.18, white - black)).clip(0, 1)
            labels.append(character_ids(char)[0])
            patches.append((coverage * 255).round().astype(np.uint8))
            metrics.append([advance, (right-left)/em, (top-y0)/em, (bottom-top)/em])
    return (torch.tensor(labels, device="cuda"), torch.from_numpy(np.stack(patches)).cuda(),
            torch.tensor(metrics, device="cuda"), torch.from_numpy(np.median(colors, axis=0)).cuda())


def read_logits(reader, ink):
    patch = F.interpolate(ink[:, None], (24, 16), mode="bilinear", align_corners=False)
    centered = patch - patch.mean((2,3), keepdim=True)
    return reader(centered / centered.square().mean((2,3), keepdim=True).clamp_min(1e-6).sqrt())


def fit(args, train_rows):
    data = collect(train_rows, args.data)
    ids, crops, layout, colors = data
    freq = ids.bincount(minlength=95)
    reader = make_reader().cuda().eval().requires_grad_(False)
    reader.load_state_dict(torch.load(args.previous/"letter-reader.pt", weights_only=True))
    lookup = torch.tensor([CASES.find(chr(i+32)) for i in range(95)], device="cuda")
    for name, recognition_weight in (("control", 0.), ("reader", .12)):
        torch.manual_seed(112713)
        field = PrintedField().cuda()
        field.paper.copy_(colors[0])
        field.ink.copy_(colors[1])
        opt = torch.optim.AdamW(field.parameters(), lr=.002)
        samples = torch.multinomial(freq[ids].float().rsqrt(), args.steps * 192, replacement=True).view(-1,192)
        for step, ix in enumerate(samples, 1):
            labels = ids[ix]
            logits, geometry = field.letters(labels)
            target = crops[ix].float() / 255
            geom_loss = F.smooth_l1_loss(geometry, layout[ix], beta=.055)
            shape_loss = F.binary_cross_entropy_with_logits(logits, target)
            eligible = lookup[labels] >= 0
            ce = F.cross_entropy(read_logits(reader, logits[eligible].sigmoid()), lookup[labels[eligible]])
            loss = shape_loss + 2.5 * geom_loss + recognition_weight * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == args.steps * 3 // 4:
                opt.param_groups[0]["lr"] *= .3
            if step % 400 == 0 or step == args.steps:
                print(name, step, "ink", float(shape_loss.detach()), "reader", float(ce.detach()), flush=True)
        save_file({k:v.cpu().contiguous() for k,v in field.state_dict().items()}, str(args.output/f"{name}.safetensors"))
    (args.output/"fit.json").write_text(json.dumps(dict(
        lines=len(train_rows), character_crops=len(ids), unseen_training_characters=[chr(i+32) for i in range(95) if not int(freq[i])],
        steps=args.steps, seed=112713, reader_loss_weight=.12, reader_sha256=file_digest(args.previous/"letter-reader.pt"),
        manifest_sha256=file_digest(args.data/"lines.jsonl"), book_sha256=file_digest(args.book/"adapter.safetensors"),
        source_sha256=file_digest(__file__), field_sha256=file_digest(Path(__file__).resolve().parents[1]/"architecture/printed_field.py"),
        checkpoints={name:file_digest(args.output/f"{name}.safetensors") for name in ("control","reader")}),indent=2))


def load_fields(folder):
    result = {}
    for name in ("control", "reader"):
        field = PrintedField().cuda().eval()
        field.load_state_dict(load_file(str(folder/f"{name}.safetensors")))
        result[name] = field
    return result


@torch.inference_mode()
def sample(pipe, fields, conditions, text, seed):
    h, w = PAPER_HEIGHT, PAPER_WIDTH
    images = torch.cat([fields[name](text, h, w) for name, _ in conditions])
    prior = pipe.vae.encode(images).latent_dist.mode() * pipe.vae.config.scaling_factor
    return sample_prior(pipe, prior, [weight for _, weight in conditions], text, seed)


@torch.inference_mode()
def sample_prior(pipe, prior, weights, text, seed):
    count, _, lh, lw = prior.shape
    h, w = lh*8, lw*8
    context, pooled = embeddings(pipe, [prompt(text)] * count)
    cond = dict(text_embeds=pooled, time_ids=pooled.new_tensor([[h,w,0,0,h,w]]).expand(2*count,-1))
    sched = pipe.scheduler
    sched.set_timesteps(STEPS, device="cuda")
    noise = torch.randn((1,4,lh,lw), generator=torch.Generator().manual_seed(seed)).cuda()
    z = (noise * sched.init_noise_sigma).expand(count,-1,-1,-1).clone()
    weight = torch.tensor(weights, device="cuda").view(-1,1,1,1)
    for j, t in enumerate(sched.timesteps):
        sigma = float(sched.sigmas[j])
        prediction = pipe.unet(sched.scale_model_input(torch.cat((z,z)),t).half(),t,
                               encoder_hidden_states=context,added_cond_kwargs=cond).sample.float()
        u, c = prediction.chunk(2)
        eps = u + CFG*(c-u)
        eps = torch.lerp(eps, (z-prior)/sigma, weight)
        z = sched.step(eps,t,z).prev_sample
    images = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
    if not torch.isfinite(images).all():
        raise FloatingPointError("Nonfinite diffusion sample")
    return [rgb(t) for t in images]


def scores(rows, modes):
    count = sum(len(r["text"]) for r in rows)
    return {m:dict(n=len(rows), exact=sum(r["text"]==r[m] for r in rows),
                   cer=sum(distance(r["text"],r[m]) for r in rows)/count) for m in modes}


def evaluate(args, splits):
    record = json.loads((args.output/"fit.json").read_text())
    if (record["manifest_sha256"] != file_digest(args.data/"lines.jsonl")
            or record["book_sha256"] != file_digest(args.book/"adapter.safetensors")
            or any(file_digest(args.output/f"{name}.safetensors") != digest for name,digest in record["checkpoints"].items())):
        raise ValueError("Evaluation inputs changed after fitting")
    base_reference = verify_base(args.model)
    pipe, fields = load_pipe(args), load_fields(args.output)
    candidates = [.40, .70, .90]
    val = random.Random(112734).sample(splits["validation"], 8)
    dev = []
    for i, row in enumerate(val):
        text=row["text"]
        imgs=sample(pipe,fields,[("reader",v) for v in candidates],text,903100+i)
        dev.append(dict(text=text,**{str(v):ocr(im) for v,im in zip(candidates,imgs)}))
    summary=scores(dev, [str(v) for v in candidates])
    weight=min(candidates,key=lambda v:(summary[str(v)]["cer"],-summary[str(v)]["exact"],v))
    print("validation",summary,"selected",weight,flush=True)
    (args.output/"selection.json").write_text(json.dumps(dict(weight=weight, scores=summary, rows=dev),indent=2))
    modes = ("control","reader")
    for group, texts, seed in (("previous",PROMPTS,241125),("unseen",UNSEEN,810300)):
        rows = []
        for i, text in enumerate(texts):
            images=sample(pipe,fields,[(name,weight) for name in modes],text,seed+i)
            row=dict(text=text,seed=seed+i)
            for name, im in zip(modes,images):
                im.save(args.output/f"{group}-{i:02}-{name}.png")
                row[name]=ocr(im)
            rows.append(row)
            print(group,i,row,flush=True)
        (args.output/f"{group}.json").write_text(json.dumps(dict(scores=scores(rows,modes),rows=rows,
                     seed=seed,steps=STEPS,CFG=CFG,weight=weight,base_reference=base_reference),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit","evaluate","run"))
    for key in ("model","book","data","previous","output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    splits=load_rows(args)
    all_prompts = {t.casefold() for t in (*PROMPTS,*UNSEEN)}
    if any(r["text"].casefold() in all_prompts for r in splits["train"]):
        raise ValueError("Benchmark overlap in training")
    if args.action in ("fit","run"):
        fit(args, splits["train"])
    if args.action in ("evaluate","run"):
        evaluate(args,splits)


if __name__=="__main__":
    main()
