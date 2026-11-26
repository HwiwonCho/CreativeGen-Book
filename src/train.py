"""Fit an SDXL attention LoRA; the book head also learns glyph and page conditioning."""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
import random

if __package__:
    from . import background, illustration, book
    from .training_data import digest, read_records
else:
    import background
    import illustration
    import book
    from training_data import digest, read_records


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


def attach_adapter(unet, rank):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class AdapterLinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.base = linear.requires_grad_(False)
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=linear.weight.device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=linear.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        @property
        def weight(self):
            return self.base.weight

        def forward(self, value):
            native = self.base(value)
            with torch.autocast(device_type=value.device.type, enabled=False):
                update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
            return native + update.to(native.dtype)

    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    unet.requires_grad_(False)
    targets = [(name, mod) for name, mod in unet.named_modules()
               if isinstance(mod, nn.Linear) and name.endswith((".attn1.to_q", ".attn1.to_v", ".attn2.to_q", ".attn2.to_v"))]
    if not targets:
        raise ValueError("Expected native UNet attention projections")
    for name, mod in targets:
        parent, key = name.rsplit(".", 1)
        setattr(unet.get_submodule(parent), key, AdapterLinear(mod))
    return {name: p for name, p in unet.named_parameters() if p.requires_grad}


def load_adapter(unet, run):
    """Attach trained weights to a fresh base UNet, for evaluation or inference."""
    import torch
    from safetensors.torch import load_file
    run = Path(run)
    settings = json.loads((run / "settings.json").read_text())
    if settings["base_id"] != MODEL_ID or settings["base_revision"] != REVISION:
        raise ValueError("Adapter base model differs")
    params = attach_adapter(unet, settings["rank"])
    weights = load_file(str(run / "adapter.safetensors"))
    if set(params) != set(weights):
        raise ValueError("Adapter parameter names differ")
    with torch.no_grad():
        for name, p in params.items():
            value = weights[name]
            if p.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
            p.copy_(value)
    return unet.eval()


def load_book_conditioner(unet, run):
    from safetensors.torch import load_file
    if __package__:
        from .book_conditioner import BookConditionEncoder
    else:
        from book_conditioner import BookConditionEncoder
    cond = BookConditionEncoder(unet.conv_in.out_channels)
    cond.load_state_dict(load_file(str(Path(run)/"conditioner.safetensors")), strict=True)
    return cond.to(device=unet.conv_in.weight.device).eval()


def encode_examples(root, size, rows, module, manifest_dir, joint_font=None):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    if joint_font:
        if __package__:
            from .joint_book import build_prompt
            from .joint_data import line_hint
        else:
            from joint_book import build_prompt
            from joint_data import line_hint
    elif module == "book":
        def build_prompt(caption):
            return book.build_condition(caption).scene_prompt
    else:
        build_prompt = {"background": background.build_prompt, "illustration": illustration.build_prompt}[module]
    prompts = list(dict.fromkeys(build_prompt(r["caption"]) for r in rows))
    if joint_font:
        prompts.append("")
    hidden = {p: [] for p in prompts}
    pooled = {}
    local = dict(local_files_only=True, variant="fp16", use_safetensors=True)
    with torch.no_grad():
        for suffix, cls in (("", CLIPTextModel), ("_2", CLIPTextModelWithProjection)):
            tok = CLIPTokenizer.from_pretrained(str(root / ("tokenizer" + suffix)), local_files_only=True)
            encoder = cls.from_pretrained(str(root / ("text_encoder" + suffix)), torch_dtype=torch.float16, **local).cuda().eval()
            for prompt in prompts:
                ids = tok(prompt, padding="max_length", max_length=77, truncation=False, return_tensors="pt").input_ids
                if ids.shape[1] != 77:
                    raise ValueError(f"Caption and instruction exceed SDXL token budget: {prompt}")
                features = encoder(ids.cuda(), output_hidden_states=True)
                hidden[prompt].append(features.hidden_states[-2].cpu())
                if suffix:
                    pooled[prompt] = features.text_embeds.cpu()
                del features
            del encoder
            torch.cuda.empty_cache()
        vae = AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=torch.float32, **local).cuda().eval()
        vae.enable_tiling()
        cache = []
        for row in rows:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            crop = row["crop"] if joint_font else (0, 0, width, height)
            image = image.crop(crop) if joint_font else image
            preserve_spread = module == "book" and row.get("selection") == "IA-PD-facing-pages-v1"
            preserve_plate = module == "illustration" and row.get("selection") == "BL-plates-captioned-v1"
            if preserve_plate or preserve_spread or joint_font:
                target_size = min(512, size) if joint_font and row["view"] == "detail" else size
                scale = target_size / max(image.size)
                scaled = tuple(max(64, round(length*scale/64)*64) for length in image.size)
                left, top = 0, 0
                box = (0, 0, *scaled)
            else:
                scale = size / min(width, height)
                scaled = (max(size, round(width*scale)), max(size, round(height*scale)))
                left, top = (scaled[0]-size)//2, (scaled[1]-size)//2
                box = (left, top, left+size, top+size)
            image = image.resize(scaled, Image.Resampling.LANCZOS).crop(box)
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].cuda().float() / 127.5 - 1
            latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
            if not torch.isfinite(latent).all():
                raise FloatingPointError(f"Nonfinite latent: {row['image']}")
            prompt = build_prompt(row["caption"])
            time_ids = torch.tensor([[height, width, crop[1]+top, crop[0]+left, image.height, image.width]], dtype=torch.float16)
            masks = None
            if module == "book":
                channels = []
                for key in ("glyph_mask","page_mask"):
                    if joint_font and key == "glyph_mask":
                        src_mask = line_hint((width, height), row["lines"], joint_font)
                    else:
                        with Image.open(manifest_dir/row[key]) as src:
                            src_mask = src.convert("L")
                    transformed = src_mask.crop(crop).resize(scaled, Image.Resampling.BILINEAR)
                    if key == "glyph_mask":
                        ink_box = transformed.getbbox()
                        if (ink_box is None or ink_box[0] < left or ink_box[1] < top
                                or ink_box[2] > box[2] or ink_box[3] > box[3]):
                            raise ValueError(f"Square training crop cuts off printed text: {row['image']}")
                    channels.append(torch.from_numpy(np.array(transformed.crop(box))))
                masks = torch.stack(channels)[None]
            cache.append((latent.cpu().half(), torch.cat(hidden[prompt], -1), pooled[prompt], time_ids, masks))
        if joint_font:
            return cache, (torch.cat(hidden[""], -1), pooled[""])
        return cache


def train(args, splits, base_digest):
    import torch
    from torch.utils.checkpoint import checkpoint
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(.95, 19e9 / torch.cuda.get_device_properties(0).total_memory))
    rng = random.Random(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = splits["train"] + splits["validation"]
    settings = {**vars(args), "base_id": MODEL_ID, "base_revision": REVISION, "base_manifest_sha256": base_digest,
                "manifest_sha256": digest(args.manifest), "split_counts": {k:len(v) for k,v in splits.items()},
                "examples": [{k:v for k,v in row.items() if k != "path"} for row in rows]}
    joint = args.joint_lines is not None
    if joint:
        settings.update(book_mode="joint_print", annotation_sha256=digest(args.joint_lines), font_sha256=digest(args.font),
                        scan_counts={k:len({r["group_id"] for r in v}) for k,v in splits.items()},
                        view_schedule=["spread", "spread", "detail"], conditioner_lr=4*args.lr,
                        loss_definition="Half global epsilon MSE, half normalized epsilon MSE around the printed line.")
        if args.init_book:
            settings["init_sha256"] = {key:digest(args.init_book/key) for key in
                                       ("settings.json","conditioner.safetensors","adapter.safetensors")}
    (output/"settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    encoded = encode_examples(args.model, args.size, rows, args.module, args.manifest.parent, args.font if joint else None)
    data, empty = encoded if joint else (encoded, None)
    torch.cuda.empty_cache()
    scheduler = DDPMScheduler.from_pretrained(str(args.model / "scheduler"), local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Unexpected diffusion objective")
    unet = UNet2DConditionModel.from_pretrained(str(args.model/"unet"), local_files_only=True,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16).cuda()
    if args.init_book:
        load_adapter(unet, args.init_book)
        params = {name:p for name,p in unet.named_parameters() if p.requires_grad}
    else:
        params = attach_adapter(unet, args.rank)
    cond = None
    trainable = list(params.values())
    if args.module == "book":
        if __package__:
            from .book_conditioner import BookConditionEncoder, condition_unet
        else:
            from book_conditioner import BookConditionEncoder, condition_unet
        cond = (load_book_conditioner(unet, args.init_book) if args.init_book
                else BookConditionEncoder(unet.conv_in.out_channels).cuda())
        trainable.extend(cond.parameters())
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    groups = ([{"params":list(params.values())}, {"params":list(cond.parameters()), "lr":4*args.lr}]
              if joint else trainable)
    opt = torch.optim.AdamW(groups, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=64.)
    def loss_for(index, generator, null_text=False, null_glyph=False):
        z, text, pooled, time_ids = [v.cuda() for v in data[index][:4]]
        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=torch.float32)
        t = torch.randint(scheduler.config.num_train_timesteps, (1,), device=z.device, generator=generator)
        noisy = scheduler.add_noise(z.float(), noise, t).half()
        scope = nullcontext()
        if cond is not None:
            masks = data[index][4].cuda().float()/255
            inputs = masks
            if null_glyph:
                inputs = masks.clone()
                inputs[:,0] = 0
            scope = condition_unet(unet, cond(inputs))
        if null_text:
            text, pooled = (v.cuda() for v in empty)
        with scope:
            pred = unet(noisy, t, encoder_hidden_states=text,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
        error = (pred.float()-noise).square()
        if joint:
            near = torch.nn.functional.max_pool2d(masks[:,:1], 15, stride=1, padding=7)
            zone = torch.nn.functional.adaptive_avg_pool2d(near, z.shape[-2:])
            local_error = (error.mean(1,keepdim=True)*zone).sum() / zone.sum().clamp_min(1e-6)
            return .5 * (error.mean() + local_error)
        if cond is not None:
            dilated = torch.nn.functional.max_pool2d(masks[:,:1], 5, stride=1, padding=2)
            weights = 1 + 2*torch.nn.functional.adaptive_avg_pool2d(dilated, z.shape[-2:])
            return (error*weights).mean()/weights.mean()
        return error.mean()
    def evaluate(null_glyph=False):
        unet.eval()
        fixed = torch.Generator(device="cuda").manual_seed(args.seed+1)
        with torch.no_grad():
            losses = [float(loss_for(i, fixed, null_glyph=null_glyph)) for i in range(len(splits["train"]), len(data))]
        unet.train()
        return sum(losses)/len(losses)
    before = evaluate()
    if not math.isfinite(before):
        raise FloatingPointError("Nonfinite base validation loss")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    order = []
    view_orders = {"spread": [], "detail": []}
    best = float("inf")
    best_step = None
    best_weights = None
    best_conditioner = None
    with (output/"metrics.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, args.steps+1):
            if joint:
                view = "detail" if step % 3 == 0 else "spread"
                order = view_orders[view]
            if not order:
                order.extend(i for i, row in enumerate(splits["train"]) if not joint or row["view"] == view)
                rng.shuffle(order)
            opt.zero_grad(set_to_none=True)
            unconditional = joint and rng.random() < .1
            loss = loss_for(order.pop(), generator, null_text=unconditional, null_glyph=unconditional)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            scaler.step(opt)
            scaler.update()
            metrics = {"step": step, "loss": float(loss.detach())}
            if joint:
                metrics.update(view=view, unconditional=unconditional)
            if step == args.steps or (args.eval_every and step % args.eval_every == 0):
                score = evaluate()
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite validation loss")
                metrics["validation_epsilon_mse"] = score
                if score < best:
                    best, best_step = score, step
                    best_weights = {k:p.detach().cpu().contiguous() for k,p in params.items()}
                    if cond is not None:
                        best_conditioner = {k:p.detach().cpu().contiguous() for k,p in cond.state_dict().items()}
            log.write(json.dumps(metrics)+"\n")
            log.flush()
    temporary = output / "adapter.tmp"
    save_file(best_weights, str(temporary))
    if cond is not None:
        mask_temp = output / "conditioner.tmp"
        save_file(best_conditioner, str(mask_temp))
        mask_temp.replace(output/"conditioner.safetensors")
    temporary.replace(output/"adapter.safetensors")
    result = {"updates": args.steps, "validation_epsilon_mse_before": before,
              "validation_epsilon_mse_after": best, "selected_step": best_step,
              "final_step_validation_epsilon_mse": score, "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if joint:
        with torch.no_grad():
            for k, param in params.items():
                param.copy_(best_weights[k])
            cond.load_state_dict(best_conditioner)
        result["validation_epsilon_mse_without_glyph"] = evaluate(null_glyph=True)
    (output/"evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", choices=("background","illustration","book"), required=True)
    project = Path(__file__).resolve().parents[1]
    p.add_argument("--config", type=Path, default=project / "configs/project.json")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--joint-lines", type=Path, help="Train on matched transcriptions and printed scan lines.")
    p.add_argument("--init-book", type=Path, help="Warm start joint training from an existing book adapter.")
    p.add_argument("--font", type=Path, help="Typeface for the clean line hint in joint training.")
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    module = config["modules"][args.module]
    args.model = args.model or project / config["model"]
    args.manifest = args.manifest or project / module["manifest"]
    args.output = args.output or project / module["adapter"]
    if args.joint_lines:
        if args.module != "book":
            p.error("Joint printed lines are supported by the book module.")
        args.font = args.font or project/module["font"]
        if not args.font.is_file():
            p.error("Supply the font used by the book sampler.")
    elif args.font or args.init_book:
        p.error("Use --joint-lines with --font or --init-book.")
    if args.size < 64 or args.size % 64 or min(args.steps,args.rank) <= 0 or args.eval_every < 0 or not 0 < args.lr <= 1:
        p.error("Use positive rank/steps/lr and a resolution divisible by 64")
    if not args.manifest.is_file():
        p.error(f"No dataset manifest for {args.module}: {args.manifest}")
    splits = read_records(args.manifest, args.module)
    if __package__:
        from .approved_base import verify_base
    else:
        from approved_base import verify_base
    base_digest = verify_base(args.model)
    if args.joint_lines:
        if __package__:
            from .joint_data import training_views
        else:
            from joint_data import training_views
        splits = training_views(splits, args.joint_lines)
        if args.init_book:
            parent = json.loads((args.init_book/"settings.json").read_text(encoding="utf-8"))
            if (parent["module"],parent["rank"],parent["base_manifest_sha256"]) != ("book",args.rank,base_digest):
                p.error("The initial book checkpoint must match the rank and SDXL base.")
    if args.preflight:
        print(json.dumps({"module":args.module,"base":MODEL_ID,"splits":{k:len(v) for k,v in splits.items()}}))
    else:
        train(args, splits, base_digest)


if __name__ == "__main__":
    main()
