"""Guide the last Euler steps of the book generator with an authored glyph mask."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from . import book, train
    from .approved_base import file_digest, verify_base
else:
    import book
    import train
    from approved_base import file_digest, verify_base


def paper_ink(paper, glyph):
    """Reflectance of printed type under the generated leaf's local illumination."""
    import torch
    from torch.nn import functional as F

    # Fractional coverage models the camera footprint along a printed edge.
    kernel = paper.new_tensor([.09, .82, .09])
    coverage = F.conv2d(glyph, (kernel[:, None] * kernel[None, :])[None, None], padding=1)
    luma = paper.mean(1, keepdim=True)
    surface = luma - F.avg_pool2d(F.pad(luma, (2, 2, 2, 2), mode="replicate"), 5, 1)
    density = paper.new_tensor([1.05, 1.10, 1.14]).view(1, 3, 1, 1)
    transmittance = torch.exp(-density * (1 - .8 * surface).clamp(.9, 1.1))
    return paper * (1 - coverage * (1 - transmittance))


PROJECT = Path(__file__).resolve().parents[1]
PAPER_PROMPTS = {
    "bound": ("A perfectly centered upright open book lying flat on a surface viewed directly from above. "
              "Two facing blank pages of cream paper, one page on either side of the center fold, "
              "visible binding down the middle. The entire landscape-format spread fits inside the frame. Photorealistic."),
    "sketchbook": ("Top-down product photograph of one open unused unlined artist's sketchbook, "
                   "lying flat and centered horizontally on a wood desk. The left and right sheets "
                   "are both completely blank, smooth ivory drawing paper. Visible center seam, "
                   "all four corners in frame, soft diffuse daylight."),
}
NEGATIVE = ("text, lettering, handwriting, dense writing, printed paragraphs, printed words, "
            "symbols, scribbles, ruled lines, illustrations, ornaments, other books, pens, hands")


def encode_glyph(vae, blank, glyph, profile):
    """Build the starting estimate used by the final book denoising steps."""
    import torch
    from torch.nn import functional as F

    scale = vae.config.scaling_factor
    rgb = (vae.decode(blank.float()/scale, return_dict=False)[0]+1)/2
    if profile == "paper":
        paper = rgb.clamp(0, 1)
        target = paper_ink(paper, glyph)
        sketch = vae.encode(target*2-1).latent_dist.mode()*scale
        return blank.float(), sketch
    window = F.max_pool2d(glyph, 33, stride=1, padding=16)
    shade = (rgb.clamp(0,1)*window).sum(dim=(2,3),keepdim=True)/window.sum()
    flat = shade.expand_as(rgb).contiguous()
    plain = vae.encode(flat*2-1).latent_dist.mode()*scale
    sketch = vae.encode(flat*(1-.79*glyph)*2-1).latent_dist.mode()*scale
    return plain, sketch


def guide_clean(vae, clean, blank, glyph, prior, iterations, lr, profile, *, ink_target=None, ink_window=41):
    """Use the full decoded page to fit local ink contrast without a pixel overlay."""
    import torch
    from torch.nn import functional as F

    if not iterations:
        return clean
    ys, xs = torch.where(glyph[0, 0] > .01)
    if not xs.numel():
        raise ValueError("The glyph mask contains no strokes.")
    h, w = clean.shape[-2:]
    x0 = max(0, int(xs.min())//8 - 4)
    x1 = min(w, int(xs.max())//8 + 5)
    y0 = max(0, int(ys.min())//8 - 4)
    y1 = min(h, int(ys.max())//8 + 5)
    pad = (x0, w-x1, y0, h-y1)
    roi = (..., slice(8*y0, 8*y1), slice(8*x0, 8*x1))
    alpha = glyph[roi]
    ink_area = F.max_pool2d(alpha, ink_window, stride=1, padding=ink_window//2)
    support = F.avg_pool2d(ink_area, 8, stride=8)
    original = clean.detach().float()

    def decode(z):
        rgb = vae.decode(z / vae.config.scaling_factor, return_dict=False)[0]
        return (rgb[roi] + 1) / 2

    with torch.no_grad():
        plain, sketch = prior
        initial = (blank.float() + (sketch-plain) - original)[...,y0:y1,x0:x1]
        if ink_target is None:
            paper = decode(blank.float()).clamp(0, 1)
            target = paper_ink(paper, alpha) if profile == "paper" else paper*(1-.79*alpha)
        else:
            target = ink_target[roi]
        # Weight the narrow strokes and counters across all colour channels.
        weights = alpha / alpha.sum().clamp_min(1)
        gaps = ink_area*(1-alpha)
        weights = weights + 3 * gaps / gaps.sum().clamp_min(1)
        frame = 1-ink_area
        weights = weights + frame / frame.sum().clamp_min(1)

    with torch.enable_grad():
        delta = initial.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=lr)
        for j in range(iterations):
            opt.param_groups[0]["lr"] = lr * (.12 + .88 * (1+math.cos(math.pi*j/iterations))/2)
            rgb = decode(original + F.pad(support*delta, pad))
            error = rgb-target
            colour = (weights*error.square().mean(1,keepdim=True)).sum()
            chroma = error-error.mean(1, keepdim=True)
            colour = colour + 2*(weights*chroma.square().mean(1,keepdim=True)).sum()
            dx = error[:,:,:,1:] - error[:,:,:,:-1]
            dy = error[:,:,1:,:] - error[:,:,:-1,:]
            edges = (weights[:,:,:,1:]*dx.square().mean(1,keepdim=True)).sum()
            edges += (weights[:,:,1:,:]*dy.square().mean(1,keepdim=True)).sum()
            loss = colour + .35*edges + .005*(delta-initial).square().mean()
            (delta.grad,) = torch.autograd.grad(loss, delta)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return (original + F.pad(support*delta.detach(), pad)).to(clean.dtype)


def register_scene_page(vae, latent, geometry, table_info, glyph_image):
    """Register the lettering to the measured right leaf of the sampled scene."""
    import numpy as np
    import torch
    from PIL import Image, ImageFilter

    _, _, h, w = latent.shape
    size = (w*8,h*8)
    if glyph_image.size != size:
        raise ValueError("The projected glyph and sampled scene have different sizes.")
    plane = table_info["unprojected_geometry"]
    transform = np.array(table_info["projection_matrix"],dtype=float)
    transform /= transform[2,2]
    warp = (Image.Transform.PERSPECTIVE,tuple(transform.flat[:8]),Image.Resampling.BICUBIC)
    glyph = glyph_image.convert("L").transform(size,*warp)
    bbox = glyph.point(lambda x: 255 if x>=8 else 0).getbbox()
    report = {"passed": False, "method": "four_page_edges", "edges": {}, "glyph_box_rectified": bbox}
    visible_box = glyph_image.convert("L").getbbox()
    if bbox is None or visible_box is None:
        return None, {**report,"reason":"empty glyph"}
    lx,ly,rx,ry = geometry["right_page_box"]
    if not (lx <= visible_box[0] and ly <= visible_box[1] and rx >= visible_box[2] and ry >= visible_box[3]):
        return None, {**report,"reason":"glyph outside expected right leaf"}
    with torch.no_grad():
        decoded = vae.decode(latent.float()/vae.config.scaling_factor,return_dict=False)[0]
        pixels = ((decoded[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
    unit = min(size)/768
    rectified = Image.fromarray(pixels).transform(size,*warp)
    rgb = np.asarray(rectified.filter(ImageFilter.GaussianBlur(1.2*unit)),dtype=np.float32)
    step = max(1,round(2*unit))
    dx = np.linalg.norm(rgb[:,2*step:]-rgb[:,:-2*step],axis=-1)
    dy = np.linalg.norm(rgb[2*step:]-rgb[:-2*step],axis=-1).T
    fold, top, right, bottom = plane["right_page_box"]
    x_trim, y_trim = round(.12*(right-fold)), round(.13*(bottom-top))
    search = max(4,round(18*unit))
    neighbors = max(1,round(4*unit))
    positions = {}
    for key, deriv, expected, start, stop in (
        ("fold",dx,fold,top+y_trim,bottom-y_trim), ("right",dx,right,top+y_trim,bottom-y_trim),
        ("top",dy,top,fold+x_trim,right-x_trim), ("bottom",dy,bottom,fold+x_trim,right-x_trim)):
        strip = deriv[max(0,start):min(deriv.shape[0],stop)]
        index = np.arange(max(step,expected-search),min(expected+search+1,deriv.shape[1]+step))
        if len(strip)<7 or len(index)<3:
            return None, {**report,"reason":"insufficient visible page extent"}
        bands = np.stack([np.median(row[:,index-step],axis=0) for row in np.array_split(strip,7)])
        peak = int(index[np.percentile(bands,35,axis=0).argmax()])
        strength = bands[:,abs(index-peak)<=neighbors].max(axis=1)
        supported = int((strength>=14).sum())
        report["edges"][key] = {"coordinate":peak, "supported_bands":supported,"total_bands":7}
        positions[key] = peak
    top_slope = 0.
    if report["edges"]["top"]["supported_bands"]<5:
        # A raised page can shorten its visible height while the spine and
        # bottom stay put. Trace the transition into the leaf's own paper.
        reach = round(.22*(bottom-top))
        offset = max(2,round(8*unit))
        ix = np.arange(max(offset,top-reach),min(rgb.shape[0]-offset,top+reach+1))
        x_bands = np.array_split(np.arange(fold+x_trim,right-x_trim),9)
        points = []
        for band in x_bands:
            col = np.median(rgb[:,band],axis=1)
            stock = np.median(col[top+round(.3*(bottom-top)):bottom-y_trim],axis=0)
            distance = np.linalg.norm(col-stock,axis=1)
            gain = distance[ix-offset]-distance[ix+offset]
            jump = np.linalg.norm(col[ix-step]-col[ix+step],axis=1)
            score = np.maximum(gain,0)*np.minimum(jump/12,1)*np.exp(-distance[ix+offset]/38)
            i = int(np.argmax(score))
            if gain[i]>=14 and jump[i]>=10 and distance[ix[i]+offset]<55:
                near = abs(ix-ix[i])<=neighbors
                row = ix[near][jump[near].argmax()]
                points.append((np.mean(band),row))
        if len(points)>=6:
            p = np.asarray(points)
            inliers = abs(p[:,1]-np.median(p[:,1]))<=.055*(bottom-top)
            p = p[inliers]
            if len(p)>=6 and np.ptp(p[:,0])>.55*(right-fold):
                slope, intercept = np.polyfit(p[:,0],p[:,1],1)
                if abs(slope)<.20 and np.max(abs(p[:,1]-slope*p[:,0]-intercept))<.035*(bottom-top):
                    top_slope = float(slope)
                    positions["top"] = round(float(slope*(positions["fold"]+positions["right"])/2+intercept))
                    report["edges"]["top"] = {"coordinate":positions["top"],"supported_bands":len(p),
                                              "total_bands":9,"method":"paper_colour_transition",
                                              "slope":top_slope}
    missing = [key for key,edge in report["edges"].items() if edge["supported_bands"]<5]
    if missing:
        return None, {**report,"reason":"page boundary uncertain","uncertain_edges":missing}
    margin = max(2,round(12*unit))
    x0,y0,x1,y1 = bbox
    width = positions["right"]-positions["fold"]
    height = positions["bottom"]-positions["top"]
    cx, cy = (x0+x1)/2, (y0+y1)/2
    fraction = (cy-top)/(bottom-top)
    if min(width,height)<=2*margin or not 0<fraction<1:
        return None, {**report,"reason":"no room inside measured page"}
    x, y = (positions["fold"]+positions["right"])/2, positions["top"]+fraction*height
    scale = min(1.45, .72*width/(x1-x0), .55*height/(y1-y0))
    half_w, half_h = .5*scale*(x1-x0), .5*scale*(y1-y0)
    baseline_slope = (1-fraction)*top_slope
    if min(width/2-half_w,y-positions["top"]-half_h-abs(top_slope-baseline_slope)*half_w,
           positions["bottom"]-y-half_h-abs(baseline_slope)*half_w)<margin:
        return None, {**report,"reason":"glyph too close to measured page boundary"}
    layout = np.array([[scale,0,x-scale*cx],[scale*baseline_slope,scale,y-scale*(cy+baseline_slope*cx)],[0,0,1]])
    pullback = np.diag([2.,2.,1.]) @ transform @ np.linalg.inv(layout) @ np.linalg.inv(transform)
    pullback /= pullback[2,2]
    enlarged = glyph_image.convert("L").resize((size[0]*2,size[1]*2),Image.Resampling.LANCZOS)
    aligned = enlarged.transform(size,Image.Transform.PERSPECTIVE,tuple(pullback.flat[:8]),Image.Resampling.BICUBIC)
    aligned_box = aligned.point(lambda value: 255 if value>=8 else 0).getbbox()
    cx_page = (positions["fold"]+positions["right"])/2
    corners = np.array([(xp,yp + (xp-cx_page)*top_slope if yp==positions["top"] else yp,1.)
                        for xp,yp in [(positions["fold"],positions["top"]),(positions["right"],positions["top"]),
                                      (positions["right"],positions["bottom"]),(positions["fold"],positions["bottom"])]])
    projected = corners @ transform.T
    return aligned, {**report,"passed":True,"reason":"four page boundaries supported",
                     "registered_glyph_box":aligned_box,"scale":scale,"baseline_slope":baseline_slope,
                     "right_page_corners":(projected[:,:2]/projected[:,2:]).tolist()}


def guide_scene_ink(vae, latent, glyph_image, *, page_registration=None):
    """Fit printed strokes in the generated scene while keeping its lighting and page position."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    _, _, h, w = latent.shape
    glyph_image = glyph_image.convert("L")
    if latent.shape[:2] != (1,4) or glyph_image.size != (8*w,8*h):
        raise ValueError("Scene lettering requires one SDXL latent and a registered full-resolution glyph mask.")
    bounds = glyph_image.point(lambda x: 255 if x >= 3 else 0).getbbox()
    if bounds is None:
        raise ValueError("The projected sentence contains no ink.")
    if page_registration is not None:
        from PIL import Image, ImageDraw
        from scipy import ndimage
        if __package__:
            from .joint_framing import clean_left_stock
        else:
            from joint_framing import clean_left_stock
        if not page_registration["passed"]:
            raise ValueError("Cannot guide paper stock without a registered page.")
        page = Image.new("L",glyph_image.size)
        ImageDraw.Draw(page).polygon([tuple(p) for p in page_registration["right_page_corners"]], fill=255)
        page_bounds = page.getbbox()
        if page_bounds is None:
            raise ValueError("Registered page falls outside the scene.")
        distance = ndimage.distance_transform_edt(np.pad(np.asarray(page)>0,1))[1:-1,1:-1]
        inset = max(6,round(.045*min(page_bounds[2]-page_bounds[0],page_bounds[3]-page_bounds[1])))
        coverage = np.clip((distance-inset)/inset,0,1).astype(np.float32)
    left, top, right, bottom = bounds
    x0, y0 = max(0,left//8-12), max(0,top//8-12)
    x1, y1 = min(w,(right+7)//8+12), min(h,(bottom+7)//8+12)
    roi = (...,slice(y0,y1),slice(x0,x1))
    pixels = (...,slice(8*y0,8*y1),slice(8*x0,8*x1))
    scale = vae.config.scaling_factor

    def decode(z):
        return (vae.decode(z/scale, return_dict=False)[0]+1)/2

    def encode(rgb):
        return vae.encode(2*rgb.clamp(0,1)-1).latent_dist.mode()*scale

    with torch.inference_mode(False):
        with torch.no_grad():
            scene = latent.detach().to(dtype=torch.float32).clone()
            whole_glyph = torch.from_numpy(np.array(glyph_image)).to(scene)[None,None]/255
            alpha = whole_glyph[pixels]
            full_page = decode(scene).clamp(0,1)
            if page_registration is None:
                observed = full_page[pixels].contiguous()
                light = F.max_pool2d(observed,9,1,4)
                stock = -F.max_pool2d(-light,9,1,4)
                ink_zone = F.max_pool2d(alpha,15,1,7)
                paper = torch.lerp(observed,stock,ink_zone)
                desired = paper_ink(paper,alpha)
                prior = (encode(observed),encode(desired))
                iterations, lr, ink_window = 48, .045, 41
            else:
                full_rgb = (full_page[0].permute(1,2,0)*255).round().byte().cpu().numpy()
                estimate = clean_left_stock(Image.fromarray(full_rgb),page,{"left_page_box":page_bounds})
                stock = torch.from_numpy(np.array(estimate)).to(scene).permute(2,0,1)[None]/255
                support = torch.from_numpy(coverage).to(scene)[None,None]
                paper = torch.lerp(full_page,stock,support)
                # A single full-context encoding gives bare stock a stable
                # latent, including its colour. Only type needs a local fit.
                appearance = paper_ink(paper,whole_glyph)
                scene = torch.lerp(scene,encode(appearance),F.avg_pool2d(support,8))
                observed = decode(scene)[pixels].clamp(0,1).contiguous()
                desired = appearance[pixels]
                prior = (scene[roi],scene[roi])
                iterations, lr, ink_window = 20, .016, 15
            crop = scene[roi].contiguous()
            local_target = decode(crop) + (desired-observed)
        corrected = guide_clean(vae,crop,crop,alpha,prior,iterations,lr,"paper",
                                ink_target=local_target,ink_window=ink_window)
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("Nonfinite scene lettering latent.")
        scene[roi] = corrected
        return scene.to(latent.dtype).detach()


def prepare_layout(path, font_path, font_size):
    from PIL import ImageFont

    layout = book.PageLayout.from_json(path)
    left, top, right, bottom = layout.print_box
    page_top, page_bottom = layout.right_page_box[1::2]
    metrics = ImageFont.truetype(str(font_path), font_size).getmetrics()
    height = max(bottom-top, 2*sum(metrics) + layout.line_spacing)
    if height > page_bottom-page_top:
        raise ValueError("The selected font size exceeds the available page height.")
    top = max(page_top, min((top+bottom-height)//2, page_bottom-height))
    return replace(layout, font_size=font_size, print_box=(left, top, right, top+height))


def run(args, output):
    import diffusers
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from PIL import Image
    if __package__:
        from .book_conditioner import condition_unet
    else:
        from book_conditioner import condition_unet

    code_hash = file_digest(Path(__file__))
    base_hash = verify_base(args.model)
    prompt = PAPER_PROMPTS[args.paper]
    settings = json.loads((args.adapter / "settings.json").read_text())
    if settings["module"] != "book" or settings["base_manifest_sha256"] != base_hash:
        raise ValueError("Expected a book adapter for the verified SDXL Base.")
    layout = prepare_layout(args.layout, args.font, args.font_size)
    size = layout.canvas_size
    masks = [book.render_masks(text, args.font, layout) for text in args.sentence]
    device = torch.device("cuda")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(args.model), variant="fp16", torch_dtype=torch.float16, local_files_only=True,
        use_safetensors=True, add_watermarker=False,
    ).to(device)
    if not isinstance(pipe.scheduler, EulerDiscreteScheduler) or pipe.scheduler.config.prediction_type != "epsilon":
        raise ValueError("This print sampler requires the approved epsilon-predicting Euler model.")
    train.load_adapter(pipe.unet, args.adapter).requires_grad_(False)
    with torch.no_grad():
        for name, weight in pipe.unet.named_parameters():
            if name.endswith(".lora_B"):
                weight.mul_(args.adapter_scale)
    cond = train.load_book_conditioner(pipe.unet, args.adapter).requires_grad_(False)
    pipe.vae.float().eval().requires_grad_(False)
    pipe.vae.enable_gradient_checkpointing()
    offload_unet = torch.cuda.get_device_properties(device).total_memory < 18 * 1024**3

    def tensor(mask):
        return torch.from_numpy(np.array(mask)).to(device=device, dtype=torch.float32)[None,None] / 255

    @torch.no_grad()
    def save_image(z, name):
        rgb = pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor, return_dict=False)[0]
        pixels = ((rgb[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        Image.fromarray(pixels).save(output/name)

    @torch.no_grad()
    def predict(z, timestep):
        model_input = pipe.scheduler.scale_model_input(torch.cat([z,z]), timestep)
        uncond, scene = pipe.unet(model_input, timestep, encoder_hidden_states=text,
                                  added_cond_kwargs=conditions, return_dict=False)[0].chunk(2)
        return uncond + args.cfg*(scene - uncond)

    with torch.no_grad():
        pos, neg, pooled, neg_pooled = pipe.encode_prompt(prompt, device=device, negative_prompt=NEGATIVE)
        pipe.text_encoder.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()
        text = torch.cat([neg, pos])
        time_ids = torch.tensor([[size,size,0,0,size,size]], device=device, dtype=text.dtype).repeat(2,1)
        conditions = {"text_embeds": torch.cat([neg_pooled,pooled]), "time_ids": time_ids}
        page = tensor(masks[0][1])
        residual = cond(torch.cat([torch.zeros_like(page), page], 1)).repeat(2,1,1,1)
        pipe.scheduler.set_timesteps(args.steps, device=device)
        z = pipe.prepare_latents(1,4,size,size,text.dtype,device,
                                 torch.Generator("cpu").manual_seed(args.seed),None)
    first = args.steps - args.guide_steps
    with condition_unet(pipe.unet, residual), torch.no_grad():
        reference = [z]
        clean_reference = []
        for timestep in pipe.scheduler.timesteps:
            update = pipe.scheduler.step(predict(z,timestep), timestep, z)
            clean_reference.append(update.pred_original_sample)
            z = update.prev_sample
            reference.append(z)
        save_image(z, "baseline.png")
        if args.print_profile == "paper":
            if __package__:
                from .print_geometry import register_print
                from .spread_geometry import resolve_spread
            else:
                from print_geometry import register_print
                from spread_geometry import resolve_spread
            with Image.open(output/"baseline.png") as image:
                leaf = image.convert("RGB")
        for index,(glyph,_) in enumerate(masks):
            if args.print_profile == "paper":
                glyph.save(output/f"layout-glyph-{index:04d}.png")
                outline, _, geometry = resolve_spread(leaf, glyph)
                glyph = register_print(leaf, glyph, geometry, outline, layout.font_size)
            glyph.save(output/f"glyph-{index:04d}.png")
            alpha = tensor(glyph)
            prior = encode_glyph(pipe.vae, reference[-1], alpha, args.print_profile)
            neighborhood = F.max_pool2d(alpha, 65, stride=1, padding=32)
            support = F.avg_pool2d(neighborhood, 8, stride=8).to(text.dtype)
            pipe.scheduler.set_timesteps(args.steps, device=device)
            pipe.scheduler.set_begin_index(first)
            z = reference[first]
            for i in range(first, args.steps):
                timestep = pipe.scheduler.timesteps[i]
                update = pipe.scheduler.step(predict(z,timestep), timestep, z)
                iters = args.final_iterations if i+1 == args.steps else args.iterations
                clean = update.pred_original_sample
                if offload_unet:
                    pipe.unet.cpu()
                    torch.cuda.empty_cache()
                guided = guide_clean(pipe.vae, clean, clean_reference[i], alpha, prior, iters, args.lr, args.print_profile)
                if offload_unet:
                    torch.cuda.empty_cache()
                    pipe.unet.to(device)
                fraction = float(1-pipe.scheduler.sigmas[i+1]/pipe.scheduler.sigmas[i])
                proposed = update.prev_sample + fraction*(guided-clean).to(z.dtype)
                z = torch.lerp(reference[i+1], proposed.to(z.dtype), support)
            save_image(z, f"{index:04d}.png")
            if args.save_latent:
                from safetensors.torch import save_file
                save_file({"latent": z.detach().cpu().contiguous()}, str(output/f"{index:04d}.safetensors"))
            print(f"Generated {index+1}/{len(masks)}", flush=True)
    record = {**vars(args), "prompt": prompt, "negative_prompt": NEGATIVE, "page_layout": asdict(layout),
              "base_manifest_sha256": base_hash, "font_sha256": file_digest(args.font),
              "adapter_sha256": file_digest(args.adapter/"adapter.safetensors"),
              "conditioner_sha256": file_digest(args.adapter/"conditioner.safetensors"),
              "scheduler": type(pipe.scheduler).__name__, "torch": torch.__version__,
              "diffusers": diffusers.__version__, "code_sha256": code_hash,
              "scale": "book LoRA output multiplier; page conditioner fixed",
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if args.print_profile == "paper":
        record["geometry_code_sha256"] = file_digest(Path(__file__).with_name("print_geometry.py"))
    if args.save_latent:
        record["output_sha256"] = {path.name: file_digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output/"run.json").write_text(json.dumps(record, ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def main():
    cfg = json.loads((PROJECT/"configs/project.json").read_text(encoding="utf-8-sig"))
    module = cfg["modules"]["book"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sentence", required=True, action="append", help="Repeat to print on the same sampled book.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-latent", action="store_true", help="Keep the exact scaled book latent for the assembled spread.")
    p.add_argument("--paper", choices=tuple(PAPER_PROMPTS), default="bound")
    p.add_argument("--print-profile", choices=("paper", "legacy"), default="legacy")
    p.add_argument("--model", type=Path, default=Path(cfg["model"]))
    p.add_argument("--adapter", type=Path, default=PROJECT/module["adapter"])
    p.add_argument("--layout", type=Path, default=PROJECT/module["layout"])
    p.add_argument("--font", type=Path, default=PROJECT/module["font"])
    p.add_argument("--font-size", type=int, default=36)
    p.add_argument("--seed", type=int, default=241126)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--guide-steps", type=int, default=4)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--final-iterations", type=int, default=80)
    p.add_argument("--lr", type=float, default=.065)
    p.add_argument("--adapter-scale", type=float, default=1.)
    p.add_argument("--cfg", type=float, default=6.)
    args = p.parse_args()
    if (not (1 <= args.guide_steps <= args.steps) or min(args.iterations,args.final_iterations) < 0
            or not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.adapter_scale <= 1
            or not 1 <= args.cfg <= 30 or args.font_size < 1):
        p.error("Check the denoising steps, guidance iterations and learning rate.")
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        run(args,stage)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
