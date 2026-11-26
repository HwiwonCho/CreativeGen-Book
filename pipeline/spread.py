"""Sample the final book from its independently generated background, pages and illustration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from architecture.approved_base import file_digest, verify_base
from architecture.background import SCENE_SAMPLER
from . import geometry as page_geometry
from . import print_page
from .text_conditioning import text_conditioning


def fit_plate(image, box):
    x0, y0, x1, y1 = box
    x0, y0 = (x0+7)//8*8, (y0+7)//8*8
    x1, y1 = x1//8*8, y1//8*8
    scale = min((x1-x0)/image.width, (y1-y0)/image.height)
    w, h = (int(dim*scale)//8*8 for dim in image.size)
    if min(w, h) < 32:
        raise ValueError("The detected left page is too small to illustrate.")
    left = x0 + (x1-x0-w)//16*8
    top = y0 + (y1-y0-h)//16*8
    return image.resize((w, h), Image.Resampling.LANCZOS), (left, top, left+w, top+h)


def plate_transmission(art):
    rgb = np.asarray(art, dtype=np.float32)/255
    brightness = rgb @ np.array([.299, .587, .114], np.float32)
    white = np.median(rgb[brightness >= np.quantile(brightness, .92)], axis=0)
    normalized = np.clip(rgb / np.maximum(white, .1), 0, 1)
    if np.quantile(np.ptp(normalized, axis=2), .95) < .10:
        tone = normalized @ np.array([.299, .587, .114], np.float32)
        normalized = tone[:, :, None]
    return .16 + .84*normalized


def locate_book(leaf, glyph):
    try:
        return page_geometry.resolve_joint_spread(leaf, glyph)
    except ValueError as scan_error:
        try:
            return page_geometry.resolve_spread(leaf, glyph)
        except ValueError as framed_error:
            raise ValueError(f"Could not locate the direct book: {scan_error} {framed_error}") from framed_error


def page_observations(leaf, glyph, illustration, native_mask, source_geometry, geometry, frame, size):
    """Sample the book and its printed illustration in the coordinates of the final photograph."""
    w, h = size
    sw, sh = leaf.size
    fx, fy, fr, fb = frame
    leaf = page_geometry.clean_left_stock(leaf, native_mask, source_geometry)
    left, upper, right, lower = source_geometry["right_page_box"]
    flip = Image.Transpose.FLIP_LEFT_RIGHT
    flipped_box = (sw-right, upper, sw-left, lower)
    leaf = page_geometry.clean_left_stock(leaf.transpose(flip), native_mask.transpose(flip),
                                          {"left_page_box": flipped_box}).transpose(flip)
    framed_leaf = leaf.resize((fr-fx, fb-fy), Image.Resampling.LANCZOS)
    art, ink_box = fit_plate(illustration, geometry["illustration_box"])
    x0, y0, x1, y1 = ink_box
    stock = framed_leaf.crop((x0-fx, y0-fy, x1-fx, y1-fy))
    transmission = plate_transmission(art)
    ah, aw = transmission.shape[:2]
    yy, xx = np.ogrid[:ah, :aw]
    border = np.minimum(np.minimum(yy, ah-1-yy), np.minimum(xx, aw-1-xx))
    opacity = np.clip(border/1.5, 0, 1)[..., None]
    transmission = 1 + (transmission-1)*opacity
    plate_rgb = np.asarray(stock, dtype=np.float32)/255 * transmission
    art_canvas = np.pad(np.uint8(np.rint(255*plate_rgb)), ((y0,h-y1),(x0,w-x1),(0,0)), mode="edge")
    art_rgb = np.asarray(art_canvas, dtype=np.float32)/255
    plate_mask = Image.new("L", size)
    plate_mask.paste(255, ink_box)

    pad = 32
    sx, sy = sw/(fr-fx), sh/(fb-fy)
    warp = (Image.Transform.PERSPECTIVE, (sx,0,pad-fx*sx,0,sy,pad-fy*sy,0,0), Image.Resampling.BICUBIC)
    padded = Image.fromarray(np.pad(np.asarray(leaf), ((pad,pad),(pad,pad),(0,0)), mode="edge"))
    page_rgb = np.asarray(padded.transform(size, *warp), dtype=np.float32)/255
    glyph_plane = Image.new("L", padded.size)
    glyph_plane.paste(glyph, (pad,pad))
    return page_rgb, art_rgb, plate_mask, glyph_plane.transform(size, *warp), ink_box


def run_fusion(*, model, sentence, seed, steps, guidance, background, illustration,
               book, glyph, book_latent, output, scene_strength):
    import diffusers
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from torch.nn import functional as F

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not sentence.strip() or steps < 1 or not 0 <= guidance <= 30 or not 0 < scene_strength <= 1:
        raise ValueError("Invalid spread sampling parameters.")
    paths = dict(background=background, illustration=illustration, book=book, glyph=glyph, book_latent=book_latent)
    hashes = {name: file_digest(path) for name, path in paths.items()}
    images = {}
    for name in ("background", "illustration", "book", "glyph"):
        with Image.open(paths[name]) as image:
            images[name] = image.convert("L" if name == "glyph" else "RGB")
    leaf = images["book"]
    sw, sh = leaf.size
    w, h = images["background"].size
    if any(n%8 for n in (w,h,sw,sh)) or images["glyph"].size != (sw,sh):
        raise ValueError("Incompatible scene, print condition or book dimensions.")
    native_z = load_file(str(book_latent))["latent"]
    if tuple(native_z.shape) != (1,4,sh//8,sw//8) or not torch.isfinite(native_z).all():
        raise ValueError("Invalid printed-book latent.")
    native_mask, _, source_geometry = locate_book(leaf, images["glyph"])
    mask, geometry, frame = page_geometry.frame_book(native_mask, source_geometry, (w,h))
    page_rgb, art_rgb, plate_mask, scene_glyph, ink_box = page_observations(
        leaf, images["glyph"], images["illustration"], native_mask, source_geometry, geometry, frame, (w,h))
    left, top, right, bottom = geometry["book_box"]
    shadow = page_geometry.book_shadow(mask, right-left, bottom-top)
    bg_rgb = np.asarray(images["background"], dtype=np.float32)/255
    backdrop = bg_rgb * (1 - np.asarray(shadow, dtype=np.float32)[:,:,None]/255)

    prompt = ("One open book viewed straight down, centered on a full-bleed photographic backdrop. "
              "The visible scene around all four edges shows: " + sentence +
              " Printed illustration on the left, a blank right page, a visible binding and a contact shadow.")
    negative = "empty tabletop, uniform studio background, picture frame, tilted book, overlapping books, hands, watermark"
    base_sha = verify_base(model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model), variant="fp16", torch_dtype=torch.float16, use_safetensors=True,
        local_files_only=True, add_watermarker=False).to("cuda")
    scheduler = pipe.scheduler
    if not isinstance(scheduler, EulerDiscreteScheduler) or scheduler.config.prediction_type != "epsilon":
        raise ValueError("The spread sampler requires the approved SDXL Euler scheduler.")
    pipe.vae.float().eval().requires_grad_(False)
    pipe.vae.enable_gradient_checkpointing()

    def encode(rgb):
        pixels = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.float32, device="cuda")
        encoded = pipe.vae.encode((pixels.permute(2,0,1)[None]*2-1).contiguous())
        return encoded.latent_dist.mode()*pipe.vae.config.scaling_factor

    def pool_mask(image):
        values = torch.tensor(np.asarray(image), device="cuda", dtype=torch.float32)[None,None]/255
        return F.avg_pool2d(values, 8)

    def condition_contact(current, outside, inside, matte, weight):
        # The encoder needs both sides of the material boundary in the same receptive field.
        alpha = np.asarray(matte, dtype=np.float32)[:,:,None]/255
        contact = encode(outside + alpha*(inside-outside))
        core = F.max_pool2d(weight,3,1,1) + F.max_pool2d(-weight,3,1,1)
        belt = F.avg_pool2d(F.max_pool2d(core,3,1,1),3,1,1,count_include_pad=False)
        return torch.lerp(current, contact, belt)

    with torch.inference_mode():
        conditioning, clip_windows = text_conditioning(pipe, prompt, negative_prompt=negative)
        pos, neg, pooled, neg_pooled = pipe.encode_prompt(**conditioning, device="cuda", do_classifier_free_guidance=True)
        embeddings = torch.cat([neg,pos])
        sizes = torch.tensor([[h,w,0,0,h,w]], dtype=pos.dtype, device="cuda").repeat(2,1)
        conditions = dict(text_embeds=torch.cat([neg_pooled,pooled]), time_ids=sizes)
        pipe.text_encoder.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()
        plate = pool_mask(plate_mask)
        cover = pool_mask(mask)
        paper_z = torch.lerp(encode(page_rgb), encode(art_rgb), plate)
        paper_z = condition_contact(paper_z, page_rgb, art_rgb, plate_mask, plate)
        scene = encode(backdrop)
        reference = condition_contact(torch.lerp(scene,paper_z,cover), backdrop, page_rgb, mask, cover)

        scheduler.set_timesteps(steps, device="cuda")
        first = steps - max(1, int(steps*scene_strength))
        scheduler.set_begin_index(first)
        noise = torch.randn(reference.shape, generator=torch.Generator("cpu").manual_seed(seed)).to("cuda")
        z = reference + float(scheduler.sigmas[first])*noise
        exterior = 1 - F.max_pool2d(cover, 7, stride=1, padding=3)
        exterior = F.avg_pool2d(exterior, 5, stride=1, padding=2, count_include_pad=False)
        guided_updates = max(1, round((steps-first)*.65))
        for i in range(first, steps):
            t = scheduler.timesteps[i]
            model_input = scheduler.scale_model_input(torch.cat([z,z]), t).to(pos.dtype)
            eps0, eps1 = pipe.unet(model_input, t, encoder_hidden_states=embeddings,
                                   added_cond_kwargs=conditions, return_dict=False)[0].float().chunk(2)
            z = scheduler.step(eps0+guidance*(eps1-eps0), t, z).prev_sample
            if i-first < guided_updates and i+1 < steps:
                weight = .55 * (1 - (i-first)/guided_updates)
                next_backdrop = scene + float(scheduler.sigmas[i+1])*noise
                z = torch.lerp(z, next_backdrop, weight*exterior)
        aligned_glyph, registration = print_page.register_scene_page(pipe.vae, z, geometry, scene_glyph)
        if aligned_glyph is None:
            raise ValueError(f"Could not place the sentence on the right page: {registration['reason']}")
        z = print_page.guide_scene_ink(pipe.vae, z, aligned_glyph, page_registration=registration)
        rgb = pipe.vae.decode(z/pipe.vae.config.scaling_factor, return_dict=False)[0]
        if not torch.isfinite(rgb).all():
            raise FloatingPointError("Nonfinite final spread.")
        pixels = ((rgb[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()

    exterior_fraction = float((1-np.asarray(mask, dtype=np.float32)/255).mean())
    table_info = dict(surface="generated_full_frame_scene", surface_source="background",
                      projection_matrix=np.eye(3).tolist(), unprojected_geometry=geometry.copy(),
                      unprojected_illustration_box=list(ink_box), background_weight_fraction=exterior_fraction,
                      exposed_top_fraction=exterior_fraction, boundary_condition="paired_surface_encoder")
    geometry["book_area_fraction"] = float(np.asarray(mask).mean()/255)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        Image.fromarray(pixels).save(stage/"final.png")
        mask.save(stage/"silhouette.png")
        shadow.save(stage/"contact_shadow.png")
        Image.new("L", (w,h), 255).save(stage/"table_mask.png")
        geometry_hash = file_digest(page_geometry.__file__)
        record = dict(
            sentence=sentence, seed=seed, steps=steps, guidance=guidance, prompt=prompt,
            input_sha256=hashes, base_manifest_sha256=base_sha, code_sha256=file_digest(__file__),
            geometry=geometry, plate_box=list(geometry["illustration_box"]), illustration_box=list(ink_box),
            width=w, height=h, scheduler=type(scheduler).__name__, first_step=first, clip_windows=clip_windows,
            reverse_updates=steps-first, start_sigma=float(scheduler.sigmas[first]), paper_iterations=0,
            final_sha256=file_digest(stage/"final.png"), silhouette_sha256=file_digest(stage/"silhouette.png"),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), torch_version=torch.__version__,
            diffusers_version=diffusers.__version__, source_geometry=source_geometry, source_frame_box=list(frame),
            framing_code_sha256=geometry_hash, table=table_info, table_code_sha256=geometry_hash,
            table_mask_sha256=file_digest(stage/"table_mask.png"), generate_board=True,
            contact_shadow_sha256=file_digest(stage/"contact_shadow.png"), scene_strength=scene_strength,
            scene_method="global_img2img_euler", negative_prompt=negative, regional_lock=False, separate_ink=True,
            geometry_role="input_layout", scene_layout="flatlay", background_sampler=SCENE_SAMPLER,
            exterior_guidance=dict(initial_weight=.55, updates=min(guided_updates,steps-first-1)),
            ink_applied=True, page_registration=registration, print_guide_sha256=file_digest(print_page.__file__),
            initial_glyph_box=list(scene_glyph.getbbox()), projected_glyph_box=registration["registered_glyph_box"])
        (stage/"record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        stage.rename(output)
    return record
