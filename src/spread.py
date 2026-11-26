"""Sample a regional spread or redraw one scene from the generated stages."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter

if __package__:
    from .approved_base import file_digest, verify_base
    from . import background as background_model
    from .joint_framing import frame_book, resolve_joint_spread
    from .print_page import paper_ink
    from .spread_geometry import resolve_spread
else:
    from approved_base import file_digest, verify_base
    import background as background_model
    from joint_framing import frame_book, resolve_joint_spread
    from print_page import paper_ink
    from spread_geometry import resolve_spread


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
    return image.resize((w,h), Image.Resampling.LANCZOS), (left,top,left+w,top+h)


def plate_transmission(art):
    rgb = np.asarray(art, dtype=np.float32)/255
    brightness = rgb @ np.array([.299,.587,.114], np.float32)
    white = np.median(rgb[brightness >= np.quantile(brightness,.92)], axis=0)
    normalized = np.clip(rgb / np.maximum(white,.1), 0, 1)
    if np.quantile(np.ptp(normalized,axis=2),.95) < .10:
        tone = normalized @ np.array([.299,.587,.114], np.float32)
        normalized = tone[:,:,None]
    return .16 + .84*normalized


def match_paper(vae, latent, blank, printed, glyph, iterations, registered=False):
    """Constrain the last clean estimate against the two generated page observations."""
    import torch
    from torch.nn import functional as F

    with torch.inference_mode(False), torch.enable_grad():
        z = latent.detach().float().clone()
        def pixels(image):
            return torch.from_numpy(np.array(image)).to(z).permute(2,0,1)[None]/255

        left, top, right, bottom = glyph.getbbox()
        x0, y0 = max(0,left//8-5), max(0,top//8-5)
        x1, y1 = min(z.shape[3],(right+7)//8+5), min(z.shape[2],(bottom+7)//8+5)
        crop = (8*x0,8*y0,8*x1,8*y1)
        paper = pixels(blank.crop(crop))
        alpha = torch.from_numpy(np.array(glyph.crop(crop))).to(z)[None,None]/255
        ink = paper_ink(paper, alpha) if registered else pixels(printed.crop(crop))
        strokes = F.max_pool2d(alpha,3,1,1)
        clear = 1-strokes
        if registered:
            # Fit the immediate surrounds as well as the strokes. A small bright
            # fringe occupies too few pixels to influence a page-wide average.
            clear = clear * (1 + 4 * F.max_pool2d(strokes, 13, 1, 6))
            unprinted = (1 - F.max_pool2d(alpha, 5, 1, 2)) * F.max_pool2d(strokes, 41, 1, 20)
        support = F.avg_pool2d(F.max_pool2d(alpha,57,1,28),8)
        delta = torch.zeros_like(z[...,y0:y1,x0:x1],requires_grad=True)
        opt = torch.optim.Adam([delta],lr=.035)
        pad = (x0,z.shape[3]-x1,y0,z.shape[2]-y1)
        for i in range(iterations):
            opt.param_groups[0]["lr"] = .004 + .031*(1+math.cos(math.pi*i/iterations))/2
            current = z + F.pad(delta*support,pad)
            rgb = vae.decode(current/vae.config.scaling_factor,return_dict=False)[0]
            rgb = (rgb[...,8*y0:8*y1,8*x0:8*x1] + 1)/2
            paper_error = rgb-paper
            tint = paper_error-paper_error.mean(1,keepdim=True)
            clear_loss = ((paper_error.abs() + tint.abs())*clear).sum()/(3*clear.sum().clamp_min(1))
            ink_loss = ((rgb-ink).abs()*strokes).sum()/(3*strokes.sum().clamp_min(1))
            loss = clear_loss + ink_loss + .002*delta.square().mean()
            if registered:
                # Retain the smoothness of bare paper between letters. Isolated
                # creases from the decoder are visible even at small amplitudes.
                residual = paper_error.abs().mean(1, keepdim=True) + tint.abs().mean(1, keepdim=True)
                local_peak = F.max_pool2d(residual*unprinted, 7, 3, 3)
                coverage = F.max_pool2d(unprinted, 7, 3, 3).sum().clamp_min(1)
                loss = loss + 3 * clear_loss + 2 * local_peak.sum()/coverage
            (delta.grad,) = torch.autograd.grad(loss,delta)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return (z + F.pad(delta.detach()*support,pad)).detach()


def board_material(pipe, size, seed, edge, shadow):
    import torch
    prompt = ("Close-up photograph of the satin surface of a single warm brown oak board, "
              "fine horizontal wood grain, soft diffuse light, wood fills the entire frame.")
    negative = "objects, lettering, furniture, tools, book, gaps, seams, planks, frame"
    material_seed = (seed ^ 81214) % (2**63)
    latent = pipe(prompt=prompt, negative_prompt=negative, width=640, height=448,
                  num_inference_steps=24, guidance_scale=5.0, output_type="latent",
                  generator=torch.Generator("cpu").manual_seed(material_seed)).images
    decoded = pipe.vae.decode(latent.float()/pipe.vae.config.scaling_factor, return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Nonfinite generated wood material.")
    texture = ((decoded[0].clamp(-1,1).permute(1,2,0)+1)*127.5).round().byte().cpu().numpy()
    rgb = np.asarray(Image.fromarray(texture).resize(size, Image.Resampling.LANCZOS), dtype=np.float32)/255
    rgb = .50*rgb + .50*np.median(rgb.reshape(-1,3),axis=0)
    face = np.asarray(edge, dtype=np.float32)/255
    shade = np.asarray(shadow, dtype=np.float32)/255
    # Darken just the front bevel and the wood shadowed by the cover.
    rgb *= (1-.34*face)[:,:,None]*(1-shade)[:,:,None]
    return rgb, dict(material_prompt=prompt, material_negative_prompt=negative,
                     material_seed=material_seed, material_guidance=5.0, material_steps=24,
                     material_resolution=[640,448])


def run_fusion(*, model, sentence, seed, steps, guidance, background, illustration,
               book, glyph, book_latent, output, blank=None, layout_glyph=None, keep_table=False,
               generate_board=False, scene_strength=None, separate_ink=False, scene_layout="board",
               background_sampler=None):
    import diffusers
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from torch.nn import functional as F

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not sentence.strip() or steps < 1 or not (0 <= guidance <= 30 and math.isfinite(guidance)):
        raise ValueError("Invalid spread sampling parameters.")
    flatlay = scene_layout == "flatlay"
    scenic = flatlay and background_sampler in background_model.SCENE_SAMPLERS
    if scene_layout not in ("board", "table", "flatlay") or (scene_layout != "board" and scene_strength is None):
        raise ValueError("The selected surface layout requires global scene sampling.")
    if scene_strength is not None:
        if not generate_board or not math.isfinite(scene_strength) or not 0 < scene_strength <= 1:
            raise ValueError("Global scene sampling requires board inputs and a strength in (0, 1].")
    elif separate_ink:
        raise ValueError("Separate ink requires global scene sampling.")
    paths = dict(background=background, illustration=illustration, book=book,
                 glyph=glyph, book_latent=book_latent)
    if blank is not None:
        paths["blank"] = blank
    if layout_glyph is not None:
        paths["layout_glyph"] = layout_glyph
    hashes = {name:file_digest(path) for name,path in paths.items()}
    images = {}
    for name in ("background", "illustration", "book", "glyph"):
        with Image.open(paths[name]) as image:
            images[name] = image.convert("L" if name == "glyph" else "RGB")
    leaf = images["book"]
    if blank is not None:
        with Image.open(blank) as image:
            leaf = image.convert("RGB")
    sw,sh = images["book"].size
    w,h = images["background"].size
    if (any(n%8 for n in (w,h,sw,sh)) or leaf.size != (sw,sh) or images["glyph"].size != (sw,sh)
            or (not flatlay and images["illustration"].size != (w,h)) or (blank is not None and (sw,sh) != (w,h))):
        raise ValueError("Incompatible scene, print condition or book dimensions.")
    if layout_glyph is None:
        anchor = images["glyph"]
    else:
        with Image.open(layout_glyph) as image:
            anchor = image.convert("L")
    frame = (0, 0, w, h)
    direct = blank is None
    if direct:
        try:
            native, _, source_geometry = resolve_joint_spread(leaf, anchor)
        except ValueError as scan_error:
            try:
                native, _, source_geometry = resolve_spread(leaf, anchor, require_blank_left=False)
            except ValueError as framed_error:
                raise ValueError(f"Could not locate the direct book: {scan_error} {framed_error}") from framed_error
        mask, geometry, frame = frame_book(native, source_geometry,(w,h),overhead=flatlay)
        plate_box = geometry["illustration_box"]
    else:
        mask, plate_box, geometry = resolve_spread(leaf, anchor)
    if layout_glyph is not None:
        geometry["layout_glyph_box"] = geometry["glyph_box"]
        geometry["glyph_box"] = list(images["glyph"].getbbox())
    art, ink_box = fit_plate(images["illustration"], plate_box)
    table_mask = table_info = contact_shadow = None
    if generate_board:
        if not direct or keep_table:
            raise ValueError("A generated board requires a directly printed book and its own support mask.")
        if __package__:
            from .grounded import board_masks, board_projection, book_shadow, project_box, project_point
            from .joint_framing import clean_left_stock
        else:
            from grounded import board_masks, board_projection, book_shadow, project_box, project_point
            from joint_framing import clean_left_stock
        leaf = clean_left_stock(leaf, native, source_geometry)
        if separate_ink:
            left, upper, right, lower = source_geometry["right_page_box"]
            flip = Image.Transpose.FLIP_LEFT_RIGHT
            flipped_box = (sw-right, upper, sw-left, lower)
            leaf = clean_left_stock(leaf.transpose(flip), native.transpose(flip),
                                    {"left_page_box": flipped_box}).transpose(flip)
        if flatlay:
            table_mask = Image.new("L",(w,h),255)
            left, upper, right, lower = geometry["book_box"]
            contact_shadow = book_shadow(mask,table_mask,right-left,lower-upper)
            forward = np.eye(3)
            inverse = tuple(forward.flat[:8])
            surface = "generated_full_frame_scene" if scenic else "generated_overhead_still_life"
            table_info = dict(surface=surface, surface_source="background")
        else:
            top, edge, contact_shadow, table_info = board_masks(mask, geometry)
            forward, inverse = board_projection(geometry,(w,h))
        def project(image):
            return image.transform((w,h), Image.Transform.PERSPECTIVE, inverse, Image.Resampling.BICUBIC)
        def project_rgb(pixels):
            return np.asarray(project(Image.fromarray(pixels)), dtype=np.float32)/255
        if scene_layout == "table":
            rear_native = table_info["board_top_corners"][0][1]
            rear_y = project_point(w/2, rear_native, forward)[1]
            transition = max(4, round(.055*h))
            t = np.clip((np.arange(h,dtype=np.float32)+.5-rear_y)/transition+.5,0,1)
            coverage = np.rint(255*t*t*(3-2*t)).astype(np.uint8)
            table_mask = Image.fromarray(np.repeat(coverage[:,None], w, axis=1))
            edge = Image.new("L",(w,h))
            table_info = dict(surface="full_width_table", back_edge_y=rear_y,
                              depth_transition_px=transition)
        elif not flatlay:
            table_mask = project(Image.fromarray(np.asarray(top) + np.asarray(edge)))
        table_info.update(projection_matrix=forward.tolist(), unprojected_geometry=geometry.copy(),
                          unprojected_illustration_box=list(ink_box))
    if keep_table:
        if not direct or frame != (0,0,w,h):
            raise ValueError("Tabletop composition requires a directly generated book surrounded by its own surface.")
        if __package__:
            from .grounded import support_weight
        else:
            from grounded import support_weight
        table_mask, table_info = support_weight(leaf, mask)
    book_z = load_file(str(book_latent))["latent"].float()
    if tuple(book_z.shape) != (1,4,sh//8,sw//8) or not torch.isfinite(book_z).all():
        raise ValueError("Invalid printed-book latent.")
    prompt = "A photograph of an open illustrated book in the foreground of this scene. " + sentence
    negative = None
    if scene_strength is not None:
        page_description = "a blank right page" if separate_ink else "printed type on the right page"
        prompt = (sentence + " An environmental photograph. An open book rests on a broad wooden table "
                  "in the foreground. A printed illustration on the left page, " + page_description
                  + ". The table extends towards the camera. Natural light falls across the scene, "
                  "curved pages, binding and wood, with shadows underneath the book.")
        negative = "collage, layered cutouts, pasted objects, floating book, picture frame, split screen"
        if flatlay:
            prompt = ("Overhead still life of one open book resting flat on a surface. Straight-down camera, parallel pages. "
                      "The book occupies the center, a few related objects rest around its outer edges. "
                      "Illustrated left page, " + page_description + ". " + sentence)
            negative = "horizon, skyline in the background, lectern, angled camera, upright book, overlapping books, hands, fingers, collage, watermark"
        if scenic:
            prompt = ("One open book viewed straight down, centered on a full-bleed photographic backdrop. "
                      "The visible scene around all four edges shows: " + sentence
                      + " Printed illustration on the left, " + page_description + ", a visible binding and a contact shadow.")
            negative = "empty tabletop, uniform studio background, picture frame, tilted book, overlapping books, hands, watermark"
    base_sha = verify_base(model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model), variant="fp16", torch_dtype=torch.float16, use_safetensors=True,
        local_files_only=True, add_watermarker=False).to("cuda")
    if not isinstance(pipe.scheduler, EulerDiscreteScheduler) or pipe.scheduler.config.prediction_type != "epsilon":
        raise ValueError("The spread sampler requires the approved SDXL Euler scheduler.")
    pipe.vae.float().eval().requires_grad_(False)
    pipe.vae.enable_gradient_checkpointing()

    def encode(rgb):
        pixels = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.float32, device="cuda")
        return pipe.vae.encode((pixels.permute(2,0,1)[None]*2-1).contiguous()).latent_dist.mode()*pipe.vae.config.scaling_factor

    def pool_mask(image):
        values = torch.tensor(np.asarray(image), device="cuda", dtype=torch.float32)[None,None]/255
        return F.avg_pool2d(values, 8)

    def condition_contact(current, outside, inside, matte, weight):
        # The nonlinear VAE needs to see both sides of an edge within the same
        # receptive field. Use this encoding only along that material boundary.
        alpha = np.asarray(matte, dtype=np.float32)[:,:,None]/255
        contact_z = encode(outside + alpha*(inside-outside))
        core = F.max_pool2d(weight,3,1,1) + F.max_pool2d(-weight,3,1,1)
        belt = F.avg_pool2d(F.max_pool2d(core,3,1,1),3,1,1,count_include_pad=False)
        return torch.lerp(current,contact_z,belt), belt

    with torch.inference_mode():
        if generate_board and not flatlay:
            material, material_info = board_material(pipe, (w,h), seed, edge, contact_shadow)
            table_info.update(material_info)
        pos, neg, pooled, neg_pooled = pipe.encode_prompt(prompt,negative_prompt=negative,device="cuda",do_classifier_free_guidance=True)
        embeddings = torch.cat([neg,pos])
        sizes = torch.tensor([[h,w,0,0,h,w]], dtype=pos.dtype, device="cuda").repeat(2,1)
        conditions = dict(text_embeds=torch.cat([neg_pooled,pooled]), time_ids=sizes)
        pipe.text_encoder.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()
        bg_rgb = np.asarray(images["background"],dtype=np.float32)/255
        bg_z = encode(bg_rgb)
        book_z = book_z.to("cuda")
        fx, fy, fr, fb = frame
        framed_leaf = leaf
        if frame != (0,0,w,h) or (sw,sh) != (w,h):
            # Re-encode the model's own decoded page at its printed size. Direct
            # interpolation of latent cells loses narrow strokes and counters.
            framed_leaf = leaf.resize((fr-fx,fb-fy),Image.Resampling.LANCZOS)
            if not generate_board:
                small = encode(np.asarray(framed_leaf,dtype=np.float32)/255)
                book_z = F.pad(small,(fx//8,(w-fr)//8,fy//8,(h-fb)//8),mode="replicate")
        x0,y0,x1,y1 = ink_box
        paper_crop = framed_leaf.crop((x0-fx,y0-fy,x1-fx,y1-fy))
        transmission = plate_transmission(art)
        if direct:
            # Estimate the stock under the plate without transferring the old
            # printed lines into the new illustration.
            if not generate_board:
                radius = max(2, round(6*framed_leaf.width/sw))
                paper_crop = paper_crop.filter(ImageFilter.MedianFilter(2*radius+1))
            ah, aw = transmission.shape[:2]
            yy, xx = np.ogrid[:ah,:aw]
            border = np.minimum(np.minimum(yy,ah-1-yy),np.minimum(xx,aw-1-xx))
            fade = 1.5 if generate_board else max(2,.04*min(aw,ah))
            opacity = np.clip(border/fade, 0, 1)[...,None]
            transmission = 1 + (transmission-1)*opacity
        plate_rgb = np.asarray(paper_crop,dtype=np.float32)/255 * transmission
        if generate_board:
            # Project each generated observation in pixels before encoding. A
            # bilinear warp of the latent smears small type near the binding.
            # Sample native print directly at its photographed position.
            pad = 32
            sx, sy = sw/(fr-fx), sh/(fb-fy)
            to_page = np.array([[sx,0,pad-fx*sx],[0,sy,pad-fy*sy],[0,0,1]]) @ np.linalg.inv(forward)
            to_page /= to_page[2,2]
            padded = Image.fromarray(np.pad(np.asarray(leaf),((pad,pad),(pad,pad),(0,0)),mode="edge"))
            page = padded.transform((w,h),Image.Transform.PERSPECTIVE,tuple(to_page.flat[:8]),Image.Resampling.BICUBIC)
            if separate_ink:
                glyph_plane = Image.new("L", padded.size)
                glyph_plane.paste(images["glyph"], (pad,pad))
                scene_glyph = glyph_plane.transform((w,h),Image.Transform.PERSPECTIVE,tuple(to_page.flat[:8]),
                                                    Image.Resampling.BICUBIC)
            page_rgb = np.asarray(page,dtype=np.float32)/255
            book_z = encode(page_rgb)
            art_canvas = np.pad(np.uint8(np.rint(255*plate_rgb)), ((y0,h-y1),(x0,w-x1),(0,0)), mode="edge")
            art_rgb = project_rgb(art_canvas)
            illustration_z = encode(art_rgb)
            plate_mask = Image.new("L",(w,h))
            plate_mask.paste(255,(x0,y0,x1,y1))
            plate_mask = project(plate_mask)
            plate = pool_mask(plate_mask)
            mask = project(mask)
        else:
            margin = 32
            patch = np.pad(plate_rgb,((margin,margin),(margin,margin),(0,0)),mode="reflect")
            plate_latent = encode(patch)[:,:,margin//8:-margin//8,margin//8:-margin//8]
            illustration_z = book_z.clone()
            region = (...,slice(y0//8,y1//8),slice(x0//8,x1//8))
            illustration_z[region] = plate_latent
            plate = torch.zeros_like(book_z[:,:1])
            plate[region] = 1
        if not generate_board:
            plate = F.avg_pool2d(plate,3,stride=1,padding=1)
        cover = pool_mask(mask)
        paper_z = torch.lerp(book_z, illustration_z, plate)
        if generate_board:
            paper_z, plate_belt = condition_contact(paper_z,page_rgb,art_rgb,plate_mask,plate)
            if flatlay:
                material = bg_rgb*(1-np.asarray(contact_shadow,dtype=np.float32)[:,:,None]/255)
                top_projected = table_mask
            elif scene_layout == "table":
                from scipy.ndimage import map_coordinates
                # The plane continues past the texture frame on either side.
                yy, xx = np.ogrid[:h,:w]
                xx, yy = xx+.5, yy+.5
                ia, ib, ic, id_, ie, iff, ig, ih = inverse
                depth = ig*xx + ih*yy + 1
                plane_x = (ia*xx + ib*yy + ic)/depth - .5
                plane_y = (id_*xx + ie*yy + iff)/depth - .5
                material = np.stack([map_coordinates(channel,(plane_y,plane_x),order=1,mode="nearest")
                                     for channel in material.transpose(2,0,1)],axis=2)
                top_projected = table_mask
            else:
                material = project_rgb(np.uint8(np.rint(255*material)))
                top_projected = project(top)
                table_info["projected_top_corners"] = [list(project_point(x,y,forward)) for x,y in table_info["board_top_corners"]]
            tabletop = pool_mask(table_mask)
            if flatlay:
                scene = encode(material)
            else:
                scene = torch.lerp(bg_z,encode(material),tabletop)
                scene, table_belt = condition_contact(scene,bg_rgb,material,table_mask,tabletop)
            foreground = cover
            contact_shadow = project(contact_shadow)
            table_info["background_weight_fraction"] = float(1-np.asarray(table_mask).mean()/255)
            table_info["exposed_top_fraction"] = float((np.asarray(top_projected,dtype=np.float32)/255 *
                                                       (1-np.asarray(mask,dtype=np.float32)/255)).mean())
            if flatlay:
                table_info["background_weight_fraction"] = table_info["exposed_top_fraction"]
            geometry = geometry.copy()
            for field in ("book_box","left_page_box","right_page_box","illustration_box","glyph_box"):
                geometry[field] = project_box(geometry[field],forward)
            geometry["book_area_fraction"] = float(np.asarray(mask).mean()/255)
            mid_y = (geometry["book_box"][1]+geometry["book_box"][3])/2
            geometry["gutter_x"] = round(project_point(geometry["gutter_x"],mid_y,forward)[0])
            plate_box = project_box(plate_box,forward)
            ink_box = project_box(ink_box,forward)
        else:
            scene = bg_z
            foreground = cover if table_mask is None else pool_mask(table_mask)
        reference = torch.lerp(scene,paper_z,foreground)
        if generate_board:
            reference, book_belt = condition_contact(reference,material,page_rgb,mask,cover)
            table_info["boundary_condition"] = "paired_surface_encoder"
        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps,device="cuda")
        if scene_strength is None:
            boundary = F.max_pool2d(foreground,5,1,2) + F.max_pool2d(-foreground,5,1,2)
            if generate_board:
                board_edge = F.max_pool2d(tabletop,5,1,2) + F.max_pool2d(-tabletop,5,1,2)
                boundary = torch.maximum(boundary, board_edge)
            plate_edge = F.max_pool2d(plate,3,1,1) + F.max_pool2d(-plate,3,1,1)
            lock = .77*(1-boundary.clamp(0,1))*(1-.7*plate_edge.clamp(0,1))
            if generate_board:
                contact = torch.maximum(plate_belt,torch.maximum(book_belt,table_belt))
                lock = torch.maximum(lock,.96*contact)
            if direct:
                right_leaf = -F.max_pool2d(-cover,5,1,2)
                right_leaf[...,:math.ceil(geometry["gutter_x"]/8)] = 0
                lock = torch.maximum(lock,right_leaf)
            else:
                text_zone = F.max_pool2d((pool_mask(images["glyph"])>.001).float(),9,1,4)
                lock = torch.lerp(lock, torch.ones_like(lock), text_zone)
            first = int((scheduler.sigmas[:-1]-.68).abs().argmin())
        else:
            first = steps - max(1, int(steps * scene_strength))
        if scenic:
            # Keep the photographed margins during the coarse updates. Leave the
            # book and its contact shadow free, then finish with whole-frame diffusion.
            exterior = 1 - F.max_pool2d(cover, 7, stride=1, padding=3)
            exterior = F.avg_pool2d(exterior, 5, stride=1, padding=2, count_include_pad=False)
            guided_updates = max(1, round((steps-first)*.65))
        scheduler.set_begin_index(first)
        noise = torch.randn(reference.shape,generator=torch.Generator("cpu").manual_seed(seed)).to("cuda")
        z = reference + float(scheduler.sigmas[first])*noise
        paper_iterations = 0 if blank is None else (64 if layout_glyph is not None else 32)
        for i in range(first, steps):
            t = scheduler.timesteps[i]
            model_input = scheduler.scale_model_input(torch.cat([z,z]),t).to(pos.dtype)
            eps0, eps1 = pipe.unet(model_input,t,encoder_hidden_states=embeddings,
                added_cond_kwargs=conditions,return_dict=False)[0].float().chunk(2)
            z = scheduler.step(eps0+guidance*(eps1-eps0),t,z).prev_sample
            if scene_strength is None:
                next_reference = reference + float(scheduler.sigmas[i+1])*noise
                z = torch.lerp(z,next_reference,lock)
            elif scenic and i-first < guided_updates and i+1 < steps:
                weight = .55 * (1 - (i-first)/guided_updates)
                next_backdrop = scene + float(scheduler.sigmas[i+1])*noise
                z = torch.lerp(z,next_backdrop,weight*exterior)
            if i == steps-1 and paper_iterations:
                if torch.cuda.get_device_properties("cuda").total_memory < 18 * 1024**3:
                    pipe.unet.cpu()
                    torch.cuda.empty_cache()
                z = match_paper(pipe.vae,z,leaf,images["book"],images["glyph"],paper_iterations,layout_glyph is not None)
        if separate_ink:
            if __package__:
                from .print_page import guide_scene_ink, register_scene_page
            else:
                from print_page import guide_scene_ink, register_scene_page
            aligned_glyph, registration = register_scene_page(pipe.vae,z,geometry,table_info,scene_glyph)
            ink_applied = aligned_glyph is not None
            if flatlay and not ink_applied:
                raise ValueError(f"Could not place the sentence on the right page: {registration['reason']}")
            if ink_applied:
                if flatlay:
                    z = guide_scene_ink(pipe.vae,z,aligned_glyph,page_registration=registration)
                else:
                    z = guide_scene_ink(pipe.vae,z,aligned_glyph)
        rgb = pipe.vae.decode(z/pipe.vae.config.scaling_factor,return_dict=False)[0]
        if not torch.isfinite(rgb).all():
            raise FloatingPointError("Nonfinite final spread.")
        pixels = ((rgb[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()

    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        Image.fromarray(pixels).save(stage/"final.png")
        mask.save(stage/"silhouette.png")
        record = dict(sentence=sentence, seed=seed, steps=steps, guidance=guidance, prompt=prompt,
            input_sha256=hashes, base_manifest_sha256=base_sha, code_sha256=file_digest(__file__),
            geometry=geometry, plate_box=list(plate_box), illustration_box=list(ink_box),
            width=w, height=h, scheduler=type(scheduler).__name__, first_step=first,
            reverse_updates=steps-first, start_sigma=float(scheduler.sigmas[first]), paper_iterations=paper_iterations,
            final_sha256=file_digest(stage/"final.png"), silhouette_sha256=file_digest(stage/"silhouette.png"),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), torch_version=torch.__version__,
            diffusers_version=diffusers.__version__)
        if direct:
            record.update(source_geometry=source_geometry, source_frame_box=list(frame),
                          framing_code_sha256=file_digest(Path(__file__).with_name("joint_framing.py")))
        if table_mask is not None:
            table_mask.save(stage/"table_mask.png")
            record.update(table=table_info, table_mask_sha256=file_digest(stage/"table_mask.png"),
                          table_code_sha256=file_digest(Path(__file__).with_name("grounded.py")))
        if generate_board:
            contact_shadow.save(stage/"contact_shadow.png")
            record.update(generate_board=True, contact_shadow_sha256=file_digest(stage/"contact_shadow.png"))
        if scene_strength is not None:
            record.update(scene_strength=scene_strength, scene_method="global_img2img_euler",
                          negative_prompt=negative, regional_lock=False, separate_ink=separate_ink,
                          geometry_role="input_layout")
            if scene_layout != "board":
                record["scene_layout"] = scene_layout
            if scenic:
                record.update(background_sampler=background_sampler,
                              exterior_guidance=dict(initial_weight=.55, updates=min(guided_updates,steps-first-1)))
            if separate_ink:
                record.update(ink_applied=ink_applied, page_registration=registration)
                record["print_guide_sha256"] = file_digest(Path(__file__).with_name("print_page.py"))
                record["initial_glyph_box"] = list(scene_glyph.getbbox())
                if ink_applied:
                    record["projected_glyph_box"] = registration["registered_glyph_box"]
        (stage/"record.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        stage.rename(output)
    return record
