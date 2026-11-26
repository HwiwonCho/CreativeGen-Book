"""Measure the sampled right page and fit its printed strokes in the SDXL latent."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from .geometry import clean_left_stock


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


def fit_ink(vae, clean, glyph, ink_target):
    """Use the full decoded page to fit local ink contrast without a pixel overlay."""
    import torch
    from torch.nn import functional as F

    iterations, lr, ink_window = 20, .016, 15
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
        initial = torch.zeros_like(original[...,y0:y1,x0:x1])
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


def register_scene_page(vae, latent, geometry, glyph_image):
    """Register the lettering to the measured right leaf of the sampled scene."""
    import torch

    _, _, h, w = latent.shape
    size = (w*8,h*8)
    if glyph_image.size != size:
        raise ValueError("The projected glyph and sampled scene have different sizes.")
    bbox = glyph_image.point(lambda x: 255 if x>=8 else 0).getbbox()
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
    rectified = Image.fromarray(pixels)
    rgb = np.asarray(rectified.filter(ImageFilter.GaussianBlur(1.2*unit)),dtype=np.float32)
    step = max(1,round(2*unit))
    dx = np.linalg.norm(rgb[:,2*step:]-rgb[:,:-2*step],axis=-1)
    dy = np.linalg.norm(rgb[2*step:]-rgb[:-2*step],axis=-1).T
    fold, top, right, bottom = geometry["right_page_box"]
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
    # Give a tall block of prose the page area it needs at the final image resolution.
    prose = (y1-y0) > .18*(bottom-top)
    width_fraction, height_fraction = (.84, .76) if prose else (.72, .55)
    scale = min(1.45, width_fraction*width/(x1-x0), height_fraction*height/(y1-y0))
    half_w, half_h = .5*scale*(x1-x0), .5*scale*(y1-y0)
    baseline_slope = (1-fraction)*top_slope
    if min(width/2-half_w,y-positions["top"]-half_h-abs(top_slope-baseline_slope)*half_w,
           positions["bottom"]-y-half_h-abs(baseline_slope)*half_w)<margin:
        return None, {**report,"reason":"glyph too close to measured page boundary"}
    layout = np.array([[scale,0,x-scale*cx],[scale*baseline_slope,scale,y-scale*(cy+baseline_slope*cx)],[0,0,1]])
    pullback = np.diag([2.,2.,1.]) @ np.linalg.inv(layout)
    pullback /= pullback[2,2]
    enlarged = glyph_image.convert("L").resize((size[0]*2,size[1]*2),Image.Resampling.LANCZOS)
    aligned = enlarged.transform(size,Image.Transform.PERSPECTIVE,tuple(pullback.flat[:8]),Image.Resampling.BICUBIC)
    aligned_box = aligned.point(lambda value: 255 if value>=8 else 0).getbbox()
    cx_page = (positions["fold"]+positions["right"])/2
    corners = np.array([(xp,yp + (xp-cx_page)*top_slope if yp==positions["top"] else yp,1.)
                        for xp,yp in [(positions["fold"],positions["top"]),(positions["right"],positions["top"]),
                                      (positions["right"],positions["bottom"]),(positions["fold"],positions["bottom"])]])
    return aligned, {**report,"passed":True,"reason":"four page boundaries supported",
                     "registered_glyph_box":aligned_box,"scale":scale,"baseline_slope":baseline_slope,
                     "right_page_corners":corners[:,:2].tolist()}


def guide_scene_ink(vae, latent, glyph_image, *, page_registration):
    """Fit printed strokes in the generated scene while keeping its lighting and page position."""
    import torch
    from torch.nn import functional as F

    _, _, h, w = latent.shape
    glyph_image = glyph_image.convert("L")
    if latent.shape[:2] != (1,4) or glyph_image.size != (8*w,8*h):
        raise ValueError("Scene lettering requires one SDXL latent and a registered full-resolution glyph mask.")
    bounds = glyph_image.point(lambda x: 255 if x >= 3 else 0).getbbox()
    if bounds is None:
        raise ValueError("The projected sentence contains no ink.")
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
            crop = scene[roi].contiguous()
            local_target = decode(crop) + (desired-observed)
        corrected = fit_ink(vae, crop, alpha, local_target)
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("Nonfinite scene lettering latent.")
        scene[roi] = corrected
        return scene.to(latent.dtype).detach()
