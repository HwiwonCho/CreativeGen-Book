"""Locate and frame the generated pages, then estimate their paper and shadow."""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def clean_left_stock(book, silhouette, geometry):
    """Estimate the left leaf's unprinted paper from the generated book's exposed stock."""
    if book.size != silhouette.size:
        raise ValueError("The book and its silhouette must have matching dimensions.")
    left, top, right, bottom = geometry["left_page_box"]
    if not (0 <= left < right <= book.width and 0 <= top < bottom <= book.height):
        raise ValueError("The left page must lie inside the book canvas.")
    original = np.asarray(book.convert("RGB")).copy()
    leaf = original[top:bottom,left:right].astype(np.float32)
    h, w = leaf.shape[:2]
    solid = np.asarray(silhouette.convert("L"))[top:bottom,left:right] >= 240
    distance = ndimage.distance_transform_edt(np.pad(solid,1))[1:-1,1:-1]
    inset = max(2, round(.018*min(w,h)))
    usable = distance > inset
    if not usable.any():
        raise ValueError("No exposed page interior for estimating paper stock.")

    def profile(pixels, valid, axis, quantile, smooth):
        # A scan's remaining margin supplies the paper value even where a
        # single row or column is almost entirely occupied by an old engraving.
        if axis == 1:
            pixels, valid = pixels.swapaxes(0,1), valid.T
        present = valid.any(axis=0)
        values = np.where(valid[:,present,None], pixels[:,present], np.nan)
        values = np.nanquantile(values, quantile, axis=0)
        coordinates = np.arange(len(present))
        filled = np.stack([np.interp(coordinates, coordinates[present], c) for c in values.T], axis=1)
        return ndimage.gaussian_filter1d(filled, smooth, axis=0, mode="nearest")

    y, x = np.ogrid[:h,:w]
    caps = usable & ((y < .095*h) | (y > .93*h))
    sides = usable & ((x < .13*w) | (x > .952*w))
    columns = profile(leaf, caps if caps.any() else usable, 0, .67, max(1.5,.014*w))
    rows = profile(leaf-columns[None], sides if sides.any() else usable, 1, .67, max(1.5,.012*h))
    stock = columns[None] + rows[:,None]
    # Keep the fine grain where the source actually resembles this stock.
    # Averaging the residual of an engraved region would carry the plate through.
    grain = leaf - ndimage.gaussian_filter(leaf,(.9,.9,0))
    match = np.clip(1-np.abs(leaf-stock).mean(axis=2)/10, 0, 1)
    stock += np.clip(grain, -3, 3)*match[:,:,None]
    weight = np.clip((distance-0.6*inset)/(1.4*inset),0,1)[:,:,None]
    original[top:bottom,left:right] = np.rint(np.clip(leaf + weight*(stock-leaf),0,255)).astype(np.uint8)
    return Image.fromarray(original)


def resolve_joint_spread(book, glyph):
    if book.size != glyph.size or min(book.size) < 256:
        raise ValueError("A joint book and its glyph hint must share a canvas of at least 256px.")
    w, h = book.size
    glyph_box = glyph.convert("L").getbbox()
    if glyph_box is None:
        raise ValueError("The joint book has no right-page text condition.")
    gray = np.asarray(book.convert("L").filter(ImageFilter.GaussianBlur(max(1, w/600))), dtype=np.float32)
    # The surrounding dark surface remains visible along at least one horizontal
    # edge of these full-bleed scans. Interior dark ink belongs to the paper.
    paper = np.quantile(gray[::4, ::4], .65)
    cutoff = .57 * paper
    margin = max(3, h//40)
    top_edge = (gray[0] < cutoff) & (gray[2*margin] > cutoff)
    bottom_edge = (gray[-1] < cutoff) & (gray[-2*margin] > cutoff)
    if paper < 75 or max(top_edge.mean(), bottom_edge.mean()) < .65:
        raise ValueError("No extended exterior-to-paper transition on the full-bleed book.")
    # A fold has a sustained, signed transverse luminance change along its spine.
    d = max(2, w//192)
    cross = gray[h//12:-h//12, 2*d:] - gray[h//12:-h//12, :-2*d]
    strips = np.stack([np.median(strip, axis=0) for strip in np.array_split(cross, 15)])
    strength = np.maximum(np.quantile(strips, .2, axis=0), -np.quantile(strips, .8, axis=0))
    candidates = np.arange(round(.33*w), min(round(.67*w), glyph_box[0]))
    if candidates.size == 0:
        raise ValueError("No central fold to the left of the requested print.")
    gutter = int(candidates[np.argmax(strength[candidates-d])])
    confidence = float(strength[gutter-d])
    if confidence < max(8., .045*paper):
        raise ValueError("A sustained central book fold was not detected.")

    labels, count = ndimage.label(gray > cutoff)
    if not count:
        raise ValueError("No paper component in the joint book.")
    areas = np.bincount(labels.ravel())
    component = labels == (1 + np.argmax(areas[1:]))
    inside = ndimage.binary_fill_holes(component)
    if inside.mean() < .67:
        raise ValueError("Detected foreground does not fill enough of the book scan.")
    ys, xs = np.where(inside)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    if (not .3 < (gutter-left)/(right-left) < .7 or
            not inside[np.asarray(glyph.convert("L")) > 32].all()):
        raise ValueError("The fold and intended print do not lie on facing leaves.")
    mask = Image.fromarray(inside.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(max(.7, w/800)))
    side = int(min(.84*(gutter-left), .68*(bottom-top)))
    x = (left + gutter - side)//2
    y = (top + bottom - side)//2
    if side < .18*w or not inside[y:y+side, x:x+side].all():
        raise ValueError("Detected left page has too little surface for the illustration.")
    box = (x, y, x+side, y+side)
    roi = gray[y:y+side, x:x+side]
    dark_detail = float(np.mean(ndimage.gaussian_filter(roi, 3)-roi > 12))
    book_box = [left, top, right, bottom]
    metadata = dict(book_box=book_box, gutter_x=gutter, left_page_box=[left, top, gutter, bottom],
                    right_page_box=[gutter, top, right, bottom], illustration_box=list(box),
                    glyph_box=list(glyph_box), book_area_fraction=float(inside.mean()),
                    left_dark_detail_fraction=dark_detail, fold_luma_contrast_p20=confidence,
                    method="full_bleed_dark_exterior_and_continuous_fold",
                    normalized_book_box=[left/w, top/h, right/w, bottom/h],
                    normalized_gutter_x=gutter/w)
    return mask, box, metadata


def _peak(profile, start, stop, outward, neighborhood):
    start, stop = max(1, start), min(len(profile)-1, stop)
    if stop <= start:
        raise ValueError("No search interval for the book edge.")
    peak = start + int(np.argmax(profile[start:stop]))
    strength = float(profile[peak])
    if strength < 10:
        raise ValueError("The page outline or binding is indistinct.")
    lo, hi = max(start, peak-neighborhood), min(stop, peak+neighborhood+1)
    strong = np.flatnonzero(profile[lo:hi] >= .55*strength) + lo
    return int(strong[0] if outward < 0 else strong[-1] if outward > 0 else peak)


def _trace(edge, center, radius, outer):
    lo, hi = max(0, center-radius), min(edge.shape[0], center+radius+1)
    candidates = edge[lo:hi]
    threshold = np.maximum(10, .40*candidates.max(axis=0))
    strong = candidates >= threshold
    path = lo + np.argmax(strong, axis=0) if outer < 0 else hi-1-np.argmax(strong[::-1], axis=0)
    path[~strong.any(axis=0)] = center
    # Remove isolated marks and gaps while retaining broad curvature of the leaf.
    half_width = max(5, len(path)//36)
    median = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(path, half_width, mode="edge"), 2*half_width+1), axis=1)
    return np.convolve(np.pad(median, 2, mode="edge"), np.ones(5)/5, mode="valid")


def resolve_spread(blank, glyph):
    """Return a soft book silhouette, (left, top, right, bottom) art box and diagnostics."""
    if blank.size != glyph.size or min(blank.size) < 256:
        raise ValueError("The generated book and its glyph condition must share a canvas of at least 256px.")
    glyph_box = glyph.convert("L").getbbox()
    if glyph_box is None:
        raise ValueError("The book needs a nonempty right-page glyph condition.")
    w, h = blank.size
    gx0, gy0, gx1, gy1 = glyph_box
    smoothed = blank.convert("RGB").filter(ImageFilter.GaussianBlur(max(1., min(w,h)/450)))
    rgb = np.asarray(smoothed, dtype=np.float32)
    stride = max(1, round(min(w,h)/380))
    dx = np.zeros((h,w), dtype=np.float32)
    dy = np.zeros_like(dx)
    dx[:,stride:-stride] = np.linalg.norm(rgb[:,2*stride:] - rgb[:,:-2*stride], axis=2)
    dy[stride:-stride,:] = np.linalg.norm(rgb[2*stride:,:] - rgb[:-2*stride,:], axis=2)
    y = (gy0+gy1)//2
    band = max(16, h//14)
    x_profile = dx[max(0,y-band):min(h,y+band)].mean(axis=0)
    left = _peak(x_profile, 1, int(.38*w), -1, w//32)
    right = _peak(x_profile, max(gx1, int(.62*w)), w-1, 1, w//32)
    bw = right-left
    if bw < .40*w:
        raise ValueError("The generated spread is too narrow.")
    y_profile = dy[:,left+bw//10:right-bw//10].mean(axis=1)
    top = _peak(y_profile, 1, gy0, -1, h//32)
    bottom = _peak(y_profile, gy1, h-1, 1, h//32)
    bh = bottom-top
    if not (.24*h < bh and 1.05 < bw/bh < 2.7):
        raise ValueError("A flat, horizontally opened book was not found.")
    fold_profile = dx[top+bh//6:bottom-bh//6].mean(axis=0)
    gutter = _peak(fold_profile, left+int(.35*bw), min(left+int(.65*bw), gx0), 0, 0)
    if not (gutter < gx0 and gx1 < right and top < gy0 < gy1 < bottom):
        raise ValueError("The text condition does not land on the right leaf.")

    yy, xx = np.ogrid[:h,:w]
    radius_x = max(5, w//32)
    radius_y = max(5, h//30)
    upper, lower = (_trace(dy, top, radius_y, -1), _trace(dy, bottom, radius_y, 1))
    lhs, rhs = (_trace(dx.T, left, radius_x, -1), _trace(dx.T, right, radius_x, 1))
    inside = (yy >= upper) & (yy <= lower) & (xx >= lhs[:,None]) & (xx <= rhs[:,None])
    silhouette = Image.fromarray(np.uint8(inside)*255).filter(ImageFilter.GaussianBlur(max(1., w/450)))
    glyph_pixels = np.asarray(glyph.convert("L")) > 32
    if not np.all(inside[glyph_pixels]):
        raise ValueError("The printed strokes extend beyond the detected book.")

    side = int(min(.76*(gutter-left), .68*bh))
    cx, cy = (left+gutter)//2, (top+bottom)//2
    x0, y0 = cx-side//2, cy-side//2
    box = (x0, y0, x0+side, y0+side)
    if side < .15*w or not inside[y0:y0+side,x0:x0+side].all():
        raise ValueError("The detected left leaf has insufficient illustration space.")
    gray = np.asarray(blank.convert("L"), dtype=np.float32)
    paper = np.asarray(blank.convert("L").filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)
    left_ink = float(np.mean((paper-gray)[y0:y0+side,x0:x0+side] > 22))
    metadata = {"book_box": [left,top,right+1,bottom+1], "gutter_x": gutter,
                "left_page_box": [left,top,gutter,bottom+1], "right_page_box": [gutter,top,right+1,bottom+1],
                "illustration_box": list(box), "glyph_box": list(glyph_box),
                "left_dark_detail_fraction": left_ink, "book_area_fraction": float(inside.mean())}
    return silhouette, box, metadata


def frame_book(mask, source, canvas_size):
    """Reserve visible space for the generated scene around a full-bleed scan."""
    w, h = mask.size
    cw, ch = canvas_size
    l,t,r,b = source["book_box"]
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError("The detected book must lie inside its source canvas.")
    scale = min(.69*cw/(r-l), .80*ch/(b-t))
    # Prefer an exact aspect ratio on the VAE grid when it fits the layout.
    divisor = int(np.gcd(w,h))
    grid_scale = 8*max(1, round(divisor*scale/8))/divisor
    if abs(grid_scale-scale) <= .02*scale:
        scale = grid_scale
    fw, fh = round(w*scale), round(h*scale)
    # A native scan can have unequal margins. Centre the visible spread.
    x = 8*round((cw-(l+r)*fw/w)/16)
    y = 8*round((ch-(t+b)*fh/h)/16)
    frame = (x, y, x+fw, y+fh)
    dest = Image.new("L", (cw,ch))
    dest.paste(mask.resize((fw,fh), Image.Resampling.LANCZOS), (x,y))
    def canvas_box(box):
        l,t,r,b = box
        return [x+round(l*fw/w), y+round(t*fh/h), x+round(r*fw/w), y+round(b*fh/h)]
    geo = {key:value for key,value in source.items() if not key.startswith("normalized_")}
    for name in ("book_box","left_page_box","right_page_box","glyph_box","illustration_box"):
        geo[name] = canvas_box(source[name])
    geo["gutter_x"] = x + round(source["gutter_x"]*fw/w)
    geo["book_area_fraction"] = float(np.asarray(dest,dtype=np.float32).mean()/255)
    return dest, geo, frame


def book_shadow(book, bw, bh):
    alpha = np.asarray(book, dtype=np.float32)/255
    light = np.ones_like(alpha)
    # Close occlusion along the cover, followed by the wider offset cast shadow.
    for dx, dy, radius, strength in [
        (.002*bw, .008*bh, .006*bh, .48),
        (.013*bw, .026*bh, .017*bh, .33),
    ]:
        shifted = Image.new("L", book.size)
        shifted.paste(book, (round(dx), round(dy)))
        blurred = shifted.filter(ImageFilter.GaussianBlur(max(.5, radius)))
        light *= 1 - strength*np.asarray(blurred, dtype=np.float32)/255
    shadow = (1-light) * (1-alpha)
    shadow[alpha >= .5] = 0
    return Image.fromarray(np.uint8(np.rint(255*shadow)))
