"""Locate the two leaves of a flat, upright generated book for regional diffusion."""
import numpy as np
from PIL import Image, ImageFilter


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


def resolve_spread(blank, glyph, *, require_blank_left=True):
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
    if require_blank_left and left_ink > .006:
        raise ValueError("Detected print or clutter on the expected blank left leaf.")
    metadata = {"book_box": [left,top,right+1,bottom+1], "gutter_x": gutter,
                "left_page_box": [left,top,gutter,bottom+1], "right_page_box": [gutter,top,right+1,bottom+1],
                "illustration_box": list(box), "glyph_box": list(glyph_box),
                "left_dark_detail_fraction": left_ink, "book_area_fraction": float(inside.mean())}
    return silhouette, box, metadata
