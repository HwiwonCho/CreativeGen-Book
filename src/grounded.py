"""Keep the generated desk, book and contact shadow together during scene sampling."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def support_weight(book, silhouette):
    """Return a canvas-space matte for the near tabletop and the book."""
    if book.size != silhouette.size or min(book.size) < 256:
        raise ValueError("Book and silhouette require the same canvas of at least 256 pixels.")
    w, h = book.size
    alpha = np.asarray(silhouette.convert("L"), dtype=np.float32) / 255
    solid = alpha > .5
    columns = np.flatnonzero(solid.any(axis=0))
    if len(columns) < .24*w:
        raise ValueError("No broad book surface in the framing mask.")
    back = float(np.percentile(np.argmax(solid[:,columns], axis=0), 10))
    if back < .16*h:
        raise ValueError("Too little generated surface above the book for this composition.")

    # Search above the book. Requiring contrast across most of the image width
    # rejects a candidate desk edge confined to one side.
    radius = max(2, round(h/150))
    rgb = np.asarray(book.convert("RGB").filter(ImageFilter.GaussianBlur(radius/2)), dtype=np.float32)
    first, last = max(round(.10*h), int(back-.30*h)), int(back-2*radius)
    rows = np.arange(first, last+1)
    delta = np.linalg.norm(rgb[rows+radius] - rgb[rows-radius], axis=2)
    bands = [np.median(c,axis=1) for c in np.array_split(delta, 6, axis=1)]
    scores = np.percentile(bands, 25, axis=0)
    index = int(scores.argmax())
    strength = float(scores[index])
    if strength >= 16:
        horizon = float(rows[index])
        mode = "native_horizontal_transition"
    else:
        # A generated overhead desk can cover the entire canvas. Establish the
        # far edge of the near surface while retaining its original shadow.
        horizon = back - .07*h
        mode = "layout_above_book"
    half_width = max(4, round(.02*h))
    t = np.clip((np.arange(h) - horizon + half_width) / (2*half_width), 0, 1)
    desk = (t*t*(3-2*t)).astype(np.float32)[:,None]
    weight = np.maximum(alpha,desk)
    return Image.fromarray(np.uint8(np.rint(weight*255))), {
        "desk_boundary_y": round(horizon), "boundary_source": mode,
        "native_edge_strength": round(strength,2), "blend_half_width_px": half_width,
        "background_weight_fraction": float(1-weight.mean())}


def board_masks(silhouette, geometry):
    """Return tabletop, bevel and book-shadow opacity in final canvas coordinates."""
    book = silhouette.convert("L")
    w, h = book.size
    solid_box = book.point(lambda value: 255 if value >= 128 else 0).getbbox()
    if solid_box is None:
        raise ValueError("Cannot place a tabletop beneath an empty book mask.")
    x0, y0, x1, y1 = geometry["book_box"]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Book geometry must be in final canvas coordinates.")
    x0, y0 = min(x0, solid_box[0]), min(y0, solid_box[1])
    x1, y1 = max(x1, solid_box[2]), max(y1, solid_box[3])
    bw, bh = x1-x0, y1-y0
    far_y = max(0, round(y0 - .065*bh))
    flare = max(2, round(.12*bw))
    # Limit the sides on tightly framed books to retain the distant scene.
    available = .60*w*h/(h-far_y) - bw - flare
    side = round(min(.065*bw, max(.022*bw, available/2)))
    flare = max(flare, round((.94*w-bw)/2-side))
    thickness = max(2, round(.018*bh))
    near_y = max(y1, h - round(.65*thickness))
    corners = [(x0-side, far_y), (x1+side, far_y),
               (x1+side+flare, near_y), (x0-side-flare, near_y)]

    scale = 2
    top_hi = Image.new("L", (w*scale, h*scale))
    upper = [(round(x*scale), round(y*scale)) for x, y in corners]
    ImageDraw.Draw(top_hi).polygon(upper, fill=255)
    full_hi = top_hi.copy()
    lower = [(x+max(1, thickness//4)*scale, y+thickness*scale) for x, y in upper]
    draw = ImageDraw.Draw(full_hi)
    for i in range(4):
        j = (i+1) % 4
        draw.polygon([upper[i], upper[j], lower[j], lower[i]], fill=255)
    top = top_hi.resize(book.size, Image.Resampling.BOX)
    full = full_hi.resize(book.size, Image.Resampling.BOX)
    edge = Image.fromarray(np.asarray(full) - np.asarray(top))

    shadow = book_shadow(book, top, bw, bh)
    alpha = np.asarray(book, dtype=np.float32)/255
    top_alpha = np.asarray(top, dtype=np.float32)/255
    return top, edge, shadow, {
        "board_top_corners": corners,
        "board_thickness_px": thickness,
        "background_weight_fraction": round(1-float(np.asarray(full).mean())/255, 5),
        "exposed_top_fraction": round(float((top_alpha*(1-alpha)).mean()), 5),
    }


def book_shadow(book, surface, bw, bh):
    alpha = np.asarray(book, dtype=np.float32)/255
    top_alpha = np.asarray(surface, dtype=np.float32)/255
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
    shadow = (1-light) * top_alpha * (1-alpha)
    shadow[alpha >= .5] = 0
    return Image.fromarray(np.uint8(np.rint(255*shadow)))


def board_projection(geometry, canvas_size):
    """One plane-to-image mapping for the tabletop, both pages and their ink."""
    w, h = canvas_size
    x0, y0, x1, y1 = geometry["book_box"]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Book geometry must be in final canvas coordinates.")
    cx, cy = w/2, (y0+y1)/2
    half_height = (y1-y0)/2
    rear_scale = .84
    perspective = (1-rear_scale) / ((1+rear_scale)*half_height)
    # Keep the full source and output canvases on the visible side of the horizon.
    perspective = min(perspective, .75/max(cy, h-cy, 1))
    depth = perspective*half_height
    plane = np.array([[1-depth, 0., 0.],
                      [0., .975*(1-depth**2), 0.],
                      [0., -perspective, 1.]], dtype=np.float64)
    center = np.array([[1., 0., cx], [0., 1., cy], [0., 0., 1.]])
    forward = center @ plane @ np.linalg.inv(center)
    inverse = np.linalg.inv(forward)
    inverse /= inverse[2, 2]
    return forward, tuple(float(n) for n in inverse.flat[:8])


def project_point(x, y, forward):
    px, py, depth = forward @ (x, y, 1.)
    if depth <= 0:
        raise ValueError("Cannot project a point behind the camera.")
    return float(px/depth), float(py/depth)


def project_box(box, forward):
    """Axis-aligned half-open bounding box of a projected rectangle."""
    l, t, r, b = box
    xs, ys = zip(*(project_point(x, y, forward) for x, y in
                   [(l, t), (r, t), (r, b), (l, b)]))
    return [int(np.floor(min(xs)+1e-9)), int(np.floor(min(ys)+1e-9)),
            int(np.ceil(max(xs)-1e-9)), int(np.ceil(max(ys)-1e-9))]
