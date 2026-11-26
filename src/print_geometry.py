import numpy as np
from PIL import Image


def register_print(blank, glyph, geometry, silhouette, font_size=42):
    """Lay the typeset paragraph onto the measured surface of the right leaf."""
    glyph = glyph.convert("L")
    if blank.size != glyph.size or silhouette.size != glyph.size:
        raise ValueError("The page, silhouette and type must have the same size.")
    bounds = glyph.getbbox()
    if not bounds:
        return glyph
    left, top, right, bottom = geometry["right_page_box"]
    page_width, page_height = right-left, bottom-top
    if not (0 <= left < right <= blank.width and 0 <= top < bottom <= blank.height):
        raise ValueError("The right-page outline is outside the generated book.")

    paragraph = glyph.crop(bounds)
    # Preserve small fonts already selected to fit longer passages.
    body_size = .065 * page_height
    scale = min(1., body_size / max(1, font_size),
                .75 * page_width / paragraph.width, .60 * page_height / paragraph.height)
    size = tuple(max(1, round(d * scale)) for d in paragraph.size)
    paragraph = paragraph.resize(size, Image.Resampling.LANCZOS)
    width, height = paragraph.size
    center_x = (left + right) / 2
    _, art_top, _, art_bottom = geometry["illustration_box"]
    center_y = (art_top + art_bottom) / 2
    x0, y0 = round(center_x - width / 2), round(center_y - height / 2)

    inside = np.asarray(silhouette.convert("L"))[:, left:right] >= 128
    valid = inside.any(axis=0)
    if not valid.any():
        raise ValueError("The right page has no visible surface.")
    upper = inside.argmax(axis=0)
    lower = blank.height - 1 - inside[::-1].argmax(axis=0)
    fraction = np.clip((center_y - top) / page_height, 0., 1.)
    baseline = (1 - fraction) * upper + fraction * lower
    columns = np.arange(page_width)
    baseline = np.interp(columns, columns[valid], baseline[valid])
    radius = max(2, round(.04 * page_width))
    window = np.hanning(2 * radius + 1)
    baseline = np.convolve(np.pad(baseline, radius, mode="edge"), window / window.sum(), mode="valid")
    displacement = baseline[x0-left:x0-left+width] - np.interp(center_x-left, columns, baseline)
    displacement = np.clip(displacement, -.025*page_height, .025*page_height)
    pad = int(np.ceil(abs(displacement).max())) + 2
    pixels = np.asarray(paragraph, dtype=np.float32)
    yy = np.arange(-pad, height+pad)[:, None] - displacement
    iy = np.floor(yy).astype(int)
    frac = yy - iy
    pixels = np.pad(pixels, ((1,1),(0,0)))
    xx = np.arange(width)
    warped = ((1-frac)*pixels[np.clip(iy+1,0,height+1),xx]
              + frac*pixels[np.clip(iy+2,0,height+1),xx])
    result = Image.new("L", glyph.size)
    result.paste(Image.fromarray(warped.round().astype(np.uint8)), (x0,y0-pad))
    if not np.all(np.asarray(silhouette)[np.asarray(result)>8] >= 128):
        raise ValueError("The curved type crosses the detected page boundary.")
    return result
