"""Sentence and layout conditions for directly generated printed books."""
from dataclasses import asdict, replace

from PIL import Image

from . import book


def build_prompt(sentence, *, board=False):
    if not sentence.strip():
        raise ValueError("Supply the text to print.")
    if board:
        return "An open book seen from above, spread flat on a table. The text printed on the right-hand page reads: " + sentence
    return "An open book seen from above. The text printed on the right-hand page reads: " + sentence


def board_conditions(sentence, font, size, layout_path=None):
    unit = 8 * round(size * 7 / 192)
    width, height = 4 * unit, 3 * unit
    offset = ((width - size) // 2, (height - size) // 2)
    if layout_path is None:
        def scaled(values):
            return tuple(round(value * size / 768) for value in values)
        layout = book.PageLayout(size, scaled((384, 237, 680, 630)),
                                 scaled((440, 347, 652, 495)), *scaled((30, 5)))
        min_font = max(1, round(18 * size / 768))
    else:
        layout = book.PageLayout.from_json(layout_path)
        if layout.canvas_size != size:
            raise ValueError("Layout canvas differs from --size.")
        min_font = layout.font_size
    while True:
        try:
            source = book.render_masks(sentence, font, layout)
            break
        except ValueError as exc:
            width_limited = str(exc).startswith("The sentence does not fit")
            if not width_limited and not str(exc).startswith("The text is too tall"):
                raise
            if layout.font_size <= min_font:
                if layout_path is not None or not width_limited:
                    raise
                left, _, right, _ = layout.right_page_box
                inset = round((right-left)*.08)
                _, top, _, bottom = layout.print_box
                layout = replace(layout, print_box=(left+inset, top, right-inset, bottom))
                source = book.render_masks(sentence, font, layout)
                break
            layout = replace(layout, font_size=max(min_font, layout.font_size - 2))
    masks = []
    for mask in source:
        left, top, right, bottom = mask.getbbox()
        if not (0 <= left + offset[0] < right + offset[0] <= width and
                0 <= top + offset[1] < bottom + offset[1] <= height):
            raise ValueError("The book condition falls outside the native landscape frame.")
        frame = Image.new("L", (width, height))
        frame.paste(mask, offset)
        masks.append(frame)
    layout_data = asdict(layout)
    for field in ("right_page_box", "print_box"):
        layout_data[field] = list(layout_data[field])
    framing = dict(size=[width, height], native_width=width, native_height=height,
                   layout_offset=list(offset), layout=layout_data)
    return masks, framing
