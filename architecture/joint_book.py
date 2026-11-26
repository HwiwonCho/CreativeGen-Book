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


def _layouts(size, layout_path):
    if layout_path is not None:
        layout = book.PageLayout.from_json(layout_path)
        if layout.canvas_size != size:
            raise ValueError("Layout canvas differs from --size.")
        yield layout, False
        yield layout, True
        return

    def scaled(values):
        return tuple(round(value * size / 768) for value in values)

    layout = book.PageLayout(size, scaled((384, 237, 680, 630)),
                             scaled((440, 347, 652, 495)), *scaled((30, 5)))
    min_font = max(1, round(18 * size / 768))
    while True:
        yield layout, False
        if layout.font_size <= min_font:
            break
        layout = replace(layout, font_size=max(min_font, layout.font_size - 2))
    left, top, right, bottom = layout.right_page_box
    side_margin = round((right-left)*.08)
    x0, x1 = left + side_margin, right - side_margin
    yield replace(layout, print_box=(x0, layout.print_box[1], x1, layout.print_box[3])), False

    end_margin = round((bottom-top)*.08)
    paragraph_box = (x0, top + end_margin, x1, bottom - end_margin)
    for font_size in range(round(26*size/768), max(1, round(12*size/768)) - 1, -1):
        yield replace(layout, print_box=paragraph_box, font_size=font_size,
                      line_spacing=max(1, round(font_size*.12))), True


def board_conditions(sentence, font, size, layout_path=None):
    unit = 8 * round(size * 7 / 192)
    width, height = 4 * unit, 3 * unit
    offset = ((width - size) // 2, (height - size) // 2)
    for layout, multiline in _layouts(size, layout_path):
        try:
            source = book.render_masks(sentence, font, layout, multiline=multiline)
            break
        except book.TextDoesNotFit:
            continue
    else:
        raise book.TextDoesNotFit("The text does not fit on the right-hand page at the smallest print size.")
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
