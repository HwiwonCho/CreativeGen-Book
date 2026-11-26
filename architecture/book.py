"""Prepare the SDXL prompt and two grayscale conditioning masks for an open book."""

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class BookCondition:
    scene_prompt: str
    glyph_text: str


@dataclass(frozen=True)
class PageLayout:
    """Team-authored page and print rectangles, in half-open canvas pixels."""

    canvas_size: int
    right_page_box: tuple[int, int, int, int]
    print_box: tuple[int, int, int, int]
    font_size: int
    line_spacing: int

    @classmethod
    def from_json(cls, path: str | Path) -> "PageLayout":
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(**values)

    def __post_init__(self):
        if type(self.canvas_size) is not int or self.canvas_size < 256 or self.canvas_size % 8:
            raise ValueError("The square canvas size must be at least 256 and divisible by eight.")
        if type(self.font_size) is not int or not 1 <= self.font_size <= self.canvas_size:
            raise ValueError("Choose a positive font_size that fits the canvas.")
        if type(self.line_spacing) is not int or self.line_spacing < 0:
            raise ValueError("line_spacing must be a nonnegative number of pixels.")
        parent = (0, 0, self.canvas_size, self.canvas_size)
        for name in ("right_page_box", "print_box"):
            box = getattr(self, name)
            if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError(f"{name} must be four integer pixel coordinates.")
            left, top, right, bottom = box
            if not (parent[0] <= left < right <= parent[2] and parent[1] <= top < bottom <= parent[3]):
                raise ValueError(f"{name} must be a nonempty rectangle inside its parent.")
            object.__setattr__(self, name, tuple(box))
            parent = box


def build_condition(sentence: str) -> BookCondition:
    if not sentence.strip():
        raise ValueError("A nonempty sentence is required.")
    return BookCondition(
        scene_prompt=(
            "Overhead photograph of an open book with a blank left page "
            "and a short passage printed on the right page."
        ),
        glyph_text=sentence,
    )


def _lines(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    def fits(line: str) -> bool:
        left, _, right, _ = font.getbbox(line)
        return max(font.getlength(line), right) - min(left, 0) <= width

    explicit = text.split("\n")
    if len(explicit) <= 2 and all(fits(line) for line in explicit):
        return explicit
    if len(explicit) == 1:
        candidates = []
        for index, char in enumerate(text):
            if char != " ":
                continue
            first, second = text[:index], text[index + 1:]
            if first.strip() and second.strip() and fits(first) and fits(second):
                imbalance = abs(font.getlength(first) - font.getlength(second))
                candidates.append((imbalance, first, second))
        if candidates:
            _, first, second = min(candidates, key=lambda entry: entry[0])
            return [first, second]
    raise ValueError("The sentence does not fit the two-line print area at this font size.")


def render_masks(
    sentence: str, font_path: str | Path, layout: PageLayout | str | Path
) -> tuple[Image.Image, Image.Image]:
    """Return (glyph, right page) masks, in that order, as square PIL L images.

    Inference requires the team's authored layout and a licensed font file.
    Training photographs require separately aligned masks.
    """
    text = build_condition(sentence).glyph_text
    if any(not char.isprintable() and char != "\n" for char in text):
        raise ValueError("Use printable text and optional line breaks.")
    if not isinstance(layout, PageLayout):
        layout = PageLayout.from_json(layout)
    font_file = Path(font_path)
    if not font_file.is_file():
        raise FileNotFoundError(font_file)
    font = ImageFont.truetype(str(font_file), layout.font_size)
    size = layout.canvas_size
    page = Image.new("L", (size, size))
    glyph = Image.new("L", (size, size))
    x0, y0, x1, y1 = layout.right_page_box
    ImageDraw.Draw(page).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)

    x0, y0, x1, y1 = layout.print_box
    lines = _lines(text, font, x1 - x0)
    ascent, descent = font.getmetrics()
    spacing = ascent + descent + layout.line_spacing
    height = ascent + descent + (len(lines) - 1) * spacing
    if height > y1 - y0:
        raise ValueError("The text is too tall for the authored print_box.")
    baseline = (y0 + y1 - height) // 2 + ascent
    draw = ImageDraw.Draw(glyph)
    for index, line in enumerate(lines):
        bearing = min(0, font.getbbox(line)[0])
        position = (x0 - bearing, baseline + index * spacing)
        left, top, right, bottom = draw.textbbox(position, line, font=font, anchor="ls")
        if not (x0 <= left <= right <= x1 and y0 <= top <= bottom <= y1):
            raise ValueError("The glyph bounds exceed the authored print_box.")
        draw.text(position, line, font=font, fill=255, anchor="ls")
    if glyph.getbbox() is None:
        raise ValueError("The chosen font produced no visible ink.")
    return glyph, page
