"""Matched full-spread and line-detail training views of the photographed print."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def line_crop(box, width, height):
    left, top, right, bottom = box
    h = min(height, max(6 * (bottom-top), round(.24 * height)))
    w = min(width, max(round(1.40 * (right-left)), round(1.4 * h)))
    x = max(0, min((left+right-w)//2, width-w))
    y = max(0, min((top+bottom-h)//2, height-h))
    return (x, y, x+w, y+h)


def training_views(splits, annotation_path):
    annotations = {}
    for line in Path(annotation_path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["image"] in annotations:
                raise ValueError(f"Duplicate annotation for {row['image']}")
            annotations[row["image"]] = row
    originals = [row for rows in splits.values() for row in rows]
    if set(annotations) != {r["image"] for r in originals}:
        raise ValueError("Annotations must account for every source book in the split.")
    train_lines = {" ".join(item["text"].casefold().split())
                   for row in splits["train"] for item in annotations[row["image"]]["lines"]}
    result = {key: [] for key in splits}
    for split, rows in splits.items():
        for row in rows:
            annot = annotations[row["image"]]
            if any(annot[k] != row[k] for k in ("split", "group_id", "sha256")):
                raise ValueError(f"Source scan changed for {row['image']}")
            with Image.open(row["path"]) as source:
                w, h = source.size
            if annot["image_size"] != [w, h]:
                raise ValueError(f"Annotation canvas mismatch for {row['image']}")
            lines = annot["lines"]
            if not lines:
                continue
            for item in lines:
                x0,y0,x1,y1 = item["box"]
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and item["text"].strip()):
                    raise ValueError(f"Invalid annotated line for {row['image']}")
                if item.get("words") and " ".join(w["text"] for w in item["words"]) != item["text"]:
                    raise ValueError(f"Word transcript differs from line in {row['image']}")
                for word in item.get("words", []):
                    wx0,wy0,wx1,wy1 = word["box"]
                    if not (x0<=wx0<wx1<=x1 and y0<=wy0<wy1<=y1 and word["text"].strip()):
                        raise ValueError(f"Invalid word bounds for {row['image']}")
            result[split].append({**row, "caption": " ".join(x["text"] for x in lines),
                                  "lines": lines, "view": "spread", "crop": (0,0,w,h)})
            for item in lines:
                if split == "validation" and " ".join(item["text"].casefold().split()) in train_lines:
                    continue
                result[split].append({**row, "caption": item["text"], "lines": [item],
                                      "view": "detail", "crop": line_crop(item["box"], w, h)})
        if not result[split]:
            raise ValueError(f"Empty usable {split} annotations")
    return result


def line_hint(size, lines, font_path):
    mask = Image.new("L", size)
    font = ImageFont.truetype(str(font_path), 96)
    for line in lines:
        for word in line.get("words") or [line]:
            text, box = word["text"], word["box"]
            l,t,r,b = font.getbbox(text)
            ink = Image.new("L", (max(1,r-l), max(1,b-t)))
            ImageDraw.Draw(ink).text((-l,-t), text, font=font, fill=255)
            left,top,right,bottom = box
            mask.paste(ink.resize((right-left,bottom-top), Image.Resampling.LANCZOS), (left,top))
    return mask
