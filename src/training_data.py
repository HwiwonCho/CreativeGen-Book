"""Validate per-item training provenance and produce reproducible split records."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps, ImageChops


RIGHTS = {"CC0-1.0", "PDM-1.0", "Public Domain", "team-owned"}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_records(manifest, module):
    """Return verified train and validation rows for one generator."""
    manifest = Path(manifest).resolve(strict=True)
    splits = {"train": [], "validation": []}
    assigned_groups = {}
    sources = set()
    hashes = set()
    quote_splits = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["module"] != module:
            continue
        tag = f"{manifest.name}:{number}"
        for key in ("image", "caption", "split", "rights", "source_url", "group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{tag}: missing {key}")
        split = row["split"]
        if split not in splits or row["rights"] not in RIGHTS:
            raise ValueError(f"{tag}: unsupported split or rights declaration")
        if row["rights"] != "team-owned" and urlparse(row["source_url"]).scheme not in {"http", "https"}:
            raise ValueError(f"{tag}: external works require a source URL")
        group = row["group_id"]
        if assigned_groups.setdefault(group, split) != split:
            raise ValueError(f"{tag}: source group {group!r} occurs in different splits")
        relative = Path(row["image"])
        if relative.is_absolute():
            raise ValueError(f"{tag}: image paths must be relative to the manifest")
        image = (manifest.parent / relative).resolve(strict=True)
        with Image.open(image) as sample:
            if module == "book":
                image_size = ImageOps.exif_transpose(sample).size
            else:
                sample.verify()
        if module == "book":
            if row["rights"] != "team-owned" and urlparse(row.get("rights_url", "")).scheme not in {"http", "https"}:
                raise ValueError(f"{tag}: external book scans require a linked rights statement")
            if quote_splits.setdefault(row["caption"], split) != split:
                raise ValueError(f"{tag}: the same printed text crosses splits")
            masks = []
            for key in ("glyph_mask", "page_mask"):
                if not row.get(key) or Path(row[key]).is_absolute():
                    raise ValueError(f"{tag}: require relative {key} aligned to the oriented photo")
                path = (manifest.parent / row[key]).resolve(strict=True)
                with Image.open(path) as src:
                    mask = src.convert("L")
                if mask.size != image_size or mask.getbbox() is None:
                    raise ValueError(f"{tag}: empty or misaligned {key}")
                mask_digest = digest(path)
                if row.get(key+"_sha256", mask_digest) != mask_digest:
                    raise ValueError(f"{tag}: changed {key}")
                row[key+"_sha256"] = mask_digest
                masks.append(mask)
            if ImageChops.multiply(masks[0], ImageOps.invert(masks[1])).point(lambda x: 255 if x > 20 else 0).getbbox():
                raise ValueError(f"{tag}: glyph pixels outside right-page mask")
        checksum = digest(image)
        if row.get("sha256", checksum) != checksum:
            raise ValueError(f"{tag}: file differs from recorded sha256")
        # A volume landing page can identify many different plates. Detect
        # exact duplicates by bytes and also check canonical local paths.
        if image in sources or checksum in hashes:
            raise ValueError(f"{tag}: duplicate image")
        sources.add(image)
        hashes.add(checksum)
        splits[split].append({**row, "path": image, "sha256": checksum})
    if any(not rows for rows in splits.values()):
        raise ValueError(f"{module}: training requires nonempty train and validation groups")
    return splits
