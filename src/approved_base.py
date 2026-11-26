"""Verify the local SDXL Base snapshot used by training and inference."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_base(model_path):
    root = Path(model_path).resolve(strict=True)
    reference = Path(__file__).resolve().parents[1] / "provenance" / "sdxl-base-files.json"
    snapshot = json.loads(reference.read_text(encoding="utf-8"))
    for relative, expected in snapshot["files"].items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"Model file differs from the pinned SDXL Base snapshot: {relative}")
    return file_digest(reference)
