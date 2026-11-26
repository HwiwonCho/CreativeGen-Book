"""Verify the local SDXL Base snapshot used by training and inference."""
import hashlib
import json
from pathlib import Path


BASE_REFERENCE = Path(__file__).resolve().parents[1] / "configs/sdxl-base-files.json"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_base(model_path):
    root = Path(model_path).resolve(strict=True)
    snapshot = json.loads(BASE_REFERENCE.read_text(encoding="utf-8"))
    for relative, expected in snapshot["files"].items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"Model file differs from the pinned SDXL Base snapshot: {relative}")
    return file_digest(BASE_REFERENCE)
