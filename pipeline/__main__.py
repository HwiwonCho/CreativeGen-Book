"""Generate an illustrated book from one English sentence with three trained SDXL adapters."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

from PIL import Image

from architecture import background, illustration, joint_book
from architecture.approved_base import BASE_REFERENCE, file_digest, verify_base


PROJECT = Path(__file__).resolve().parents[1]
MODULES = ("background", "illustration", "book")


def write_json(path: Path, value: dict) -> None:
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(folder: Path):
    with (folder / ".pipeline.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Could not lock output directory: {folder}") from exc
        yield


def make_plan(args: argparse.Namespace, saved: dict) -> tuple[dict, list[str]]:
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    base = json.loads(BASE_REFERENCE.read_text(encoding="utf-8"))
    base_hash = file_digest(BASE_REFERENCE)
    model = (args.model or Path(saved.get("model", PROJECT / config["model"]))).resolve()
    defaults = {"size": 768, "steps": 32, "guidance": 6.0, "scene_strength": .52}
    values = {k: saved.get(k, default) if getattr(args, k) is None else getattr(args, k)
              for k, default in defaults.items()}
    size = values["size"]
    saved_book = saved.get("stages", {}).get("book", {})
    book_seed = saved_book.get("seed", 241126) if args.book_seed is None else args.book_seed
    if not args.sentence.strip() or not all(0 <= s < 2**63 for s in (args.seed, book_seed)):
        raise ValueError("Supply a nonempty sentence and nonnegative 63-bit seeds.")
    if size < 256 or size % 8 or values["steps"] < 1 or not 1 <= values["guidance"] <= 30 or not 0 < values["scene_strength"] <= 1:
        raise ValueError("Expected size >=256 divisible by eight, positive steps, CFG 1–30 and scene strength in (0,1].")
    errors = []
    stages = {}
    for module in MODULES:
        module_config = config["modules"][module]
        adapter_name = "integrated_adapter" if module == "book" else "adapter"
        saved_stage = saved.get("stages", {}).get(module, {})
        adapter = Path(saved_stage.get("adapter", PROJECT / module_config[adapter_name])).resolve()
        token = hashlib.sha256(f"{args.seed}/{module}".encode()).digest()
        stage = {"adapter": str(adapter), "seed": int.from_bytes(token[:8], "big") % 2**63}
        stages[module] = stage
        try:
            settings = json.loads((adapter / "settings.json").read_text(encoding="utf-8"))
            expected = (module, base["model_id"], base["revision"], base_hash)
            if tuple(settings[k] for k in ("module", "base_id", "base_revision", "base_manifest_sha256")) != expected:
                raise ValueError("Incompatible module or base checkpoint.")
            stage["adapter_sha256"] = file_digest(adapter / "adapter.safetensors")
            if module == "background":
                stage.update(background.flatlay_settings(args.sentence, background.SCENE_SAMPLER, size))
            elif module == "illustration":
                stage["prompt"] = illustration.build_prompt(args.sentence)
            else:
                if settings.get("book_mode") != "joint_print":
                    raise ValueError("A directly trained joint-print book adapter is required.")
                font = Path(saved_stage.get("font", PROJECT / module_config["font"])).resolve()
                _, framing = joint_book.board_conditions(args.sentence, font, size)
                stage.update(seed=book_seed, prompt=joint_book.build_prompt(args.sentence, board=True),
                             book_mode="joint_print", book_composition="board",
                             conditioner_sha256=file_digest(adapter / "conditioner.safetensors"),
                             font=str(font), font_sha256=file_digest(font), page_layout=framing.pop("layout"),
                             **framing)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{module}: {exc}")
    try:
        verify_base(model)
    except (OSError, ValueError) as exc:
        errors.append(f"SDXL Base: {exc}")
    plan = dict(sentence=args.sentence, seed=args.seed, model=str(model), base_manifest_sha256=base_hash,
                **values, stages=stages, modules=list(MODULES), overview=False, spread=True, separate_ink=True,
                scene_layout="flatlay")
    return plan, errors


def verify_png(path: Path, dimensions: tuple, mode: str | None = None) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != dimensions or (mode is not None and image.mode != mode):
            raise ValueError(f"Unexpected image format or dimensions: {path}")
        image.verify()


def read_stage(folder: Path, module: str, plan: dict) -> dict:
    stage = plan["stages"][module]
    record_file = folder / "record.json"
    record = json.loads(record_file.read_text(encoding="utf-8"))
    expected = {key: plan[key] for key in ("sentence", "size", "steps", "guidance", "base_manifest_sha256")}
    expected.update(module=module, base_only=False)
    keys = ["seed", "prompt", "adapter_sha256"]
    if module == "background":
        keys += ["background_layout", "layout_sampler", "negative_prompt", "size", "prompt_2"]
    elif module == "book":
        keys += ["size", "conditioner_sha256", "font_sha256", "native_width", "native_height",
                 "layout_offset", "book_mode", "book_composition"]
        expected.update(layout=stage["page_layout"], adapter_training_mode="joint_print", adapter_scale=1.0,
                        negative_prompt="")
    expected.update({key: stage[key] for key in keys})
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing {module} output does not match this run.")
    size = expected["size"]
    dimensions = tuple(size) if isinstance(size, list) else (size, size)
    path = folder / f"{module}.png"
    verify_png(path, dimensions)
    result = dict(image=f"{module}/{path.name}", record=f"{module}/record.json",
                  sha256=file_digest(path), record_sha256=file_digest(record_file))
    if module == "book":
        for name in ("book.png", "book.safetensors", "glyph.png", "right_page.png"):
            if file_digest(folder / name) != record["output_sha256"].get(name):
                raise ValueError(f"Changed generated book artifact: {name}")
        for name in ("glyph.png", "right_page.png"):
            verify_png(folder / name, dimensions, "L")
        result.update(glyph="book/glyph.png", book_latent="book/book.safetensors")
    return result


def assemble(output: Path, plan: dict, results: dict, previous: dict | None) -> dict:
    inputs = {name: output / value["image"] for name, value in results.items()}
    inputs.update({key: output / results["book"][key] for key in ("glyph", "book_latent")})
    folder = output / "spread"
    if not folder.exists():
        if previous is not None:
            raise FileNotFoundError("The previously completed spread is missing.")
        from .spread import run_fusion
        run_fusion(model=Path(plan["model"]), sentence=plan["sentence"], seed=plan["seed"], steps=plan["steps"],
                   guidance=plan["guidance"], output=folder, scene_strength=plan["scene_strength"], **inputs)
    path = folder / "record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {key: plan[key] for key in ("sentence", "seed", "steps", "guidance", "scene_strength",
                                           "separate_ink", "scene_layout", "base_manifest_sha256")}
    expected.update(background_sampler=background.SCENE_SAMPLER, ink_applied=True, generate_board=True)
    if (any(record.get(key) != value for key, value in expected.items()) or
        any(record["input_sha256"].get(key) != file_digest(p) for key, p in inputs.items())):
        raise ValueError("The spread does not match the verified module outputs.")
    final = folder / "final.png"
    verify_png(final, tuple(plan["stages"]["background"]["size"]))
    result = dict(image="spread/final.png", sha256=file_digest(final), record="spread/record.json",
                  record_sha256=file_digest(path))
    if record["final_sha256"] != result["sha256"] or (previous is not None and result != previous):
        raise ValueError("The recorded spread has changed.")
    return result


def run_pipeline(plan: dict, output: Path, resume: bool) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    manifest = output / "manifest.json"
    with run_lock(output):
        if manifest.exists():
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if not resume or state["plan"] != plan:
                raise ValueError("Output directory belongs to a different run.")
        else:
            if any(item.name != ".pipeline.lock" for item in output.iterdir()):
                raise FileExistsError("Cannot adopt a directory without a pipeline manifest.")
            state = dict(format=1, output_mode="illustrated_book", plan=plan, results={})
        try:
            for module in MODULES:
                state.update(status="running", active=module, error=None)
                write_json(manifest, state)
                folder = output / module
                if not folder.exists():
                    if module in state["results"]:
                        raise FileNotFoundError(f"Previously recorded result missing: {folder}")
                    subprocess.run([sys.executable, "-m", "pipeline.generate", "--manifest", str(manifest),
                                    "--stage", module], cwd=PROJECT, check=True)
                result = read_stage(folder, module, plan)
                if module in state["results"] and result != state["results"][module]:
                    raise ValueError(f"Previously recorded {module} output has changed.")
                state["results"][module] = result
            state.update(status="running", active="spread")
            write_json(manifest, state)
            state["final"] = assemble(output, plan, state["results"], state.get("final"))
            state.update(status="completed", active=None)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(manifest, state)
            raise
        write_json(manifest, state)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Path to the approved local SDXL Base snapshot.")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/project.json")
    parser.add_argument("--size", type=int, help="Sampling resolution (default 768).")
    parser.add_argument("--steps", type=int, help="Denoising steps (default 32).")
    parser.add_argument("--guidance", type=float, help="Classifier-free guidance (default 6).")
    parser.add_argument("--book-seed", type=int, help="Seed for book framing (default 241126).")
    parser.add_argument("--scene-strength", type=float, help="Whole-scene redraw strength (default 0.52).")
    parser.add_argument("--preflight", action="store_true", help="Check inputs and print the sampling plan.")
    parser.add_argument("--resume", action="store_true", help="Reuse verified stages of this same run.")
    args = parser.parse_args()
    existing = args.output / "manifest.json"
    saved = json.loads(existing.read_text(encoding="utf-8"))["plan"] if args.resume and existing.is_file() else {}
    try:
        plan, errors = make_plan(args, saved)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.preflight or errors:
        print(json.dumps(dict(ready=not errors, problems=errors, plan=plan), ensure_ascii=False, indent=2))
        return 2 if errors else 0
    print(run_pipeline(plan, args.output, args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
