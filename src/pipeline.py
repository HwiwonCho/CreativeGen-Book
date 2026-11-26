"""Run the three trained generators offline from one sentence and master seed."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

from PIL import Image, ImageDraw

if __package__:
    from . import background, book, illustration, joint_book, print_page
    from .approved_base import file_digest, verify_base
else:
    import background, book, illustration, joint_book, print_page
    from approved_base import file_digest, verify_base


MODULES = ("background", "illustration", "book")
PROJECT = Path(__file__).resolve().parents[1]
BASE_REFERENCE = PROJECT / "provenance/sdxl-base-files.json"


def write_json(path, value):
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(folder):
    with (folder / ".pipeline.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is using this output directory.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is using this output directory.") from exc
        yield


def make_plan(args):
    errors = []
    if not args.sentence.strip() or not 0 <= args.seed < 2**63:
        raise ValueError("Supply a nonempty sentence and a seed from 0 through 2^63-1.")
    if args.size < 256 or args.size % 8 or args.steps < 1 or not 0 <= args.guidance <= 30:
        raise ValueError("Invalid size, steps or guidance.")
    base = json.loads(BASE_REFERENCE.read_text(encoding="utf-8"))
    base_sha = file_digest(BASE_REFERENCE)
    missing_base = [name for name in base["files"] if not (args.model / name).is_file()]
    if missing_base:
        errors.append(f"SDXL Base: {len(missing_base)} required files missing at {args.model}.")
    stages = {}
    for module in args.modules:
        adapter = getattr(args, module + "_adapter")
        token = hashlib.sha256(f"{args.seed}/{module}".encode()).digest()
        seed = int.from_bytes(token[:8], "big") % 2**63
        prompt = (book.build_condition(args.sentence).scene_prompt if module == "book" else
                  (background if module == "background" else illustration).build_prompt(args.sentence))
        stage = dict(adapter=str(adapter.resolve()) if adapter else None, seed=seed, prompt=prompt)
        if module == "background" and args.scene_layout == "flatlay":
            stage.update(background.flatlay_settings(args.sentence, args.background_sampler or background.SCENE_SAMPLER, args.size))
        stages[module] = stage
        if adapter is None:
            errors.append(f"{module}: supply --{module}-adapter with a trained run directory.")
            continue
        files = ["settings.json", "adapter.safetensors"]
        if module == "book":
            files.append("conditioner.safetensors")
        missing = [name for name in files if not (adapter / name).is_file()]
        if missing:
            errors.append(f"{module}: missing {', '.join(missing)} in {adapter}.")
            continue
        try:
            settings = json.loads((adapter / "settings.json").read_text(encoding="utf-8"))
            if (settings["module"], settings["base_id"], settings["base_revision"],
                settings["base_manifest_sha256"]) != (module, base["model_id"], base["revision"], base_sha):
                raise ValueError("Incompatible module or base checkpoint.")
            stage["adapter_sha256"] = file_digest(adapter / "adapter.safetensors")
            if module == "book":
                stage["conditioner_sha256"] = file_digest(adapter / "conditioner.safetensors")
                if args.spread:
                    stage["seed"] = args.book_seed
                if settings.get("book_mode") == "joint_print":
                    board = args.book_composition == "board" or (args.book_composition is None and args.spread)
                    stage.update(book_mode="joint_print", prompt=joint_book.build_prompt(args.sentence, board=board))
                    if board:
                        stage["book_composition"] = "board"
                        if args.use_default_layout:
                            args.layout = None
                    elif args.use_default_layout:
                        args.layout = PROJECT / "configs/book-layout-joint.json"
                elif args.book_composition == "board":
                    raise ValueError("Board framing requires a joint-print book adapter.")
                elif args.spread:
                    stage.update(prompt=print_page.PAPER_PROMPTS["bound"], paper="bound")
                    if args.print_profile == "paper":
                        stage["print_profile"] = "paper"
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{module}: {exc}")
    board = stages.get("book", {}).get("book_composition") == "board"
    if args.scene_strength is not None:
        if not 0 < args.scene_strength <= 1:
            errors.append("Scene strength must be greater than zero and at most one.")
        if not args.spread or not board:
            errors.append("Whole-scene generation requires an integrated board composition.")
    elif args.separate_ink:
        errors.append("Separate ink requires an explicit scene strength.")
    if args.scene_layout in ("flatlay", "table") and args.scene_strength is None:
        errors.append("The selected scene layout requires a scene strength.")
    if args.background_sampler and args.scene_layout != "flatlay":
        errors.append("The background sampler option requires the flatlay scene layout.")
    for key in (("font", "layout") if "book" in stages else ()):
        path = getattr(args, key)
        if board and key == "layout" and path is None:
            continue
        if path is None or not path.is_file():
            errors.append(f"book: supply an existing --{key}.")
        else:
            stages["book"][key] = str(path.resolve())
            stages["book"][key+"_sha256"] = file_digest(path)
    if "book" in stages and "font" in stages["book"] and (board or "layout" in stages["book"]):
        try:
            if board:
                _, framing = joint_book.board_conditions(args.sentence, args.font, args.size, args.layout)
                stages["book"]["page_layout"] = framing.pop("layout")
                stages["book"].update(framing)
            elif "paper" in stages["book"]:
                for size in (42, 36, 30, 24, 18):
                    layout = print_page.prepare_layout(args.layout, args.font, size)
                    try:
                        glyph, _ = book.render_masks(args.sentence, args.font, layout)
                    except ValueError:
                        if size == 18:
                            raise
                    else:
                        break
                stages["book"]["font_size"] = size
            else:
                glyph, _ = book.render_masks(args.sentence, args.font, args.layout)
            if not board and glyph.size != (args.size, args.size):
                raise ValueError("Layout canvas differs from --size.")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"book: {exc}")
    if not errors:
        try:
            verify_base(args.model)
        except (OSError, ValueError) as exc:
            errors.append(f"SDXL Base: {exc}")
    plan = dict(sentence=args.sentence, seed=args.seed, model=str(args.model.resolve()),
                base_manifest_sha256=base_sha, size=args.size, steps=args.steps,
                guidance=args.guidance, stages=stages, modules=list(args.modules), overview=args.overview)
    if args.spread:
        plan["spread"] = True
    if args.scene_strength is not None:
        plan["scene_strength"] = args.scene_strength
        if args.separate_ink:
            plan["separate_ink"] = True
        if args.scene_layout in ("flatlay", "table"):
            plan["scene_layout"] = args.scene_layout
    return plan, errors


def read_stage(folder, module, plan):
    stage = plan["stages"][module]
    board = stage.get("book_composition") == "board"
    if module == "book" and "paper" in stage:
        return read_print(folder, stage, plan)
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    expected = {key: plan[key] for key in ("sentence", "size", "steps", "guidance", "base_manifest_sha256")}
    expected.update(module=module, base_only=False, seed=stage["seed"], prompt=stage["prompt"],
                    adapter_sha256=stage["adapter_sha256"])
    if stage.get("background_layout") == "flatlay":
        expected.update(background_layout="flatlay", layout_sampler=stage.get("layout_sampler"),
                        negative_prompt=stage["negative_prompt"], size=stage["size"])
        if "prompt_2" in stage:
            expected["prompt_2"] = stage["prompt_2"]
    if module == "book":
        expected.update(conditioner_sha256=stage["conditioner_sha256"], font_sha256=stage["font_sha256"])
        if board:
            expected.update({key:stage[key] for key in ("size", "native_width", "native_height", "layout_offset")})
            expected.update(layout=stage["page_layout"], adapter_training_mode="joint_print", adapter_scale=1.0, negative_prompt="")
        else:
            expected["layout"] = json.loads(Path(stage["layout"]).read_text(encoding="utf-8-sig"))
        if "book_mode" in stage:
            expected["book_mode"] = stage["book_mode"]
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing {module} output does not match this run.")
    path = folder / (module + ".png")
    dimensions = expected["size"]
    dimensions = tuple(dimensions) if isinstance(dimensions, list) else (dimensions, dimensions)
    with Image.open(path) as im:
        if im.format != "PNG" or im.size != dimensions:
            raise ValueError(f"Invalid {module} result image.")
        im.verify()
    result = dict(image=f"{module}/{path.name}", record=f"{module}/record.json",
                  sha256=file_digest(path), record_sha256=file_digest(folder / "record.json"))
    if stage.get("book_mode") == "joint_print" and plan.get("spread"):
        for key, name in (("image", "book.png"), ("glyph", "glyph.png"), ("book_latent", "book.safetensors")):
            if file_digest(folder/name) != record["output_sha256"][name]:
                raise ValueError(f"Changed generated book artifact: {name}")
            result[key] = f"book/{name}"
        if board:
            if file_digest(folder/"right_page.png") != record["output_sha256"]["right_page.png"]:
                raise ValueError("Changed generated book artifact: right_page.png")
            for name in ("glyph.png", "right_page.png"):
                with Image.open(folder/name) as mask:
                    if mask.format != "PNG" or mask.mode != "L" or mask.size != dimensions:
                        raise ValueError(f"Invalid landscape condition: {name}")
                    mask.verify()
    return result


def read_print(folder, stage, plan):
    path = folder / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(sentence=[plan["sentence"]], seed=stage["seed"], steps=plan["steps"], cfg=plan["guidance"],
                    font_size=stage["font_size"], paper=stage["paper"], prompt=stage["prompt"],
                    save_latent=True, base_manifest_sha256=plan["base_manifest_sha256"])
    expected.update({key:stage[key] for key in ("font_sha256","adapter_sha256","conditioner_sha256")})
    layout = print_page.prepare_layout(stage["layout"], stage["font"], stage["font_size"])
    if any(record.get(k) != v for k,v in expected.items()) or book.PageLayout(**record["page_layout"]) != layout:
        raise ValueError("Printed book belongs to a different pipeline run.")
    files = {"image":"0000.png", "blank":"baseline.png", "glyph":"glyph-0000.png", "book_latent":"0000.safetensors"}
    if stage.get("print_profile") == "paper":
        if record.get("print_profile") != "paper":
            raise ValueError("Expected a page-registered book.")
        files["layout_glyph"] = "layout-glyph-0000.png"
    elif record.get("print_profile", "legacy") != "legacy":
        raise ValueError("Unexpected book print profile.")
    hashes = {kind:file_digest(folder/name) for kind,name in files.items()}
    if any(record["output_sha256"].get(name) != hashes[kind] for kind,name in files.items()):
        raise ValueError("Printed book artifacts changed.")
    return {**{kind:f"book/{name}" for kind,name in files.items()}, "sha256":hashes["image"],
            "hashes":hashes, "record":"book/run.json", "record_sha256":file_digest(path)}


def run_stage(module, plan, folder):
    stage = plan["stages"][module]
    if module == "book" and "paper" in stage:
        command = [sys.executable, str(Path(__file__).with_name("print_page.py")), "--sentence", plan["sentence"],
                   "--seed", str(stage["seed"]), "--paper", stage["paper"], "--font-size", str(stage["font_size"]),
                   "--steps", str(plan["steps"]), "--guide-steps", str(min(4,plan["steps"])),
                   "--cfg", str(plan["guidance"]), "--adapter", stage["adapter"], "--model", plan["model"],
                   "--font", stage["font"], "--layout", stage["layout"], "--print-profile",
                   stage.get("print_profile", "legacy"), "--save-latent", "--output", str(folder)]
        subprocess.run(command, check=True)
        return
    command = [sys.executable, str(Path(__file__).with_name("generate.py")),
               "--module", module, "--sentence", plan["sentence"], "--seed", str(stage["seed"]),
               "--model", plan["model"], "--adapter", stage["adapter"], "--output", str(folder)]
    for key in ("size", "steps", "guidance"):
        command.extend(["--"+key, str(plan[key])])
    if "background_layout" in stage:
        command.extend(["--background-layout", stage["background_layout"], "--background-sampler", background.saved_sampler(stage)])
    if module == "book":
        command.extend(["--font", stage["font"]])
        if "layout" in stage:
            command.extend(["--layout", stage["layout"]])
        if "book_composition" in stage:
            command.extend(["--book-composition", stage["book_composition"]])
        if stage.get("book_mode") == "joint_print" and plan.get("spread"):
            command.append("--save-latent")
    subprocess.run(command, check=True)


def review_sheet(output, results, modules):
    width = 384
    sheet = Image.new("RGB", (width * len(modules), width + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for i, module in enumerate(modules):
        with Image.open(output / results[module]["image"]) as im:
            preview = im.convert("RGB")
            preview.thumbnail((width, width), Image.Resampling.LANCZOS)
            sheet.paste(preview, (i*width + (width-preview.width)//2, (width-preview.height)//2))
        draw.text((i*width+8, width+7), module.title(), fill="black")
    temp = output / ".review.png"
    sheet.save(temp)
    os.replace(temp, output / "review.png")


def assemble(output, plan, results, previous):
    book_result = results["book"]
    inputs = {key:output/results[key]["image"] for key in ("background","illustration")}
    inputs.update(book=output/book_result["image"])
    inputs.update({key:output/book_result[key] for key in ("blank","glyph","book_latent") if key in book_result})
    if "layout_glyph" in book_result:
        inputs["layout_glyph"] = output/book_result["layout_glyph"]
    folder = output/"spread"
    board = plan["stages"]["book"].get("book_composition") == "board"
    scene_strength = plan.get("scene_strength")
    separate_ink = plan.get("separate_ink", False)
    scene_layout = plan.get("scene_layout", "board")
    background_sampler = plan["stages"]["background"].get("layout_sampler")
    if not folder.exists():
        if previous is not None:
            raise FileNotFoundError("The previously completed spread is missing.")
        if __package__:
            from .spread import run_fusion
        else:
            from spread import run_fusion
        run_fusion(model=Path(plan["model"]), sentence=plan["sentence"], seed=plan["seed"],
                   steps=plan["steps"], guidance=plan["guidance"], output=folder,
                   generate_board=board, scene_strength=scene_strength, separate_ink=separate_ink,
                   scene_layout=scene_layout, background_sampler=background_sampler, **inputs)
    path = folder/"record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if scene_layout == "flatlay" and separate_ink and not record.get("ink_applied"):
        raise ValueError("The overhead spread is missing its printed sentence.")
    if (background_sampler in background.SCENE_SAMPLERS or "background_sampler" in record) and record.get("background_sampler") != background_sampler:
        raise ValueError("The spread does not match the planned background sampler.")
    if (record.get("generate_board", False) != board or record.get("scene_strength") != scene_strength or
        record.get("separate_ink", False) != separate_ink or record.get("scene_layout", "board") != scene_layout or
        any(record.get(key) != plan[key] for key in ("sentence","seed","steps","guidance")) or
        any(record["input_sha256"].get(key) != file_digest(p) for key,p in inputs.items())):
        raise ValueError("The spread does not match the verified module outputs.")
    final_hash = file_digest(folder/"final.png")
    dimensions = background.flatlay_size(plan["size"]) if scene_layout == "flatlay" else (plan["size"],plan["size"])
    with Image.open(folder/"final.png") as image:
        if image.format != "PNG" or image.size != dimensions:
            raise ValueError("Invalid spread dimensions or format.")
        image.verify()
    result = dict(image="spread/final.png", sha256=final_hash, record="spread/record.json",
                  record_sha256=file_digest(path))
    if record["final_sha256"] != final_hash or (previous is not None and result != previous):
        raise ValueError("The recorded spread has changed.")
    return result


def run_pipeline(plan, output, resume):
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
            mode = "illustrated_book" if plan.get("spread") else "independent_modules"
            state = dict(format=1, output_mode=mode, plan=plan, results={})
        try:
            for module in plan["modules"]:
                state.update(status="running", active=module, error=None)
                write_json(manifest, state)
                folder = output / module
                if not folder.exists():
                    if module in state["results"]:
                        raise FileNotFoundError(f"Previously recorded result missing: {folder}")
                    run_stage(module, plan, folder)
                result = read_stage(folder, module, plan)
                if module in state["results"] and result != state["results"][module]:
                    raise ValueError(f"Previously recorded {module} output has changed.")
                state["results"][module] = result
                write_json(manifest, state)
            if plan.get("spread"):
                state.update(status="running", active="spread")
                write_json(manifest, state)
                state["final"] = assemble(output, plan, state["results"], state.get("final"))
            if plan["overview"]:
                review_sheet(output, state["results"], plan["modules"])
                state["review_sheet"] = "review.png"
            state.update(status="completed", active=None)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(manifest, state)
            raise
        write_json(manifest, state)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/project.json")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modules", choices=MODULES, nargs="+", default=MODULES)
    parser.add_argument("--independent", action="store_true", help="Generate separate module outputs only.")
    parser.add_argument("--book-seed", type=int, help="Book framing seed used for an integrated spread (default 241126).")
    parser.add_argument("--book-composition", choices=("board", "original"),
                        help="Use a landscape book on a generated support, or the original square book.")
    parser.add_argument("--scene-strength", type=float, help="Redraw the whole scene at this noise strength (0, 1].")
    parser.add_argument("--scene-layout", choices=("flatlay", "board", "table"),
                        help="Use the default overhead flat lay, a support board or a continuous table.")
    parser.add_argument("--background-sampler", choices=(background.SCENE_SAMPLER, background.PERIPHERAL_SAMPLER),
                        help="Generate the full scene or use the earlier tabletop-and-props background.")
    parser.add_argument("--separate-ink", action=argparse.BooleanOptionalAction, default=None,
                        help="Guide the printed sentence separately in the generated scene.")
    parser.add_argument("--print-profile", choices=("paper", "legacy"), help="Ink rendering for an integrated spread.")
    for module in MODULES:
        parser.add_argument("--"+module+"-adapter", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--preflight", action="store_true", help="Check inputs and print plan without generation.")
    parser.add_argument("--resume", action="store_true", help="Reuse verified stages of this same run.")
    parser.add_argument("--overview", action="store_true", help="Add a labeled side-by-side sheet for review.")
    args = parser.parse_args()
    requested_print_profile = args.print_profile is not None
    if len(set(args.modules)) != len(args.modules):
        parser.error("List each module only once.")
    args.spread = not args.independent and set(args.modules) == set(MODULES)
    old_manifest = args.output/"manifest.json"
    if args.resume and old_manifest.is_file():
        saved_plan = json.loads(old_manifest.read_text(encoding="utf-8"))["plan"]
        if not saved_plan.get("spread"):
            args.spread = False
        saved_book = saved_plan["stages"].get("book", {})
        saved_bg = saved_plan["stages"].get("background", {})
        if args.background_sampler is None and saved_bg.get("background_layout") == "flatlay":
            args.background_sampler = background.saved_sampler(saved_bg)
        if args.print_profile is None:
            args.print_profile = saved_book.get("print_profile", "legacy")
        if args.book_composition is None:
            args.book_composition = saved_book.get("book_composition", "original")
        if args.scene_strength is None:
            args.scene_strength = saved_plan.get("scene_strength")
        if args.scene_layout is None:
            args.scene_layout = saved_plan.get("scene_layout", "board")
        if args.separate_ink is None:
            args.separate_ink = saved_plan.get("separate_ink", False)
        if args.book_seed is None and args.spread:
            args.book_seed = saved_book["seed"]
        for module, stage in saved_plan["stages"].items():
            if getattr(args, module + "_adapter") is None and stage.get("adapter"):
                setattr(args, module + "_adapter", Path(stage["adapter"]))
        for field in ("layout", "font"):
            if getattr(args, field) is None and saved_book.get(field):
                setattr(args, field, Path(saved_book[field]))
    elif (args.spread and args.scene_layout is None and args.scene_strength is None and
          args.book_composition is None and args.book_adapter is None and not requested_print_profile):
        args.scene_layout = "flatlay"
    if args.scene_layout == "flatlay":
        if args.scene_strength is None:
            args.scene_strength = .52
        if args.separate_ink is None:
            args.separate_ink = True
    args.use_default_layout = args.layout is None
    if args.book_seed is None:
        args.book_seed = 241126
    args.print_profile = args.print_profile or "paper"
    if args.spread and (not 0 <= args.book_seed < 2**63 or not 1 <= args.guidance <= 30):
        parser.error("Book sampling requires a nonnegative 63-bit book seed and CFG from 1 to 30.")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    def default_path(field, value):
        if getattr(args, field) is None and value:
            setattr(args, field, PROJECT / value)
    default_path("model", cfg.get("model"))
    if args.model is None:
        parser.error("Supply the local SDXL Base path in the config or use --model.")
    for module in args.modules:
        module_cfg = cfg["modules"][module]
        adapter = module_cfg.get("adapter")
        if module == "book" and (args.book_composition == "board" or
                                 (args.spread and args.book_composition is None and not requested_print_profile)):
            adapter = module_cfg.get("integrated_adapter", adapter)
        default_path(module + "_adapter", adapter)
        if module == "book":
            for field in ("font", "layout"):
                default_path(field, module_cfg.get(field))
    plan, errors = make_plan(args)
    if args.preflight or errors:
        print(json.dumps(dict(ready=not errors, problems=errors, plan=plan), ensure_ascii=False, indent=2))
        return 2 if errors else 0
    print(run_pipeline(plan, args.output, args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
