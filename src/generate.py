"""Generate one module output from a literal sentence and a seed, offline."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from . import background, book, illustration, joint_book
    from .approved_base import file_digest, verify_base
else:
    import background, book, illustration, joint_book
    from approved_base import file_digest, verify_base


def generate(args):
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate SDXL images.")
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = json.loads((args.adapter / "settings.json").read_text()) if args.adapter is not None else {}
    book_mode = getattr(args, "book_mode", None)
    if book_mode is not None and args.module != "book":
        raise ValueError("--book-mode applies only to the book module.")
    joint = args.module == "book" and (
        book_mode == "joint" or (book_mode is None and settings.get("book_mode") == "joint_print")
    )
    board = getattr(args, "book_composition", None) == "board"
    if board and (not joint or settings.get("book_mode") != "joint_print"):
        raise ValueError("Landscape board framing requires a directly trained joint-print book adapter.")
    flatlay = getattr(args, "background_layout", None) == "flatlay"
    if flatlay and args.module != "background":
        raise ValueError("--background-layout applies only to the background module.")
    sampler = getattr(args, "background_sampler", None)
    if sampler is not None and not flatlay:
        raise ValueError("--background-sampler requires a flatlay background.")
    sampler = sampler or background.SCENE_SAMPLER
    prompt = {
        "background": lambda sentence: background.build_prompt(sentence, flatlay=flatlay, sampler=sampler),
        "illustration": illustration.build_prompt,
        "book": (lambda sentence: joint_book.build_prompt(sentence, board=board)) if joint else
                lambda sentence: book.build_condition(sentence).scene_prompt,
    }[args.module](args.sentence)
    masks = None
    width = height = args.size
    if flatlay:
        width, height = background.flatlay_size(args.size)
    if args.module == "book":
        if not (args.adapter and args.font and (args.layout or board)):
            raise ValueError("Book generation requires trained book weights, a font and a team layout.")
        if board:
            masks, framing = joint_book.board_conditions(args.sentence, args.font, args.size, args.layout)
            width, height = masks[0].size
        else:
            masks = book.render_masks(args.sentence, args.font, args.layout)
            if masks[0].size != (args.size, args.size):
                raise ValueError("Layout canvas size differs from output size.")
    elif args.font or args.layout:
        raise ValueError("Font and layout apply only to book generation.")

    base_sha = verify_base(args.model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(args.model), variant="fp16", torch_dtype=torch.float16,
        local_files_only=True, use_safetensors=True, add_watermarker=False,
    ).to("cuda")
    pipe.enable_vae_tiling()
    record = dict(module=args.module, sentence=args.sentence, prompt=prompt,
                  seed=args.seed, steps=args.steps, guidance=args.guidance, size=args.size,
                  base_manifest_sha256=base_sha, base_only=args.adapter is None)
    context = nullcontext()
    sample_options = {}
    if flatlay:
        record.update(background.flatlay_settings(args.sentence, sampler, args.size))
        sample_options["negative_prompt"] = record["negative_prompt"]
        if "prompt_2" in record:
            sample_options["prompt_2"] = record["prompt_2"]
    for tokenizer, text in ((pipe.tokenizer, prompt), (pipe.tokenizer_2, record.get("prompt_2", prompt))):
        if len(tokenizer(text, truncation=False).input_ids) > tokenizer.model_max_length:
            raise ValueError("Prompt exceeds the native text encoder length.")
    if args.adapter is not None:
        if __package__:
            from . import train
        else:
            import train
        if settings["module"] != args.module or settings["base_manifest_sha256"] != base_sha:
            raise ValueError("Adapter module or base fingerprint mismatch.")
        train.load_adapter(pipe.unet, args.adapter)
        record["adapter_sha256"] = file_digest(args.adapter / "adapter.safetensors")
        if masks:
            if __package__:
                from .book_conditioner import condition_unet
            else:
                from book_conditioner import condition_unet
            encoder = train.load_book_conditioner(pipe.unet, args.adapter).cuda().eval()
            values = torch.from_numpy(np.stack([np.asarray(m) for m in masks])).float().cuda()[None] / 255
            with torch.no_grad():
                if joint and args.guidance > 1:
                    page_only = values.clone()
                    page_only[:, 0] = 0
                    residual = encoder(torch.cat((page_only, values)))
                else:
                    residual = encoder(values)
                    if args.guidance > 1:
                        residual = residual.repeat(2, 1, 1, 1)
            context = condition_unet(pipe.unet, residual)
            if joint:
                sample_options["negative_prompt"] = ""
                record.update(book_mode="joint_print", adapter_training_mode=settings.get("book_mode", "legacy"),
                              conditioning="literal sentence, glyph and right leaf from first denoising step",
                              cfg_unconditional="empty text and right leaf", code_sha256=file_digest(__file__),
                              prompt_code_sha256=file_digest(joint_book.__file__))
            record.update(font_sha256=file_digest(args.font), conditioner_sha256=file_digest(args.adapter / "conditioner.safetensors"))
            if board:
                record.update(**framing, book_composition="board", negative_prompt="", adapter_scale=1.0)
            else:
                record["layout"] = json.loads(args.layout.read_text(encoding="utf-8-sig"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        if getattr(args, "save_latent", False):
            from safetensors.torch import save_file
            def capture_last(pipe, index, timestep, tensors):
                if index == pipe.num_timesteps - 1:
                    latent = tensors["latents"].detach().cpu().contiguous()
                    save_file({"latent": latent}, str(stage / (args.module + ".safetensors")))
                return tensors
            sample_options.update(callback_on_step_end=capture_last, callback_on_step_end_tensor_inputs=["latents"])
        with context, torch.inference_mode():
            if flatlay and sampler == background.PERIPHERAL_SAMPLER:
                result, sampling = background.sample_flatlay(pipe, args.sentence, args.seed, (width,height),
                                                            args.steps, args.guidance, sample_options)
                record.update(sampling)
            else:
                result = pipe(prompt, height=height, width=width, num_inference_steps=args.steps,
                              guidance_scale=args.guidance, generator=torch.Generator("cpu").manual_seed(args.seed),
                              **sample_options)
        result.images[0].save(stage / (args.module + ".png"))
        if masks:
            for mask, name in zip(masks, ("glyph.png", "right_page.png")):
                mask.save(stage / name)
        if joint or getattr(args, "save_latent", False):
            if getattr(args, "save_latent", False) and not (stage / (args.module + ".safetensors")).exists():
                raise RuntimeError("The last denoising latent was not captured.")
            record["output_sha256"] = {p.name:file_digest(p) for p in sorted(stage.iterdir())}
        record["scheduler"] = type(pipe.scheduler).__name__
        record["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (stage / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.rename(stage, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("background","illustration","book"), required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--book-mode", choices=("legacy", "joint"), help="Override the book checkpoint's inference mode.")
    parser.add_argument("--background-layout", choices=("flatlay",), help="Reserve the center of an overhead tabletop.")
    parser.add_argument("--background-sampler", choices=background.FLATLAY_SAMPLERS)
    parser.add_argument("--book-composition", choices=("board", "original"), default="original",
                        help="Use a native landscape tabletop frame for joint-print books.")
    parser.add_argument("--save-latent", action="store_true", help="Keep the scaled sampled latent before VAE decoding.")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=6.0)
    args = parser.parse_args()
    if args.size < 256 or args.size % 8 or args.steps < 1 or not 0 <= args.guidance <= 30:
        parser.error("Expected size >=256 divisible by 8, positive steps and guidance from 0 to 30.")
    generate(args)


if __name__ == "__main__":
    main()
