"""Generate a single planned stage in its own GPU process."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from architecture import background, joint_book
from architecture.approved_base import file_digest, verify_base


def generate(manifest: Path, module: str) -> None:
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLPipeline
    from architecture.adapters import load_adapter, load_book_conditioner
    from architecture.book_conditioner import condition_unet

    state = json.loads(manifest.read_text(encoding="utf-8"))
    plan, output = state["plan"], manifest.parent / module
    stage = plan["stages"][module]
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate SDXL images.")
    adapter = Path(stage["adapter"])
    settings = json.loads((adapter / "settings.json").read_text(encoding="utf-8"))
    base_hash = verify_base(plan["model"])
    if settings["module"] != module or settings["base_manifest_sha256"] != base_hash or base_hash != plan["base_manifest_sha256"]:
        raise ValueError("Adapter module or base fingerprint mismatch.")
    adapter_hash = file_digest(adapter / "adapter.safetensors")
    if adapter_hash != stage["adapter_sha256"]:
        raise ValueError(f"Changed {module} adapter.")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        plan["model"], variant="fp16", torch_dtype=torch.float16,
        local_files_only=True, use_safetensors=True, add_watermarker=False,
    ).to("cuda")
    pipe.enable_vae_tiling()
    prompt = stage["prompt"]
    sentence, size = plan["sentence"], plan["size"]
    width = height = size
    record = dict(module=module, sentence=sentence, prompt=prompt, seed=stage["seed"],
                  steps=plan["steps"], guidance=plan["guidance"], size=size,
                  base_manifest_sha256=base_hash, base_only=False)
    options = {}
    if module == "background":
        record.update(background.flatlay_settings(sentence, background.SCENE_SAMPLER, size))
        width, height = record["size"]
        options.update(negative_prompt=stage["negative_prompt"], prompt_2=stage["prompt_2"])
    for tokenizer, text in ((pipe.tokenizer, prompt), (pipe.tokenizer_2, options.get("prompt_2", prompt))):
        if len(tokenizer(text, truncation=False).input_ids) > tokenizer.model_max_length:
            raise ValueError("Prompt exceeds the native text encoder length.")
    load_adapter(pipe.unet, adapter)
    record["adapter_sha256"] = adapter_hash

    context = nullcontext()
    masks = None
    if module == "book":
        if settings.get("book_mode") != "joint_print":
            raise ValueError("A directly trained joint-print book adapter is required.")
        font = Path(stage["font"])
        conditioner_hash = file_digest(adapter / "conditioner.safetensors")
        if conditioner_hash != stage["conditioner_sha256"] or file_digest(font) != stage["font_sha256"]:
            raise ValueError("Changed book conditioner or font.")
        masks, framing = joint_book.board_conditions(sentence, font, size)
        width, height = masks[0].size
        encoder = load_book_conditioner(pipe.unet, adapter).cuda().eval()
        values = torch.from_numpy(np.stack([np.asarray(m) for m in masks])).float().cuda()[None] / 255
        with torch.no_grad():
            if plan["guidance"] > 1:
                page_only = values.clone()
                page_only[:, 0] = 0
                values = torch.cat((page_only, values))
            residual = encoder(values)
        context = condition_unet(pipe.unet, residual)
        options["negative_prompt"] = ""
        record.update(book_mode="joint_print", adapter_training_mode="joint_print",
                      conditioning="literal sentence, glyph and right leaf from first denoising step",
                      cfg_unconditional="empty text and right leaf", code_sha256=file_digest(__file__),
                      prompt_code_sha256=file_digest(joint_book.__file__), font_sha256=stage["font_sha256"],
                      conditioner_sha256=conditioner_hash, **framing, book_composition="board",
                      negative_prompt="", adapter_scale=1.0)

    with TemporaryDirectory(dir=output.parent) as temporary:
        result_dir = Path(temporary) / "result"
        result_dir.mkdir()
        if masks is not None:
            from safetensors.torch import save_file

            def capture_book(pipeline, index, timestep, tensors):
                if index == pipeline.num_timesteps - 1:
                    latent = tensors["latents"].detach().cpu().contiguous()
                    save_file({"latent": latent}, str(result_dir / "book.safetensors"))
                return tensors

            options.update(callback_on_step_end=capture_book, callback_on_step_end_tensor_inputs=["latents"])
        with context, torch.inference_mode():
            image = pipe(prompt, height=height, width=width, num_inference_steps=plan["steps"],
                         guidance_scale=plan["guidance"], generator=torch.Generator("cpu").manual_seed(stage["seed"]),
                         **options).images[0]
        image.save(result_dir / f"{module}.png")
        if masks is not None:
            if not (result_dir / "book.safetensors").is_file():
                raise RuntimeError("The last denoising latent was not captured.")
            for mask, name in zip(masks, ("glyph.png", "right_page.png")):
                mask.save(result_dir / name)
            record["output_sha256"] = {p.name: file_digest(p) for p in sorted(result_dir.iterdir())}
        record["scheduler"] = type(pipe.scheduler).__name__
        record["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (result_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        result_dir.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("background", "illustration", "book"), required=True)
    args = parser.parse_args()
    generate(args.manifest, args.stage)
