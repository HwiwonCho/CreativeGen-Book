"""Text conditions for the background LoRA during training and full-frame inference."""

SCENE_SAMPLER = "sentence_scene_v2"
SCENE_SAMPLERS = (SCENE_SAMPLER,)


def flatlay_size(size: int) -> tuple[int, int]:
    unit = 8 * round(size * 5 / 96)
    return 3 * unit, 2 * unit


def build_prompt(sentence: str, *, flatlay: bool = False, sampler: str = SCENE_SAMPLER) -> str:
    if not sentence.strip():
        raise ValueError("A nonempty sentence is required.")
    if sampler != SCENE_SAMPLER:
        raise ValueError(f"Unknown background sampler: {sampler}")
    if flatlay:
        return "Generate a background: " + sentence
    return "Draw a background that matches the following sentence: " + sentence


def flatlay_settings(sentence: str, sampler: str, size: int) -> dict:
    return {
        "background_layout": "flatlay",
        "layout_sampler": sampler,
        "prompt": build_prompt(sentence, flatlay=True, sampler=sampler),
        "prompt_2": sentence + " Environment visible across the full frame.",
        "negative_prompt": "watermark, caption overlay, decorative frame, collage",
        "size": list(flatlay_size(size)),
    }
