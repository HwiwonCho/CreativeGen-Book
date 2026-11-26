"""Background generator input. Keep the user's scene and style wording intact."""
import re


FLATLAY_NEGATIVE = "open book, distant horizon, panoramic scenery, eye-level view, centered objects, clutter, captions, watermark"
PERIPHERAL_SAMPLER = "peripheral_objects_v1"
SCENE_SAMPLER = "sentence_scene_v2"
SCENE_SAMPLERS = (SCENE_SAMPLER, "sentence_scene_v1")
_GENERIC_V0 = "generic_prompt_v0"
_MOTIF_V0 = "motif_prompt_v0"
FLATLAY_SAMPLERS = (*SCENE_SAMPLERS, PERIPHERAL_SAMPLER, _GENERIC_V0, _MOTIF_V0)
_GENERIC_PREFIX = ("Overhead flat lay, camera perpendicular to the tabletop. A few small objects at the "
                   "outer edges, a wide empty center. The objects evoke this sentence: ")

_MOTIFS = (
    ("train trains railway railroad railroads locomotive subway tram platform", "a punched rail ticket", "a pocket watch"),
    ("dog dogs puppy puppies canine", "a coiled dog leash", "a paw-shaped tag"),
    ("cat cats kitten kittens feline", "a cat collar with a bell", "a small ball of yarn"),
    ("music piano violin orchestra song songs musician melody", "a tuning fork", "a slip of sheet music"),
    ("letter letters postcard postcards mail envelope", "a sealed envelope", "a postage stamp"),
    ("rain rains raining rained rainy rainfall raindrops downpour storm storms drizzle umbrella", "a folded umbrella", "a few water droplets"),
    ("snow snows snowy snowing snowfall winter frost ice", "a knitted mitten", "a pine cone"),
    ("beach beaches sea seas ocean oceans seaside coast shore tide waves", "a scallop shell", "a twist of rope"),
    ("flowers flower garden gardens bloom blooms roses rose petals", "a cut flower stem", "a seed packet"),
    ("forest forests woodland woods tree trees leaf leaves trail", "a small branch", "a fallen leaf"),
    ("moon lunar stars starry telescope constellation", "a folded star chart", "a telescope eyepiece"),
    ("city cities urban downtown skyline skyscraper skyscrapers metropolis", "a folded street map", "a transit ticket"),
    ("mountain mountains summit summits hiking hikers", "a hiking compass", "a loop of climbing cord"),
    ("coffee espresso café cafe tea teapot", "a cup on its saucer", "a teaspoon"),
)


def flatlay_size(size: int) -> tuple[int, int]:
    unit = 8 * round(size * 5 / 96)
    return 3 * unit, 2 * unit


def motifs(sentence):
    words = set(re.findall(r"[^\W\d_]+", sentence.casefold()))
    return next(((left,right) for cues,left,right in _MOTIFS if words.intersection(cues.split())), ())


def build_prompt(sentence: str, *, flatlay: bool = False, sampler: str = SCENE_SAMPLER) -> str:
    if not sentence.strip():
        raise ValueError("A nonempty sentence is required.")
    if flatlay:
        if sampler == SCENE_SAMPLER:
            return "Generate a background: " + sentence
        if sampler == "sentence_scene_v1":
            return "Draw a continuous full-frame scene with surroundings visible on both sides. The scene depicts: " + sentence
        if sampler == _GENERIC_V0:
            return _GENERIC_PREFIX + sentence
        if sampler not in (PERIPHERAL_SAMPLER, _MOTIF_V0):
            raise ValueError(f"Unknown background sampler: {sampler}")
        props = "One or two objects suggested by the sentence, at the outer corners."
        pair = motifs(sentence)
        if pair:
            props = f"{pair[0].capitalize()} at the upper left edge, {pair[1]} at the lower right edge."
        return "Overhead flat lay, camera perpendicular to a tabletop. " + props + " Broad empty center. " + sentence
    return "Draw a background that matches the following sentence: " + sentence


def saved_sampler(stage):
    if "layout_sampler" in stage:
        return stage["layout_sampler"]
    return _GENERIC_V0 if stage["prompt"].startswith(_GENERIC_PREFIX) else _MOTIF_V0


def flatlay_settings(sentence, sampler, size):
    negative = FLATLAY_NEGATIVE
    if sampler in SCENE_SAMPLERS:
        negative = "watermark, caption overlay, decorative frame, collage"
    elif sampler == _GENERIC_V0:
        negative = "book, pages, horizon, skyline, landscape view, tilted camera, cluttered center, captions, watermark"
    settings = dict(background_layout="flatlay", prompt=build_prompt(sentence, flatlay=True, sampler=sampler),
                    negative_prompt=negative, size=list(flatlay_size(size)))
    if sampler == SCENE_SAMPLER:
        settings["prompt_2"] = sentence + " Environment visible across the full frame."
    if sampler in (*SCENE_SAMPLERS, PERIPHERAL_SAMPLER):
        settings["layout_sampler"] = sampler
    return settings


def isolate_prop(image):
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h,w = rgb.shape[:2]
    n = max(3, min(h,w)//14)
    corner = [np.median(patch,axis=(0,1)) for patch in
              (rgb[:n,:n],rgb[:n,-n:],rgb[-n:,:n],rgb[-n:,-n:])]
    x = np.linspace(0,1,w,dtype=np.float32)[None,:,None]
    y = np.linspace(0,1,h,dtype=np.float32)[:,None,None]
    matte = (1-y)*((1-x)*corner[0]+x*corner[1])+y*((1-x)*corner[2]+x*corner[3])
    dist = np.sqrt(np.mean((rgb-matte)**2,axis=-1))
    perimeter = np.concatenate((dist[:n,:].ravel(),dist[-n:,:].ravel(),dist[:,:n].ravel(),dist[:,-n:].ravel()))
    threshold = float(np.clip(np.quantile(perimeter,.88)+5,10,42))
    solid = ndimage.binary_closing(dist>threshold,iterations=2)
    labels,count = ndimage.label(solid)
    areas = np.bincount(labels.ravel(), minlength=count+1)
    regions = np.flatnonzero(areas[1:] >= max(12, h*w*.0009)) + 1
    if not len(regions):
        return None
    solid = ndimage.binary_fill_holes(np.isin(labels,regions))
    if solid.mean() < .006 or solid.mean() > .78:
        return None
    alpha = ndimage.gaussian_filter(solid.astype(np.float32),.75)
    result = Image.fromarray(np.dstack((rgb,255*alpha)).round().astype(np.uint8),"RGBA")
    return result.crop(result.getbbox())


def sample_flatlay(pipe, sentence, seed, size, steps, guidance, options):
    import hashlib
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from PIL import Image, ImageFilter

    def role_seed(role):
        digest = hashlib.sha256(f"{seed}/{sentence}/{role}".encode()).digest()
        return int.from_bytes(digest[:8],"big") % 2**63

    def generator(value):
        return torch.Generator("cpu").manual_seed(value)

    w,h = size
    surface_prompt = "An empty tabletop surface seen vertically from directly above. Uninterrupted material, subtle natural surface detail, wall to wall, nothing resting on it."
    empty_negative = "objects, book, food, cutlery, hands, lettering, table edges, landscape, horizon, window, room"
    initial_steps = min(steps,24)
    surface_seed = role_seed("surface")
    guide = pipe(surface_prompt,negative_prompt=empty_negative,width=w,height=h,num_inference_steps=initial_steps,
                 guidance_scale=guidance,generator=generator(surface_seed)).images[0].convert("RGB")
    prop_negative = "close-up, cut off, hand, room, scenery, tabletop edge, decorative border, background pattern, watermark"
    prop_steps = min(steps,24)
    # A novel sentence can pick its own associated object without a captioning model.
    subjects = motifs(sentence) or ("one small object associated with this sentence: " + sentence,)
    placements = []
    for index,subject in enumerate(subjects):
        prop_prompt = ("Top-down product photograph of " + subject
                       + ". Single complete subject centered on a plain white studio background, generous white margins.")
        prop_seed = role_seed(f"object/{index}")
        attempts = []
        for candidate in (prop_seed,role_seed(f"object/{index}/retry")):
            attempts.append(candidate)
            sample = pipe(prop_prompt,negative_prompt=prop_negative,width=640,height=640,num_inference_steps=prop_steps,
                          guidance_scale=guidance,generator=generator(candidate)).images[0]
            obj = isolate_prop(sample)
            if obj is not None:
                break
        placement = dict(subject=subject,prompt=prop_prompt,seed=attempts[-1],attempted_seeds=attempts,box=None)
        if obj is not None:
            obj.thumbnail((round(w*.137),round(h*.29)),Image.Resampling.LANCZOS)
            x = round(w*.012) if index==0 else w-round(w*.012)-obj.width
            y = round(h*(.25 if index==0 else .75) - obj.height/2)
            alpha = obj.getchannel("A")
            shadow = Image.new("L",size)
            shadow.paste(alpha,(x+max(1,w//500),y+max(2,h//180)))
            shadow = shadow.filter(ImageFilter.GaussianBlur(max(.8,h/260))).point(lambda v:round(.22*v))
            guide.paste((0,0,0),(0,0,w,h),shadow)
            guide.paste(obj,(x,y),alpha)
            placement["box"] = [x,y,x+obj.width,y+obj.height]
        placements.append(placement)
    if not any(item["box"] is not None for item in placements):
        raise ValueError("No isolated tabletop object was visible after resampling.")
    scene = StableDiffusionXLImg2ImgPipeline(**pipe.components,add_watermarker=False)
    strength = max(.27,1./steps)
    rendered = scene(build_prompt(sentence,flatlay=True,sampler=PERIPHERAL_SAMPLER),image=guide,strength=strength,
                     num_inference_steps=steps,guidance_scale=guidance,generator=generator(seed),**options)
    settings = dict(layout_sampler=PERIPHERAL_SAMPLER,surface_prompt=surface_prompt,surface_negative=empty_negative,
                    surface_seed=surface_seed,surface_steps=initial_steps,prop_negative=prop_negative,
                    prop_steps=prop_steps,prop_size=[640,640],placements=placements,background_strength=strength)
    return rendered,settings
