"""Sample an unpositioned open book and project generated conditions onto its detected leaves."""

import torch
from torch.nn import functional as F

from architecture.page_locator import grid,homography,landmarks,project


BLANK_PROMPT=("Overhead flat lay photograph of a single open blank hardcover book lying flat on a broad matte table. "
              "Completely empty warm ivory paper on both facing pages. Soft, even diffuse illumination across the paper, "
              "a narrow upright central binding, and a soft contact shadow beneath the cover. "
              "All four corners of the book visible, with wide margins of table around the book.")


def page_grid(quad,h,w):
    unit=quad.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    xy=grid(h,w,quad.device).reshape(-1,2)
    return project(homography(quad,unit),xy).reshape(h,w,2)


def quad_size(quad,h,w):
    p=quad*quad.new_tensor([w,h])
    width=(p[[1,3]]-p[[0,2]]).norm(dim=-1).mean().item()
    height=(p[[2,3]]-p[[0,1]]).norm(dim=-1).mean().item()
    return max(1,round(width)),max(1,round(height))


def geometry(logits):
    points=landmarks(logits)[0]
    paper=logits[0,0].sigmoid()
    h,w=paper.shape
    union=torch.zeros_like(paper,dtype=torch.bool)
    for ix in ([0,1,3,4],[1,2,4,5]):
        q=points[ix]
        edges=q[[1,3,2,0]]-q[[0,1,3,2]]
        cross=edges[:,0]*edges.roll(-1,0)[:,1]-edges[:,1]*edges.roll(-1,0)[:,0]
        if (cross<=0).any():
            return points,0.,float(paper.mean())
        coords=page_grid(q,h,w)
        union |= ((coords>=0)&(coords<=1)).all(-1)
    truth=paper>.5
    return points,float((truth&union).sum()/(truth|union).sum().clamp_min(1)),float(truth.float().mean())


@torch.inference_mode()
def blank_scene(pipe,locator,seed,bounds):
    for name,param in pipe.unet.named_parameters():
        if name.endswith("lora_B"):
            param.mul_(.38)
    trials=[]
    try:
        for attempt in range(6):
            book_seed=(seed+104729*attempt)%(2**63)
            latent=pipe(prompt=BLANK_PROMPT,negative_prompt="text, letters, handwriting, multiple books, cropped pages, hands, side view, hard shadows across the paper",
                        width=1008,height=672,num_inference_steps=28,guidance_scale=6.,
                        generator=torch.Generator().manual_seed(book_seed),output_type="latent").images.float()
            pixels=(pipe.vae.decode(latent/pipe.vae.config.scaling_factor).sample.clamp(-1,1)+1)/2
            pred=locator(pixels)
            points,agreement,area=geometry(pred)
            trials.append(dict(seed=book_seed,agreement=agreement,area=area))
            if agreement>=bounds["min_agreement"] and bounds["min_area"]<=area<=bounds["max_area"]:
                return pixels,pred[:,:1].sigmoid(),points,trials
    finally:
        for name,param in pipe.unet.named_parameters():
            if name.endswith("lora_B"):
                param.div_(.38)
    raise ValueError(f"No complete open spread found in {len(trials)} sampled scenes: {trials}")


def project_art(illustration,local,paper_width,paper_height,region):
    xmin,xmax,center=region
    h,w=illustration.shape[-2:]
    width=(xmax-xmin)*paper_width
    height=width*h/w
    shrink=min(1.,paper_height/height)
    width,height=width*shrink/paper_width,height*shrink/paper_height
    dx=(local[...,0]-(1-(xmin+xmax)/2))/width+.5
    dy=(local[...,1]-center)/height+.5
    uv=torch.stack((dx,dy),-1)
    tex=F.grid_sample(illustration, (uv*2-1)[None], align_corners=False,padding_mode="border")
    valid=((uv>=0)&(uv<=1)).all(-1)[None,None]
    return tex,valid


def text_density(field,text,quad,h,w,region,paragraph):
    pw,ph=quad_size(quad,h,w)
    xmin,xmax,center=region
    col_width=round((xmax-xmin)*pw)
    usable=int(2*min(center,1-center)*ph)
    lines,line_h=paragraph(field,text,col_width,usable)
    alpha=torch.cat([field.coverage(line,line_h,col_width) for line in lines],-2)
    block_h=alpha.shape[-2]
    local=page_grid(quad,h,w)
    x=((local[...,0]-xmin)*pw)/col_width
    y=(local[...,1]-center)*ph/block_h+.5
    sample=F.grid_sample(alpha,torch.stack((x*2-1,y*2-1),-1)[None],align_corners=False)
    # OCR gets a straightened copy of the final image, with the same projective geometry.
    unit=quad.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    detail=grid(block_h+16,col_width+16,quad.device)*quad.new_tensor([col_width+16,block_h+16])-8
    detail=detail/quad.new_tensor([pw,ph])+quad.new_tensor([xmin,center-block_h/(2*ph)])
    read_grid=project(homography(unit,quad),detail.reshape(-1,2)).reshape(*detail.shape)
    return sample,read_grid,lines
