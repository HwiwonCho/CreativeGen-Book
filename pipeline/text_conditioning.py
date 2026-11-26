"""Build single-image SDXL text conditions with both native CLIP encoders."""


def text_conditioning(pipe, prompt, *, prompt_2=None, negative_prompt=None, guidance=True):
    encoders = (pipe.text_encoder, pipe.text_encoder_2)
    tokenizers = (pipe.tokenizer, pipe.tokenizer_2)

    def tokenize(texts):
        return [tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
                for tokenizer, text in zip(tokenizers, texts)]

    positive = tokenize((prompt, prompt_2 or prompt))
    negative = None
    if guidance and (negative_prompt is not None or not pipe.config.force_zeros_for_empty_prompt):
        negative = tokenize((negative_prompt or "", negative_prompt or ""))

    capacity = pipe.tokenizer.model_max_length - 2  # BOS and EOS in both SDXL Base encoders
    count = max(1, max((len(ids) + capacity - 1) // capacity for ids in positive + (negative or [])))
    if count == 1:
        return dict(prompt=prompt, prompt_2=prompt_2, negative_prompt=negative_prompt), count

    import torch

    def encode(pair):
        hidden = []
        for ids, tokenizer, encoder in zip(pair, tokenizers, encoders):
            chunks = [ids[i*capacity:(i+1)*capacity] for i in range(count)]
            windows = [[tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id]
                       + [tokenizer.pad_token_id] * (capacity - len(chunk)) for chunk in chunks]
            output = encoder(torch.tensor(windows, device=encoder.device), output_hidden_states=True)
            states = output.hidden_states[-2]
            hidden.append(states.reshape(1, -1, states.shape[-1]))

        # SDXL takes pooled text from CLIP-G only. Weight real text, ignoring padding-only windows.
        weights = output.text_embeds.new_tensor([len(chunk) for chunk in chunks], dtype=torch.float32)
        if not pair[-1]:
            weights[0] = 1
        pooled = ((weights / weights.sum()) @ output.text_embeds.float()).to(output.text_embeds.dtype)[None]
        return torch.cat(hidden, dim=-1).to(pipe.text_encoder_2.dtype), pooled

    with torch.no_grad():
        pos, pooled = encode(positive)
        neg = neg_pooled = None
        if negative is not None:
            neg, neg_pooled = encode(negative)
        elif guidance:
            neg, neg_pooled = torch.zeros_like(pos), torch.zeros_like(pooled)
    return dict(prompt=None, prompt_embeds=pos, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg, negative_pooled_prompt_embeds=neg_pooled), count
