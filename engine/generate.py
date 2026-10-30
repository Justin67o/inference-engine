import torch


@torch.inference_mode()
def generate(model, ids, max_new_tokens, stop_ids=()):
    """Greedy decoding with no KV cache: every step reruns the model on the whole sequence.
    
    ids: [T] prompt token ids on the model's device. Returns the new token ids, [N]."""
    seq = ids
    for _ in range(max_new_tokens):
        logits = model(seq, torch.arange(len(seq), device=seq.device))[-1] # last position only
        next_id = logits.argmax()
        seq = torch.cat([seq, next_id[None]])

        if next_id.item() in stop_ids:
            break
    return seq[len(ids):]

