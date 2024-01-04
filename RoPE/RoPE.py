import torch
import torch.nn as nn
import torch.einsum 

from einops import rearrange, repeat


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)

    x1, x2 = x.unbind(dim = -1)

    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast(enabled = False)

def apply_rotary_emb(freq, t, start_index = 0, scale = 1., seq_dim = -2):
    if t.ndim == 3:
        # [bsz, seq_len, dim]
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]

    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> n ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

