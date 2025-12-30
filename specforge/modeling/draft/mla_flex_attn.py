import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
    or_masks,
)
import torch._dynamo as dynamo
from transformer.utils import is_torchdynamo_compiling

dynamo.config.recompile_limit = 64

class MLAFlexAttention:

    _instance = None
    _compiled_flex_attention = None
    _is_flex_compiled = False

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):

        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(
                flex_attention
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention

class WrappedCreateBlockMask:
    _instance = None
    _is_create_block_mask_compiled = False
    _compiled_create_block_mask = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_create_block_mask_compiled:
            self._compiled_create_block_mask = torch.compile(create_block_mask)
            self._is_create_block_mask_compiled = True

    def __call__(self):
        return self._compiled_create_block_mask


def compile_mla_flex_attention(
    x: torch.Tensor,              # Input hidden states [B, S, D]
    
    # MLA projection weights
    W_dq: nn.Linear,              # Down-project query
    W_uq: nn.Linear,              # Up-project query
    W_dkv: nn.Linear,             # Down-project KV (compress)
    W_uk: nn.Linear,              # Up-project to K (decompress)
    W_uv: nn.Linear,              # Up-project to V (decompress)
    W_o: nn.Linear,               # Output projection
    
    num_heads: int,
    
    # FlexAttention options
    block_mask=None,
    score_mod=None,
    
    # Optional: cached latent for inference
    cached_c_kv: torch.Tensor = None,
):
    B, S, D = x.shape
    
    # === MLA Compression/Decompression ===
    # Query
    c_q = W_dq(x)
    Q = W_uq(c_q)
    
    # Key-Value through latent bottleneck
    c_kv_new = W_dkv(x)
    
    # Handle KV cache (cache the LATENT, not K/V)
    if cached_c_kv is not None:
        c_kv = torch.cat([cached_c_kv, c_kv_new], dim=1)
    else:
        c_kv = c_kv_new
    
    # Decompress to K, V
    K = W_uk(c_kv)
    V = W_uv(c_kv)
    
    # === Reshape for attention ===
    head_dim = Q.shape[-1] // num_heads
    
    Q = Q.view(B, -1, num_heads, head_dim).transpose(1, 2)  # [B, H, S_q, D]
    K = K.view(B, -1, num_heads, head_dim).transpose(1, 2)  # [B, H, S_kv, D]
    V = V.view(B, -1, num_heads, head_dim).transpose(1, 2)  # [B, H, S_kv, D]
    
    # === FlexAttention ===
    flex_attn = MLAFlexAttention()()
    out = flex_attn(Q, K, V, score_mod=score_mod, block_mask=block_mask)
    
    # === Output ===
    out = out.transpose(1, 2).contiguous().view(B, -1, num_heads * head_dim)
    out = W_o(out)
    
    return out, c_kv  # Return latent for caching


def compile_block_mask(
    mask_mod, 
    B, H, 
    Q_LEN, KV_LEN, 
    device
):
    create_block_mask_compiled = (
        WrappedCreateBlockMask()()
        if not is_torchdynamo_compiling()
        else create_block_mask
    )
    return create_block_mask_compiled(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
    )

def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod
