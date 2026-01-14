# MLA (Multi-Latent Attention) for EAGLE3

This document describes the MLA attention backend for EAGLE3 speculative decoding draft models.

## Overview

MLA compresses Key-Value pairs to a latent space, reducing memory usage while maintaining model quality. This is based on DeepSeek's Multi-Head Latent Attention architecture.

**Memory Savings**: ~4-8x reduction in KV cache size per token.

## How It Works

```
Standard Attention:
  K = k_proj(x)  →  cache K
  V = v_proj(x)  →  cache V

MLA Attention:
  c_kv = W_dkv(x)        →  compress to latent (small)
  K = W_uk(c_kv)         →  decompress on-the-fly
  V = W_uv(c_kv)         →  decompress on-the-fly
```

## Configuration

Add these to your model config JSON:

```json
{
  "mla_latent_dim": 128,
  "mla_rope_dim": 0,
  "mla_use_kv_norm": true
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mla_latent_dim` | `num_kv_heads * head_dim // 4` | Latent compression dimension |
| `mla_rope_dim` | `0` | Decoupled RoPE dim (0 = standard RoPE) |
| `mla_use_kv_norm` | `true` | Apply LayerNorm to latent |

## Usage

### Training

```bash
python scripts/train_eagle3.py \
    --model-path <target_model> \
    --attention-backend flex_attention_mla \
    --output-dir ./outputs/eagle3-mla \
    --dataset <your_dataset>
```

### Evaluation

```bash
python scripts/eval_eagle.py \
    --draft-checkpoint ./outputs/eagle3-mla/checkpoint \
    --target-model-path <target_model> \
    --attention-backend flex_attention_mla \
    --num-samples 10
```

## Important Notes

1. **Architecture Incompatibility**: Models trained with MLA use different weight names (`W_dkv`, `W_uk`, `W_uv`) than standard attention (`k_proj`, `v_proj`). You cannot mix backends.

2. **Cache Format**: MLA uses the same `cache_hidden = [[K...], [V...]]` format as SDPA for EAGLE3 compatibility.

3. **RoPE**: Standard RoPE (rope_dim=0) is recommended. Decoupled RoPE is experimental.

## References

- [EAGLE-3 Paper](https://arxiv.org/abs/2503.01840)
- [DeepSeek MLA Paper](https://arxiv.org/abs/2405.04434)
