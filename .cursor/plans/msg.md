Let me trace through the entire training code path to find exactly where this decision is made.

[2 tools called]

Now let me trace how the model is created from config:

[2 tools called]

Now the critical part - `EagleDecoderLayer.__init__()`:

[1 tool called]

**FOUND IT!** Here's the exact decision point at lines 854-861:

```python
if attention_backend == "sdpa":
    self.self_attn = EagleAttention(config=config)
elif attention_backend == "flex_attention":           # ← DEFAULT!
    self.self_attn = EagleFlexAttention(config=config)  # ← Creates q_proj, k_proj, v_proj
elif attention_backend == 'flex':
    pass
elif attention_backend == "flex_attention_mla":       # ← MLA requires this!
    self.self_attn = EagleFlexMLA(config=config)        # ← Would create W_dkv, W_uk, W_uv
```

Now let me verify what each creates:

[2 tools called]

## Complete Chain of Evidence

### What `EagleFlexAttention` creates (standard GQA):
```python
# Lines 313-324
self.q_proj = nn.Linear(hidden_size * 2, num_heads * head_dim)     # YOUR MODEL HAS THIS ✓
self.k_proj = nn.Linear(hidden_size * 2, num_kv_heads * head_dim)  # YOUR MODEL HAS THIS ✓
self.v_proj = nn.Linear(hidden_size * 2, num_kv_heads * head_dim)  # YOUR MODEL HAS THIS ✓
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)         # YOUR MODEL HAS THIS ✓
```

### What `EagleFlexMLA` would create:
```python
# Lines 593-615
self.q_proj = nn.Linear(hidden_size * 2, num_heads * head_dim)
self.W_dkv = nn.Linear(hidden_size * 2, latent_dim)                # MISSING FROM YOUR MODEL ✗
self.W_uk = nn.Linear(latent_dim, num_kv_heads * head_dim)         # MISSING FROM YOUR MODEL ✗
self.W_uv = nn.Linear(latent_dim, num_kv_heads * head_dim)         # MISSING FROM YOUR MODEL ✗
self.kv_norm = nn.LayerNorm(latent_dim)                            # MISSING FROM YOUR MODEL ✗
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
```

## The Complete Flow

```
train_eagle3.py (line 110)
    │
    │  --attention-backend default="flex_attention"
    ▼
train_eagle3.py (line 249)
    │
    │  AutoEagle3DraftModel.from_config(config, attention_backend="flex_attention")
    ▼
auto.py (line 56)
    │
    │  EagleForCausalLM(config, attention_backend="flex_attention")
    ▼
eagle3.py (line 943)
    │
    │  self.midlayer = EagleDecoderLayer(config, attention_backend="flex_attention")
    ▼
eagle3.py (lines 856-857)
    │
    │  elif attention_backend == "flex_attention":
    │      self.self_attn = EagleFlexAttention(config)  ← THIS IS WHAT YOU GOT!
    │
    │  (NOT line 860-861:)
    │  elif attention_backend == "flex_attention_mla":
    │      self.self_attn = EagleFlexMLA(config)        ← THIS IS WHAT YOU WANTED
    ▼
eagle3.py (lines 313-324)
    │
    │  Creates: q_proj, k_proj, v_proj, o_proj  ← MATCHES YOUR CHECKPOINT!
```

## Root Cause

**To train with MLA, you must explicitly pass:**
```bash
python scripts/train_eagle3.py \
    --attention-backend flex_attention_mla \
    ... other args ...
```

**Without this flag, the default is `"flex_attention"` which creates standard GQA attention.**

Your config file having `mla_latent_dim` and `mla_rope_dim` doesn't matter - those values are only READ by `EagleFlexMLA.__init__()`, but that class is never instantiated unless you pass `--attention-backend flex_attention_mla`.