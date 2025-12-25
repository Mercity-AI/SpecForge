import argparse
import os

import torch
from transformers import AutoConfig

from specforge import AutoEagle3DraftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Eagle3 checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to Eagle3 checkpoint directory (epoch_x_step_y)",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HF path or local path to target model",
    )
    return parser.parse_args()


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def estimate_target_params_from_config(cfg) -> int | None:
    """
    Rough parameter estimate for decoder-only transformers.
    Avoids loading full target model.
    """
    required = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    ]
    if not all(hasattr(cfg, k) for k in required):
        return None

    h = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    vocab = cfg.vocab_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = h // n_heads
    intermediate = getattr(cfg, "intermediate_size", h * 4)

    # embeddings
    embed_params = vocab * h

    # attention (q, k, v, o)
    attn_params = n_layers * (
        h * n_heads * head_dim               # Q
        + h * n_kv_heads * head_dim * 2      # K,V
        + n_heads * head_dim * h             # O
    )

    # MLP (gate, up, down)
    mlp_params = n_layers * (h * intermediate * 3)

    # norms
    norm_params = n_layers * h * 2 + h

    # lm head
    lm_head_params = h * vocab

    return (
        embed_params
        + attn_params
        + mlp_params
        + norm_params
        + lm_head_params
    )


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint_path)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # ===========================================
    # Load draft model (config + weights)
    # ===========================================
    draft_model = AutoEagle3DraftModel.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )

    draft_total, draft_trainable = count_parameters(draft_model)
    draft_config = draft_model.config.to_dict()

    # ===========================================
    # Load target config (NO weights)
    # ===========================================
    target_config = AutoConfig.from_pretrained(args.target_model_path)
    target_config_dict = target_config.to_dict()
    target_params = estimate_target_params_from_config(target_config)

    # ===========================================
    # Print results
    # ===========================================
    print("\n" + "=" * 60)
    print("DRAFT MODEL")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total params:     {fmt(draft_total)} ({draft_total:,})")
    print(f"Trainable params: {fmt(draft_trainable)} ({draft_trainable:,})")
    print("\nConfig:")
    for k, v in sorted(draft_config.items()):
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TARGET MODEL")
    print("=" * 60)
    print(f"Path: {args.target_model_path}")
    if target_params is not None:
        print(f"Estimated params: {fmt(target_params)} ({target_params:,})")
    else:
        print("Estimated params: <unavailable>")

    print("\nConfig (selected):")
    keys_to_show = [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "hidden_act",
        "torch_dtype",
    ]
    for k in keys_to_show:
        if k in target_config_dict:
            print(f"  {k}: {target_config_dict[k]}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if target_params is not None:
        ratio = draft_total / target_params * 100
        print(f"Draft / Target ratio: {ratio:.2f}%")
    print(f"Draft params:  {fmt(draft_total)}")
    if target_params is not None:
        print(f"Target params: {fmt(target_params)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
