#!/usr/bin/env python3
"""
EAGLE3 Draft Model Evaluation Script

Simple evaluation on a few prompts to check accuracy and speed.

Usage:
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct

    # With vocab mapping from cache directory
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/model/epoch_0_step_1000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --vocab-mapping-path ./cache/vocab_mapping/abc123.pt

    # With more prompts
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/model/epoch_0_step_1000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --num-samples 10
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.core.eagle3 import OnlineEagle3Model
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.modeling.target import get_eagle3_target_model


# ==============================================================================
# Built-in Test Prompts
# ==============================================================================

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate factorial.",
    "What are the benefits of regular exercise?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Describe the water cycle.",
    "What is the Pythagorean theorem?",
    "How do computers store information?",
    "What causes the seasons on Earth?",
]

# Corresponding simple responses for evaluation
TEST_RESPONSES = [
    "The capital of France is Paris. Paris is located in the north-central part of the country and is the largest city in France.",
    "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1.",
    "Here's a Python function to calculate factorial:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "Regular exercise improves cardiovascular health, strengthens muscles, boosts mood, helps maintain healthy weight, and increases energy levels.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.",
    "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed.",
    "The water cycle involves evaporation from bodies of water, condensation into clouds, precipitation as rain or snow, and collection back into bodies of water.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c².",
    "Computers store information using binary code (0s and 1s) in memory cells. Data is stored in RAM temporarily and on hard drives permanently.",
    "Seasons are caused by Earth's tilted axis (23.5 degrees) as it orbits the Sun, causing different hemispheres to receive varying amounts of sunlight.",
]


# ==============================================================================
# Argument Parsing
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EAGLE3 draft model")

    parser.add_argument(
        "--draft-checkpoint",
        type=str,
        required=True,
        help="Path to trained EAGLE3 draft model checkpoint",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to target model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of prompts to evaluate (default: 5, max: 10)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen",
        help="Chat template (default: qwen)",
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="TTT length - positions to predict ahead (default: 7)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["hf", "sglang"],
    )
    parser.add_argument(
        "--vocab-mapping-path",
        type=str,
        default=None,
        help="Path to vocab_mapping.pt (default: looks in checkpoint dir, then ./cache/vocab_mapping/)",
    )

    return parser.parse_args()


# ==============================================================================
# Utilities
# ==============================================================================


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.2f}K"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ==============================================================================
# Main Evaluation
# ==============================================================================


def main():
    args = parse_args()
    num_samples = min(args.num_samples, len(TEST_PROMPTS))

    print("\n" + "=" * 60)
    print("EAGLE3 EVALUATION")
    print("=" * 60)

    # Validate checkpoint
    checkpoint_path = os.path.abspath(args.draft_checkpoint)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load draft model
    print(f"\n📦 Loading draft model from: {checkpoint_path}")
    draft_model = AutoEagle3DraftModel.from_pretrained(
        checkpoint_path,
        attention_backend="flex_attention_mla",
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    draft_params = count_params(draft_model)
    print(f"   Parameters: {fmt_params(draft_params)}")

    # Load vocab mapping - check multiple locations
    vocab_mapping_path = None
    if args.vocab_mapping_path and os.path.exists(args.vocab_mapping_path):
        vocab_mapping_path = args.vocab_mapping_path
    elif os.path.exists(os.path.join(checkpoint_path, "vocab_mapping.pt")):
        vocab_mapping_path = os.path.join(checkpoint_path, "vocab_mapping.pt")
    else:
        # Try common cache locations
        cache_candidates = [
            "./cache/vocab_mapping",
            "../cache/vocab_mapping",
        ]
        for cache_dir in cache_candidates:
            if os.path.isdir(cache_dir):
                # Find any .pt file in the cache dir
                for f in os.listdir(cache_dir):
                    if f.endswith(".pt"):
                        vocab_mapping_path = os.path.join(cache_dir, f)
                        break
            if vocab_mapping_path:
                break

    if vocab_mapping_path and os.path.exists(vocab_mapping_path):
        draft_model.load_vocab_mapping(vocab_mapping_path)
        print(f"   Loaded vocab mapping from: {vocab_mapping_path}")
    else:
        print("   ⚠️  No vocab mapping found! Results may be inaccurate.")
        print("      Use --vocab-mapping-path to specify location")

    # Load target model
    print(f"\n🎯 Loading target model: {args.target_model_path}")
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=args.device,
    )

    # Set aux hidden states layers
    draft_config = draft_model.config
    if (
        hasattr(draft_config, "eagle_config")
        and draft_config.eagle_config
        and "eagle_aux_hidden_state_layer_ids" in draft_config.eagle_config
    ):
        target_model.set_aux_hidden_states_layers(
            draft_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        )
    else:
        target_model.set_aux_hidden_states_layers()

    # Estimate target params
    target_config = AutoConfig.from_pretrained(args.target_model_path)
    target_params = getattr(target_config, "num_parameters", None)
    if target_params is None:
        h = target_config.hidden_size
        n_layers = target_config.num_hidden_layers
        vocab = target_config.vocab_size
        inter = getattr(target_config, "intermediate_size", h * 4)
        target_params = vocab * h * 2 + n_layers * (h * h * 4 + h * inter * 3)
    print(f"   Parameters: ~{fmt_params(target_params)} (estimated)")

    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # Build EAGLE3 model for evaluation
    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend="flex_attention",
    ).to(args.device).eval()

    # Get chat template
    template = TEMPLATE_REGISTRY.get(args.chat_template)

    # Prepare test samples
    print(f"\n📊 Evaluating on {num_samples} prompts...")
    print("-" * 60)

    all_accs = [[] for _ in range(args.ttt_length)]
    all_losses = [[] for _ in range(args.ttt_length)]
    total_tokens = 0
    total_time = 0.0

    for i in range(num_samples):
        prompt = TEST_PROMPTS[i]
        response = TEST_RESPONSES[i]

        # Build conversation
        messages = [{"role": "user", "content": prompt}]
        if template.system_prompt:
            messages = [{"role": "system", "content": template.system_prompt}] + messages
        messages.append({"role": "assistant", "content": response})

        # Tokenize
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}{response}{template.end_of_turn_token}"

        inputs = tokenizer(text, return_tensors="pt", max_length=args.max_length, truncation=True)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        loss_mask = attention_mask.clone()

        # Evaluate
        start = time.time()

        with torch.no_grad():
            # Get target model data
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

            # Run EAGLE3 forward pass
            plosses, _, acces = eagle3_model(
                input_ids=eagle3_data.input_ids,
                attention_mask=eagle3_data.attention_mask,
                loss_mask=eagle3_data.loss_mask,
                target=eagle3_data.target,
                hidden_states=eagle3_data.hidden_states,
            )

        elapsed = time.time() - start
        total_time += elapsed
        total_tokens += input_ids.shape[1]

        # Collect metrics
        for j, (acc, loss) in enumerate(zip(acces, plosses)):
            all_accs[j].append(acc.item())
            all_losses[j].append(loss.item())

        mean_acc = sum(a.item() for a in acces) / len(acces)
        print(f"  [{i+1}/{num_samples}] \"{prompt[:40]}...\" → acc={mean_acc:.1%} ({elapsed:.2f}s)")

    # Compute final metrics
    per_pos_acc = [sum(accs) / len(accs) for accs in all_accs]
    per_pos_loss = [sum(losses) / len(losses) for losses in all_losses]
    mean_acc = sum(per_pos_acc) / len(per_pos_acc)
    mean_loss = sum(per_pos_loss) / len(per_pos_loss)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n📊 Per-Position Accuracy:")
    for i, (acc, loss) in enumerate(zip(per_pos_acc, per_pos_loss)):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"   Position {i}: {bar} {acc:.1%}  (loss: {loss:.4f})")

    print("\n📈 Summary:")
    print(f"   Mean Accuracy:    {mean_acc:.1%}")
    print(f"   Mean Loss:        {mean_loss:.4f}")
    print(f"   Total Time:       {total_time:.2f}s")
    print(f"   Tokens Processed: {total_tokens:,}")
    print(f"   Throughput:       {total_tokens / total_time:.1f} tokens/s")

    # Quality interpretation
    print("\n💡 Quality:")
    if mean_acc >= 0.7:
        print("   ✅ EXCELLENT - Expected speedup: 2.5x - 3.5x")
    elif mean_acc >= 0.5:
        print("   ⚠️  GOOD - Expected speedup: 1.5x - 2.5x")
    elif mean_acc >= 0.3:
        print("   ⚠️  FAIR - Expected speedup: 1.2x - 1.5x")
    else:
        print("   ❌ NEEDS WORK - Train longer or tune hyperparameters")

    # Decay analysis
    if per_pos_acc[0] > 0:
        decay = (per_pos_acc[0] - per_pos_acc[-1]) / per_pos_acc[0] * 100
        print(f"\n   Accuracy decay (pos 0→{len(per_pos_acc)-1}): {decay:.1f}%")

    print("\n" + "=" * 60)
    print("✅ Done!")


if __name__ == "__main__":
    main()
