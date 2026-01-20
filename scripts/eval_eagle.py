#!/usr/bin/env python3
"""
EAGLE3 Draft Model Evaluation Script

Simple evaluation on a few prompts to check accuracy and speed.

Usage:
    # Basic usage with dataset file
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --dataset-path ./cache/dataset/train.jsonl

    # For MLA models (DeepSeek-V2, etc.)
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/eagle3-mla/epoch_9_step_300 \
        --target-model-path deepseek-ai/DeepSeek-V2-Lite \
        --attention-backend flex_attention_mla \
        --dataset-path ./cache/dataset/train.jsonl

    # With vocab mapping from cache directory
    python scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/model/epoch_0_step_1000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --vocab-mapping-path ./cache/vocab_mapping/abc123.pt \
        --dataset-path ./cache/dataset/train.jsonl

    # Also supports torchrun (not required)
    torchrun --nproc_per_node=1 scripts/eval_eagle.py \
        --draft-checkpoint ./outputs/model/epoch_0_step_1000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --dataset-path ./cache/dataset/train.jsonl \
        --num-samples 10
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.core.eagle3 import OnlineEagle3Model
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.distributed import init_distributed
from specforge.modeling.target import get_eagle3_target_model


# ==============================================================================
# Dataset Loading
# ==============================================================================


def load_dataset_samples(dataset_path: str, num_samples: int) -> List[Dict]:
    """Load samples from JSONL dataset file.

    Expected format (from collect_data.py):
    {"id": "...", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
    """
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line.strip())
            samples.append(data)
    return samples


def extract_conversation(sample: Dict) -> Tuple[List[Dict], str]:
    """Extract messages and the last assistant response from a sample.

    Returns:
        messages: List of conversation messages (including assistant)
        last_assistant_content: The content of the last assistant message
    """
    conversations = sample.get("conversations", [])
    messages = []
    last_assistant_content = ""

    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"] and content:
            messages.append({"role": role, "content": content})
            if role == "assistant":
                last_assistant_content = content

    return messages, last_assistant_content


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
        "--dataset-path",
        type=str,
        required=True,
        help="Path to JSONL dataset file (e.g., train.jsonl from collect_data.py)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of prompts to evaluate (default: 5)",
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
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["sdpa", "flex_attention", "flex_attention_mla"],
        help="Attention backend (use flex_attention_mla for MLA models like DeepSeek-V2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed token predictions for debugging",
    )
    parser.add_argument(
        "--show-first-n-positions",
        type=int,
        default=20,
        help="Number of positions to show in verbose mode (default: 20)",
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
# Helper Functions
# ==============================================================================


def get_draft_predictions(eagle3_model, eagle3_data, ttt_length, attention_backend):
    """Get draft model predictions for each TTT position."""
    from transformers.cache_utils import DynamicCache
    from specforge.utils import padding
    
    draft_model = eagle3_model.draft_model
    input_ids = eagle3_data.input_ids
    attention_mask = eagle3_data.attention_mask
    hidden_states = eagle3_data.hidden_states
    
    batch_size, seq_length = input_ids.shape
    device = hidden_states.device
    
    # Project hidden states
    hidden_states = draft_model.project_hidden_states(hidden_states)
    
    # Initialize cache
    if attention_backend == "sdpa" or attention_backend == "flex_attention_mla":
        cache_hidden = [[], []]
        past_key_values = None
    elif attention_backend == "flex_attention":
        cache_hidden = None
        past_key_values = DynamicCache()
    
    # Store predictions for each TTT position
    all_predictions = []
    
    # Initialize position_ids
    past_key_values_length = 0
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    for idx in range(ttt_length):
        # Embed input ids
        inputs_embeds = draft_model.embed_input_ids(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        
        # Run backbone
        hidden_states_out = draft_model.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        hidden_states = hidden_states_out
        
        # Get logits and predictions
        logits = draft_model.compute_logits(hidden_states)
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]
        all_predictions.append(predictions)
        
        # Pad for next iteration (except last)
        if idx < ttt_length - 1:
            input_ids = padding(input_ids, left=False)
            # Also pad position_ids to match
            position_ids = padding(position_ids, left=False)
    
    return all_predictions


# ==============================================================================
# Main Evaluation
# ==============================================================================


def main():
    args = parse_args()

    # Load dataset
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset_samples = load_dataset_samples(args.dataset_path, args.num_samples)
    num_samples = len(dataset_samples)

    if num_samples == 0:
        raise ValueError(f"No samples loaded from {args.dataset_path}")

    # Initialize distributed if running with torchrun
    if 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
        # Running with torchrun - initialize distributed
        init_distributed(timeout=10, tp_size=1)
        print(f"Initialized distributed: rank {dist.get_rank()}, world size {dist.get_world_size()}")
    else:
        # Running with regular python - no distributed needed
        print("Running in single-process mode (use python, not torchrun)")

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
        attention_backend=args.attention_backend,
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
        attention_backend=args.attention_backend,
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
        sample = dataset_samples[i]
        messages, response = extract_conversation(sample)

        if len(messages) < 2:
            print(f"  [{i+1}/{num_samples}] Skipping sample with insufficient messages")
            continue

        # Get first user message for display
        prompt = next((m["content"] for m in messages if m["role"] == "user"), "")

        # Add system prompt if template has one and sample doesn't
        if template.system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": template.system_prompt}] + messages

        # Tokenize (matching training behavior)
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}{response}{template.end_of_turn_token}"

        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            max_length=args.max_length, 
            truncation=True,
            add_special_tokens=False,  # Match training
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        offsets = inputs["offset_mapping"][0]
        
        # Build loss mask like training does (only assistant tokens contribute)
        loss_mask = torch.zeros(input_ids.shape[1], dtype=torch.long, device=args.device)
        
        user_sep = f"{template.end_of_turn_token or ''}{template.user_header}"
        asst_sep = f"{template.end_of_turn_token or ''}{template.assistant_header}"
        assistant_pattern = re.escape(asst_sep) + r"(.*?)(?=" + re.escape(user_sep) + "|$)"
        
        matches_found = 0
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            matches_found += 1
            start_char = match.start(1)
            end_char = match.end(1)
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end > start_char and tok_start <= end_char:
                    loss_mask[idx] = 1
        
        # Warn if no assistant spans found (matching training behavior)
        if matches_found == 0:
            print(f"\n⚠️  WARNING: No assistant spans found in sample {i+1}!")
            print(f"   Assistant separator: {repr(asst_sep)}")
            print(f"   Text preview: {repr(text[:200])}...")
        
        loss_mask = loss_mask.unsqueeze(0)  # Add batch dimension

        if args.verbose:
            assistant_tokens = loss_mask.sum().item()
            total_tokens = input_ids.shape[1]
            print(f"\n{'='*80}")
            print(f"Sample {i+1}/{num_samples}: {prompt[:60]}...")
            print(f"{'='*80}")
            print(f"Input length: {total_tokens} tokens ({assistant_tokens} assistant tokens in loss mask)")
        elif i == 0:
            # Show token counts for first sample even in non-verbose mode
            assistant_tokens = loss_mask.sum().item()
            print(f"   (Loss mask: {assistant_tokens}/{input_ids.shape[1]} tokens are assistant responses)")

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
            
            # If verbose, also get draft predictions separately
            if args.verbose:
                draft_predictions = get_draft_predictions(
                    eagle3_model, 
                    eagle3_data, 
                    args.ttt_length,
                    args.attention_backend
                )

        elapsed = time.time() - start
        total_time += elapsed
        total_tokens += input_ids.shape[1]

        # Collect metrics
        for j, (acc, loss) in enumerate(zip(acces, plosses)):
            all_accs[j].append(acc.item())
            all_losses[j].append(loss.item())

        mean_acc = sum(a.item() for a in acces) / len(acces)
        
        if args.verbose:
            # Show detailed predictions for first N positions
            print(f"\n📊 Detailed Predictions (showing first {args.show_first_n_positions} positions):")
            print(f"\nLegend: TTT=Time-to-Target offset (0 means predicting next token, 1 means +1 position ahead, etc.)")
            print(f"\n{'Pos':<4} {'TTT':<3} {'Context (last 3 tokens)':<50} {'Draft':<20} {'Target':<20} {'✓/✗':<3}")
            print("-" * 120)
            
            # Get target tokens (ground truth from target model logits)
            target_tokens = eagle3_data.target.argmax(dim=-1)[0]  # [seq_len]
            input_tokens = input_ids[0].tolist()  # [seq_len]
            
            # Show predictions for each position
            num_positions_to_show = min(args.show_first_n_positions, input_ids.shape[1] - args.ttt_length)
            
            for pos_idx in range(num_positions_to_show):
                # Show context (last 3 tokens before prediction point)
                context_start = max(0, pos_idx - 2)
                context_tokens = input_tokens[context_start:pos_idx+1]
                context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
                context_text = repr(context_text)[1:-1]  # Escape special chars
                context_text = ("..." + context_text[-47:]) if len(context_text) > 50 else context_text
                
                # For each TTT position (0 to ttt_length-1)
                for ttt_idx in range(args.ttt_length):
                    # Draft model predicts token at position pos_idx + ttt_idx
                    target_pos = pos_idx + ttt_idx
                    
                    if target_pos >= len(target_tokens):
                        break
                    
                    # Get draft prediction
                    draft_pred = draft_predictions[ttt_idx][0, pos_idx].item()
                    target_token = target_tokens[target_pos].item()
                    
                    match = "✓" if draft_pred == target_token else "✗"
                    
                    # Decode tokens
                    draft_text = tokenizer.decode([draft_pred], skip_special_tokens=False)
                    target_text = tokenizer.decode([target_token], skip_special_tokens=False)
                    
                    # Escape special characters for display
                    draft_text = repr(draft_text)[1:-1]  # Remove outer quotes from repr
                    target_text = repr(target_text)[1:-1]
                    
                    # Truncate long tokens
                    draft_text = (draft_text[:17] + "...") if len(draft_text) > 20 else draft_text
                    target_text = (target_text[:17] + "...") if len(target_text) > 20 else target_text
                    
                    # Only show context for first TTT position
                    if ttt_idx == 0:
                        print(f"{pos_idx:<4} {ttt_idx:<3} {context_text:<50} {draft_text:<20} {target_text:<20} {match:<3}")
                    else:
                        print(f"{'':4} {ttt_idx:<3} {'':50} {draft_text:<20} {target_text:<20} {match:<3}")
                
                # Add separator between positions
                if pos_idx < num_positions_to_show - 1 and pos_idx % 5 == 4:
                    print()
            
            print(f"\n{'='*80}")
            print(f"Per-TTT Position Accuracy:")
            for ttt_pos, acc in enumerate(acces):
                print(f"  TTT Position {ttt_pos}: {acc.item():.1%}")
            print(f"  Mean: {mean_acc:.1%}")
            print(f"{'='*80}\n")
        else:
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
