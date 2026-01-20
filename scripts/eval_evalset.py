#!/usr/bin/env python3
"""
Evaluation script using prompts from train.jsonl dataset.
Performs speculative decoding and reports metrics.

Usage:
    python scripts/eval_evalset.py \
        --draft-checkpoint ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --data-path ./cache/dataset/ultrachat_train.jsonl \
        --num-samples 10

    # With vocab mapping and MLA backend
    python scripts/eval_evalset.py \
        --draft-checkpoint ./outputs/eagle3-mla/epoch_9_step_300 \
        --target-model-path deepseek-ai/DeepSeek-V2-Lite \
        --data-path ./cache/dataset/ultrachat_train.jsonl \
        --vocab-mapping-path ./cache/vocab_mapping/abc123.pt \
        --attention-backend flex_attention_mla \
        --num-samples 5
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import AutoEagle3DraftModel
from specforge.data.template import TEMPLATE_REGISTRY


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class TimingMetrics:
    """Timing metrics for a single generation."""
    ttft: float = 0.0  # Time to first token
    total_time: float = 0.0  # Total generation time
    tokens_generated: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0

    # Speculative decoding specific
    accept_count: int = 0
    draft_count: int = 0
    num_spec_steps: int = 0

    @property
    def throughput(self) -> float:
        """Tokens per second."""
        if self.total_time > 0:
            return self.tokens_generated / self.total_time
        return 0.0

    @property
    def decode_throughput(self) -> float:
        """Tokens per second (decode only, excluding prefill)."""
        if self.decode_time > 0:
            return self.tokens_generated / self.decode_time
        return 0.0

    @property
    def accept_rate(self) -> float:
        """Draft token acceptance rate."""
        if self.draft_count > 0:
            return self.accept_count / self.draft_count
        return 0.0

    @property
    def avg_accepted_per_step(self) -> float:
        """Average tokens accepted per speculative step."""
        if self.num_spec_steps > 0:
            return self.accept_count / self.num_spec_steps
        return 0.0


# ==============================================================================
# Argument Parsing
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EAGLE3 on train.jsonl dataset")

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
        "--data-path",
        type=str,
        required=True,
        help="Path to train.jsonl file (e.g., ./cache/dataset/ultrachat_train.jsonl)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of prompts to evaluate (default: 10)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen",
        help="Chat template (default: qwen)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Max prompt length (default: 512)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Draft tokens per speculative step (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
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
        default="sdpa",
        choices=["sdpa", "flex_attention", "flex_attention_mla"],
        help="Attention backend (use flex_attention_mla for MLA models like DeepSeek-V2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0=greedy, default: 0.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show generated text for each sample",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )

    return parser.parse_args()


# ==============================================================================
# Dataset Loading
# ==============================================================================


def load_samples_from_jsonl(data_path: str, num_samples: int) -> List[Dict]:
    """
    Load samples from train.jsonl file.
    Expected format:
    {
        "id": str,
        "conversations": [
            {"role": "user", "content": str},
            {"role": "assistant", "content": str},
            ...
        ]
    }
    """
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                data = json.loads(line.strip())
                if "conversations" in data and data["conversations"]:
                    samples.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i+1}")
                continue

    return samples


def prepare_prompt(
    tokenizer,
    conversations: List[Dict],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare prompt from conversation (only user turns, no assistant response).
    Returns input_ids and attention_mask for generation.
    """
    # Take only the first user message for generation prompt
    messages = []
    for conv in conversations:
        if conv["role"] == "user":
            messages.append({"role": "user", "content": conv["content"]})
            break  # Only first user turn

    if not messages:
        messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Apply chat template with generation prompt
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback
        text = f"User: {messages[0]['content']}\nAssistant:"

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )

    return inputs["input_ids"], inputs["attention_mask"]


# ==============================================================================
# Speculative Decoding
# ==============================================================================


def get_aux_hidden_states(
    hidden_states_tuple: Tuple[torch.Tensor, ...],
    aux_layer_ids: List[int],
) -> torch.Tensor:
    """
    Extract and concatenate hidden states from auxiliary layers.
    Eagle3 uses multiple layers' hidden states concatenated together.
    """
    aux_hidden = []
    for layer_id in aux_layer_ids:
        # Layer outputs are offset by 1 (index 0 is embeddings)
        idx = layer_id + 1 if layer_id < len(hidden_states_tuple) - 1 else layer_id
        if idx < len(hidden_states_tuple):
            aux_hidden.append(hidden_states_tuple[idx])

    if not aux_hidden:
        # Fallback: use last layer hidden states repeated 3 times
        return torch.cat([hidden_states_tuple[-1]] * 3, dim=-1)

    return torch.cat(aux_hidden, dim=-1)


def generate_speculative(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    num_draft_tokens: int = 4,
    temperature: float = 0.0,
    aux_layer_ids: Optional[List[int]] = None,
) -> Tuple[List[int], TimingMetrics]:
    """
    Speculative decoding using Eagle3's draft prediction.
    Returns generated tokens and timing metrics.
    """
    metrics = TimingMetrics()
    device = input_ids.device

    # Determine aux layer IDs for hidden state extraction
    if aux_layer_ids is None:
        draft_config = draft_model.config
        if (
            hasattr(draft_config, "eagle_config")
            and draft_config.eagle_config
            and "eagle_aux_hidden_state_layer_ids" in draft_config.eagle_config
        ):
            aux_layer_ids = draft_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        else:
            num_layers = getattr(draft_config, "num_hidden_layers", 32)
            aux_layer_ids = [num_layers // 3, 2 * num_layers // 3, num_layers]

    generated_tokens = []
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()

    # Prefill phase - get initial hidden states from target
    prefill_start = time.perf_counter()

    with torch.no_grad():
        target_outputs = target_model(
            input_ids=current_ids,
            attention_mask=current_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        target_past_kv = target_outputs.past_key_values
        target_hidden = get_aux_hidden_states(target_outputs.hidden_states, aux_layer_ids)
        next_logits = target_outputs.logits[:, -1, :]

    # Sample first token
    if temperature > 0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_logits.argmax(dim=-1, keepdim=True)

    generated_tokens.append(next_token.item())

    # Update current sequence
    current_ids = torch.cat([current_ids, next_token], dim=1)
    current_mask = torch.cat([current_mask, torch.ones((1, 1), device=device)], dim=1)

    # Get updated hidden states
    with torch.no_grad():
        target_outputs = target_model(
            input_ids=current_ids,
            attention_mask=current_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        target_past_kv = target_outputs.past_key_values
        target_hidden = get_aux_hidden_states(target_outputs.hidden_states, aux_layer_ids)

    prefill_end = time.perf_counter()
    metrics.prefill_time = prefill_end - prefill_start
    metrics.ttft = metrics.prefill_time

    # Decode phase with speculative decoding
    decode_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        if generated_tokens[-1] == tokenizer.eos_token_id:
            break

        metrics.num_spec_steps += 1

        # === DRAFT PHASE ===
        draft_tokens = []
        draft_logits_list = []

        with torch.no_grad():
            # Project target hidden states
            last_hidden = target_hidden[:, -1:, :]
            projected_hidden = draft_model.project_hidden_states(last_hidden)

            # Start from last generated token
            draft_input = next_token
            hidden_states = projected_hidden

            for draft_idx in range(num_draft_tokens):
                # Embed input
                input_embeds = draft_model.embed_input_ids(draft_input)
                input_embeds = input_embeds.to(hidden_states.dtype)

                # Create attention mask for single token
                attn_mask = torch.ones((1, 1), dtype=torch.long, device=device)
                position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

                # Run draft backbone
                hidden_states_out = draft_model.backbone(
                    input_embeds=input_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=[[], []],
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )

                # Get logits
                draft_logits = draft_model.compute_logits(hidden_states_out)
                last_logits = draft_logits[:, -1, :]
                draft_logits_list.append(last_logits)

                if temperature > 0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    draft_token = torch.multinomial(probs, num_samples=1)
                else:
                    draft_token = last_logits.argmax(dim=-1, keepdim=True)

                draft_tokens.append(draft_token.item())

                if draft_token.item() == tokenizer.eos_token_id:
                    break

                # For next iteration
                hidden_states = hidden_states_out[:, -1:, :]
                draft_input = draft_token

        if not draft_tokens:
            break

        metrics.draft_count += len(draft_tokens)

        # === VERIFY PHASE ===
        draft_tensor = torch.tensor([draft_tokens], device=device)
        verify_ids = torch.cat([next_token, draft_tensor], dim=1)

        with torch.no_grad():
            verify_outputs = target_model(
                input_ids=verify_ids,
                attention_mask=torch.ones((1, current_mask.shape[1] + len(draft_tokens)), device=device),
                past_key_values=target_past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            verify_logits = verify_outputs.logits

        # === ACCEPTANCE ===
        accepted_count = 0
        bonus_token = None

        for i, draft_token_id in enumerate(draft_tokens):
            target_logit = verify_logits[:, i, :]

            if temperature > 0:
                target_probs = torch.softmax(target_logit / temperature, dim=-1)
                draft_probs = torch.softmax(draft_logits_list[i] / temperature, dim=-1)

                target_prob = target_probs[0, draft_token_id].item()
                draft_prob = draft_probs[0, draft_token_id].item()

                if draft_prob <= target_prob:
                    generated_tokens.append(draft_token_id)
                    accepted_count += 1
                else:
                    accept_prob = target_prob / draft_prob
                    if torch.rand(1).item() < accept_prob:
                        generated_tokens.append(draft_token_id)
                        accepted_count += 1
                    else:
                        adjusted = torch.clamp(target_probs - draft_probs, min=0)
                        if adjusted.sum() > 0:
                            adjusted = adjusted / adjusted.sum()
                            bonus_token = torch.multinomial(adjusted, 1).item()
                        else:
                            bonus_token = target_logit.argmax(dim=-1).item()
                        break
            else:
                # Greedy verification
                target_token = target_logit.argmax(dim=-1).item()
                if draft_token_id == target_token:
                    generated_tokens.append(draft_token_id)
                    accepted_count += 1
                else:
                    bonus_token = target_token
                    break

            if draft_token_id == tokenizer.eos_token_id:
                break

        # Add bonus token
        if bonus_token is not None and generated_tokens[-1] != tokenizer.eos_token_id:
            generated_tokens.append(bonus_token)
        elif accepted_count == len(draft_tokens) and generated_tokens[-1] != tokenizer.eos_token_id:
            last_verify_logit = verify_logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(last_verify_logit / temperature, dim=-1)
                bonus_token = torch.multinomial(probs, 1).item()
            else:
                bonus_token = last_verify_logit.argmax(dim=-1).item()
            generated_tokens.append(bonus_token)

        metrics.accept_count += accepted_count

        if len(generated_tokens) >= max_new_tokens:
            break
        if generated_tokens[-1] == tokenizer.eos_token_id:
            break

        # Update state for next iteration
        new_tokens = generated_tokens[-(accepted_count + (1 if bonus_token is not None else 0)):]
        new_tokens_tensor = torch.tensor([new_tokens], device=device)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)
        current_mask = torch.cat([current_mask, torch.ones((1, len(new_tokens)), device=device)], dim=1)

        # Update target KV cache and hidden states
        with torch.no_grad():
            target_outputs = target_model(
                input_ids=current_ids,
                attention_mask=current_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            target_past_kv = target_outputs.past_key_values
            target_hidden = get_aux_hidden_states(target_outputs.hidden_states, aux_layer_ids)
            next_token = torch.tensor([[generated_tokens[-1]]], device=device)

    decode_end = time.perf_counter()
    metrics.decode_time = decode_end - decode_start
    metrics.total_time = metrics.prefill_time + metrics.decode_time
    metrics.tokens_generated = len(generated_tokens)

    return generated_tokens, metrics


# ==============================================================================
# Main Evaluation
# ==============================================================================


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("EAGLE3 EVALUATION ON TRAIN.JSONL")
    print("=" * 60)

    # Validate paths
    checkpoint_path = os.path.abspath(args.draft_checkpoint)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"\nDraft model: {checkpoint_path}")
    print(f"Target model: {args.target_model_path}")
    print(f"Data path: {data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Draft tokens per step: {args.num_draft_tokens}")

    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load target model
    print(f"\n🎯 Loading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()

    # Load draft model
    print(f"\n📦 Loading draft model...")
    draft_model = AutoEagle3DraftModel.from_pretrained(
        checkpoint_path,
        attention_backend=args.attention_backend,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    # Load vocab mapping
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

    # Load samples
    print(f"\n📊 Loading samples from {data_path.name}...")
    samples = load_samples_from_jsonl(str(data_path), args.num_samples)
    print(f"   Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("❌ No samples found! Exiting.")
        return

    # Warmup
    if args.warmup > 0:
        print(f"\n🔥 Warming up ({args.warmup} iterations)...")
        for i in range(min(args.warmup, len(samples))):
            input_ids, attention_mask = prepare_prompt(
                tokenizer, samples[i]["conversations"], args.max_prompt_length
            )
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            with torch.no_grad():
                _ = generate_speculative(
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(32, args.max_new_tokens),
                    num_draft_tokens=args.num_draft_tokens,
                    temperature=args.temperature,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("   Warmup complete!")

    # Run evaluation
    print(f"\n🚀 Running speculative decoding evaluation...")
    print("-" * 60)

    all_metrics = []
    all_outputs = []

    for i, sample in enumerate(tqdm(samples, desc="Generating")):
        # Prepare prompt (first user message only)
        input_ids, attention_mask = prepare_prompt(
            tokenizer, sample["conversations"], args.max_prompt_length
        )
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

        # Sync before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Generate
        with torch.no_grad():
            tokens, metrics = generate_speculative(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_draft_tokens=args.num_draft_tokens,
                temperature=args.temperature,
            )

        # Sync after
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Decode output
        output_text = tokenizer.decode(tokens, skip_special_tokens=True)
        all_metrics.append(metrics)
        all_outputs.append(output_text)

        # Show progress
        if args.verbose:
            # Get the user prompt
            user_prompt = ""
            for conv in sample["conversations"]:
                if conv["role"] == "user":
                    user_prompt = conv["content"]
                    break

            print(f"\n{'='*80}")
            print(f"Sample {i+1}/{len(samples)}")
            print(f"{'='*80}")
            print(f"Prompt: {user_prompt[:100]}..." if len(user_prompt) > 100 else f"Prompt: {user_prompt}")
            print(f"\nGenerated Output:\n{output_text}\n")
            print(f"Tokens: {metrics.tokens_generated}")
            print(f"TTFT: {metrics.ttft*1000:.1f}ms")
            print(f"Total time: {metrics.total_time*1000:.1f}ms")
            print(f"Throughput: {metrics.throughput:.1f} tok/s")
            print(f"Accept rate: {metrics.accept_rate:.1%}")
            print(f"Avg accepted/step: {metrics.avg_accepted_per_step:.2f}")
            print(f"{'='*80}\n")

    # Compute aggregate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    avg_ttft = sum(m.ttft for m in all_metrics) / len(all_metrics)
    avg_throughput = sum(m.throughput for m in all_metrics) / len(all_metrics)
    avg_decode_throughput = sum(m.decode_throughput for m in all_metrics) / len(all_metrics)
    avg_latency = sum(m.total_time for m in all_metrics) / len(all_metrics)
    avg_tokens = sum(m.tokens_generated for m in all_metrics) / len(all_metrics)
    avg_accept_rate = sum(m.accept_rate for m in all_metrics if m.draft_count > 0) / len([m for m in all_metrics if m.draft_count > 0])
    avg_accepted_per_step = sum(m.avg_accepted_per_step for m in all_metrics if m.num_spec_steps > 0) / len([m for m in all_metrics if m.num_spec_steps > 0])
    total_tokens = sum(m.tokens_generated for m in all_metrics)
    total_time = sum(m.total_time for m in all_metrics)

    print(f"\n📊 Aggregate Metrics ({len(all_metrics)} samples):")
    print(f"   Avg TTFT:              {avg_ttft*1000:.2f} ms")
    print(f"   Avg Latency:           {avg_latency*1000:.2f} ms")
    print(f"   Avg Throughput:        {avg_throughput:.2f} tok/s")
    print(f"   Avg Decode Throughput: {avg_decode_throughput:.2f} tok/s")
    print(f"   Avg Tokens Generated:  {avg_tokens:.1f}")
    print(f"   Total Tokens:          {total_tokens:,}")
    print(f"   Total Time:            {total_time:.2f}s")

    print(f"\n🎯 Speculative Decoding Metrics:")
    print(f"   Avg Accept Rate:       {avg_accept_rate:.1%}")
    print(f"   Avg Accepted/Step:     {avg_accepted_per_step:.2f}")

    print(f"\n💡 Quality Assessment:")
    if avg_accept_rate >= 0.7:
        print(f"   ✅ EXCELLENT - High acceptance rate indicates strong draft model!")
    elif avg_accept_rate >= 0.5:
        print(f"   ⚠️  GOOD - Decent acceptance rate, expect moderate speedup")
    elif avg_accept_rate >= 0.3:
        print(f"   ⚠️  FAIR - Lower acceptance rate, limited speedup")
    else:
        print(f"   ❌ NEEDS WORK - Low acceptance rate, may need more training")

    # Show sample outputs if not verbose
    if not args.verbose and len(all_outputs) > 0:
        print(f"\n📝 Sample Outputs (first 3):")
        for i in range(min(3, len(all_outputs))):
            user_prompt = ""
            for conv in samples[i]["conversations"]:
                if conv["role"] == "user":
                    user_prompt = conv["content"]
                    break

            print(f"\n  Sample {i+1}:")
            print(f"  Prompt: {user_prompt[:80]}..." if len(user_prompt) > 80 else f"  Prompt: {user_prompt}")
            output_preview = all_outputs[i][:150] + "..." if len(all_outputs[i]) > 150 else all_outputs[i]
            print(f"  Output: {output_preview}")
            print(f"  Metrics: {all_metrics[i].tokens_generated} tokens, {all_metrics[i].accept_rate:.1%} accept rate")

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
