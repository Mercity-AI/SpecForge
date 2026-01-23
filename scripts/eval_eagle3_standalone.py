#!/usr/bin/env python3
"""
EAGLE3 Draft Model Evaluation Script

Simple evaluation on a few prompts to check accuracy and speed.

Usage: 
python scripts/eval_eagle.py 
    --draft-checkpoint ./outputs/epoch_1_step_12528 
    --target-model-path Qwen/Qwen2.5-7B-Instruct 
    --dataset-path ./cache/dataset/train.jsonl 
    --num-samples 10 
    --ttt-length 4 
    --max-length 1024 
    --chat-template qwen 
    --attention-backend flex_attention_mla 
    --verbose
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
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
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key for embeddings in target model checkpoint (default: model.embed_tokens.weight)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode (baseline vs draft comparison on 100 samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results (markdown + JSON)",
    )
    parser.add_argument(
        "--generation-sweep",
        action="store_true",
        help="Run generation length sweep (generates at different lengths: 50, 100, 200, 500, 1000 tokens)",
    )
    parser.add_argument(
        "--generation-lengths",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="Generation lengths to benchmark (default: 50 100 200 500 1000)",
    )
    parser.add_argument(
        "--num-generation-samples",
        type=int,
        default=20,
        help="Number of samples per generation length (default: 20)",
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
# Result Formatting and Export
# ==============================================================================


def format_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None) -> str:
    """Format data as ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    def format_row(row):
        return "| " + " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths)) + " |"
    
    separator = "|-" + "-|-".join("-" * width for width in col_widths) + "-|"
    
    lines = [
        separator,
        format_row(headers),
        separator,
    ]
    for row in rows:
        lines.append(format_row(row))
    lines.append(separator)
    
    return "\n".join(lines)


def save_results_markdown(results: Dict, output_path: str):
    """Save benchmark results as markdown."""
    with open(output_path, "w") as f:
        f.write(f"# EAGLE3 Benchmark Results\n\n")
        f.write(f"**Date**: {results['timestamp']}\n\n")
        f.write(f"**Model**: {results['target_model']}\n\n")
        f.write(f"**Draft Checkpoint**: {results['draft_checkpoint']}\n\n")
        f.write(f"**Dataset**: {results['dataset_path']}\n\n")
        f.write(f"**Samples**: {results['num_samples']}\n\n")
        f.write(f"**TTT Length**: {results['ttt_length']}\n\n")
        
        f.write(f"\n## Summary\n\n")
        summary = results['summary']
        f.write(f"| Metric | Baseline | With Drafter | Speedup |\n")
        f.write(f"|--------|----------|--------------|----------|\n")
        f.write(f"| Total Time | {summary['baseline_time']:.2f}s | {summary['drafter_time']:.2f}s | **{summary['speedup']:.2f}x** |\n")
        f.write(f"| Avg Time/Sample | {summary['baseline_avg_time']:.3f}s | {summary['drafter_avg_time']:.3f}s | {summary['speedup']:.2f}x |\n")
        f.write(f"| Throughput | {summary['baseline_throughput']:.1f} tok/s | {summary['drafter_throughput']:.1f} tok/s | {summary['speedup']:.2f}x |\n")
        f.write(f"| Mean Accuracy | N/A | {summary['mean_accuracy']:.1%} | - |\n")
        
        f.write(f"\n## Per-Position Accuracy\n\n")
        f.write(f"| Position | Accuracy | Loss |\n")
        f.write(f"|----------|----------|------|\n")
        for i, (acc, loss) in enumerate(zip(results['per_position_accuracy'], results['per_position_loss'])):
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            f.write(f"| {i} | {bar} {acc:.1%} | {loss:.4f} |\n")
        
        f.write(f"\n## Sample-by-Sample Timing\n\n")
        f.write(f"| Sample | Baseline (s) | Drafter (s) | Speedup | Tokens |\n")
        f.write(f"|--------|-------------|-------------|---------|--------|\n")
        for i, sample in enumerate(results['per_sample_metrics'][:20]):  # Show first 20
            f.write(f"| {i+1} | {sample['baseline_time']:.3f} | {sample['drafter_time']:.3f} | {sample['speedup']:.2f}x | {sample['tokens']} |\n")
        if len(results['per_sample_metrics']) > 20:
            f.write(f"| ... | ... | ... | ... | ... |\n")


def save_results_json(results: Dict, output_path: str):
    """Save benchmark results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def save_generation_sweep_markdown(results: Dict, output_path: str):
    """Save generation sweep results as markdown."""
    with open(output_path, "w") as f:
        f.write(f"# EAGLE3 Generation Length Sweep\n\n")
        f.write(f"**Date**: {results['timestamp']}\n\n")
        f.write(f"**Model**: {results['target_model']}\n\n")
        f.write(f"**Draft Checkpoint**: {results['draft_checkpoint']}\n\n")
        f.write(f"**Samples per length**: {results['samples_per_length']}\n\n")
        
        f.write(f"\n## Summary\n\n")
        f.write(f"| Generation Length | Baseline Time (s) | Drafter Time (s) | Speedup | Throughput (tok/s) | Acceptance Rate |\n")
        f.write(f"|-------------------|-------------------|------------------|---------|--------------------|-----------------|\n")
        for length_result in results['length_results']:
            f.write(f"| {length_result['length']} tokens | "
                   f"{length_result['baseline_total_time']:.2f}s | "
                   f"{length_result['drafter_total_time']:.2f}s | "
                   f"**{length_result['speedup']:.2f}x** | "
                   f"{length_result['drafter_throughput']:.1f} | "
                   f"{length_result['mean_acceptance_rate']:.1%} |\n")
        
        f.write(f"\n## Speedup vs Generation Length\n\n")
        f.write(f"```\n")
        max_speedup = max(lr['speedup'] for lr in results['length_results'])
        for length_result in results['length_results']:
            bar_len = int((length_result['speedup'] / max_speedup) * 40)
            bar = "█" * bar_len
            f.write(f"{length_result['length']:4d} tokens: {bar} {length_result['speedup']:.2f}x\n")
        f.write(f"```\n")
        
        f.write(f"\n## Detailed Metrics by Generation Length\n\n")
        for length_result in results['length_results']:
            f.write(f"\n### {length_result['length']} Tokens\n\n")
            f.write(f"- **Baseline**: {length_result['baseline_total_time']:.2f}s total, "
                   f"{length_result['baseline_avg_time']:.3f}s/sample, "
                   f"{length_result['baseline_throughput']:.1f} tok/s, "
                   f"{length_result['baseline_avg_target_calls_per_token']:.2f} calls/token\n")
            f.write(f"- **Drafter**: {length_result['drafter_total_time']:.2f}s total, "
                   f"{length_result['drafter_avg_time']:.3f}s/sample, "
                   f"{length_result['drafter_throughput']:.1f} tok/s, "
                   f"{length_result['drafter_avg_target_calls_per_token']:.2f} calls/token\n")
            f.write(f"- **Speedup**: {length_result['speedup']:.2f}x\n")
            f.write(f"- **Acceptance Rate**: {length_result['mean_acceptance_rate']:.1%}\n")


# ==============================================================================
# Speculative Generation Functions
# ==============================================================================


def speculative_generate_one_step(
    draft_model, 
    target_model, 
    input_ids, 
    attention_mask, 
    num_draft_tokens, 
    tokenizer,
    device
):
    """
    One step of speculative decoding with EAGLE3.
    
    Returns:
        new_tokens: List of accepted tokens
        num_target_calls: Number of target model forward passes (should be 1)
        acceptance_rate: Fraction of draft tokens accepted
    """
    from transformers.cache_utils import DynamicCache
    from specforge.utils import padding
    
    # Step 1: Get target model hidden states for current sequence
    with torch.no_grad():
        loss_mask = torch.ones_like(input_ids)
        eagle3_data = target_model.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
        
        # Get target model's next token prediction
        target_logits = eagle3_data.target  # [batch, seq_len, vocab]
        target_next_token = target_logits[:, -1:, :].argmax(dim=-1)  # [batch, 1]
        
    # Step 2: Draft model predicts K tokens ahead using target's hidden states
    draft_tokens = []
    current_ids = input_ids
    current_mask = attention_mask
    hidden_states = draft_model.project_hidden_states(eagle3_data.hidden_states)
    
    # Initialize cache for draft model
    attention_backend = draft_model.config.eagle_config.get("attention_backend", "flex_attention")
    if attention_backend == "sdpa" or attention_backend == "flex_attention_mla":
        cache_hidden = [[], []]
        past_key_values = None
    else:
        cache_hidden = None
        past_key_values = DynamicCache()
    
    seq_length = current_ids.shape[1]
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    for k in range(num_draft_tokens):
        # Embed current token
        inputs_embeds = draft_model.embed_input_ids(current_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        
        # Draft model forward pass
        hidden_states_out = draft_model.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=current_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        # Get draft prediction
        draft_logits = draft_model.compute_logits(hidden_states_out)
        draft_token = draft_logits[:, -1:, :].argmax(dim=-1)  # [batch, 1]
        draft_tokens.append(draft_token)
        
        # Prepare for next iteration
        if k < num_draft_tokens - 1:
            current_ids = padding(current_ids, left=False)
            current_ids[:, -1:] = draft_token
            current_mask = padding(current_mask, left=False)
            position_ids = padding(position_ids, left=False)
            hidden_states = hidden_states_out
    
    # Step 3: Verify draft tokens with target model
    # Build candidate sequence: [original] + [draft_tokens]
    candidate_ids = torch.cat([input_ids] + draft_tokens, dim=1)
    candidate_mask = torch.ones_like(candidate_ids)
    
    with torch.no_grad():
        loss_mask = torch.ones_like(candidate_ids)
        verify_data = target_model.generate_eagle3_data(
            input_ids=candidate_ids,
            attention_mask=candidate_mask,
            loss_mask=loss_mask,
        )
        verify_logits = verify_data.target  # [batch, seq_len, vocab]
    
    # Step 4: Check which draft tokens match target's predictions
    accepted_tokens = [target_next_token]  # Always accept target's first prediction
    num_accepted = 1
    
    # Compare draft tokens with target's predictions
    target_predictions = verify_logits[:, -num_draft_tokens-1:-1, :].argmax(dim=-1)  # [batch, K]
    
    for k in range(num_draft_tokens):
        if draft_tokens[k] == target_predictions[:, k:k+1]:
            accepted_tokens.append(draft_tokens[k])
            num_accepted += 1
        else:
            # Mismatch - use target's prediction and stop
            accepted_tokens.append(target_predictions[:, k:k+1])
            num_accepted += 1
            break
    
    # If all drafts accepted, add one more from target
    if num_accepted == num_draft_tokens + 1:
        final_target_token = verify_logits[:, -1:, :].argmax(dim=-1)
        accepted_tokens.append(final_target_token)
        num_accepted += 1
    
    acceptance_rate = (num_accepted - 1) / num_draft_tokens if num_draft_tokens > 0 else 0
    
    return accepted_tokens, 2, acceptance_rate  # 2 target calls (initial + verify)


def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    num_draft_tokens,
    device,
    eos_token_id=None,
):
    """
    Full speculative generation with EAGLE3.
    
    Returns:
        generated_ids: Full sequence including prompt
        num_generated: Number of new tokens generated
        num_target_calls: Total target model forward passes
        avg_acceptance_rate: Average draft token acceptance rate
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    current_ids = prompt_ids
    attention_mask = torch.ones_like(current_ids)
    num_generated = 0
    num_target_calls = 0
    acceptance_rates = []
    
    while num_generated < max_new_tokens:
        # One step of speculative decoding
        new_tokens, calls, acc_rate = speculative_generate_one_step(
            draft_model,
            target_model,
            current_ids,
            attention_mask,
            num_draft_tokens,
            tokenizer,
            device,
        )
        
        # Concatenate new tokens
        new_tokens_tensor = torch.cat(new_tokens, dim=1)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)
        attention_mask = torch.ones_like(current_ids)
        
        num_generated += new_tokens_tensor.shape[1]
        num_target_calls += calls
        acceptance_rates.append(acc_rate)
        
        # Check for EOS
        if eos_token_id in new_tokens_tensor[0]:
            break
        
        # Safety check
        if num_generated >= max_new_tokens:
            break
    
    avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0
    
    return current_ids, num_generated, num_target_calls, avg_acceptance_rate


def evaluate_generation_baseline(target_model, tokenizer, prompts, gen_length, args, template):
    """Evaluate baseline generation (target model only)."""
    print(f"\n{'='*60}")
    print(f"Baseline Generation: {gen_length} tokens")
    print('='*60)
    
    total_time = 0.0
    total_tokens = 0
    total_target_calls = 0
    sample_times = []
    sample_tokens = []
    sample_target_calls = []
    
    for prompt in tqdm(prompts, desc=f"Baseline ({gen_length} tok)"):
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        if template.system_prompt:
            messages = [{"role": "system", "content": template.system_prompt}] + messages
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}"
        
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        prompt_length = inputs['input_ids'].shape[1]
        
        # Generate autoregressively (standard decoding)
        start = time.time()
        num_target_calls = 0
        
        current_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            for _ in range(gen_length):
                # Target model forward pass
                loss_mask = torch.ones_like(current_ids)
                eagle3_data = target_model.generate_eagle3_data(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                )
                
                # Get next token
                next_token_logits = eagle3_data.target[:, -1:, :]
                next_token = next_token_logits.argmax(dim=-1)
                
                # Append token
                current_ids = torch.cat([current_ids, next_token], dim=1)
                attention_mask = torch.ones_like(current_ids)
                
                num_target_calls += 1
                
                # Check EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        elapsed = time.time() - start
        generated_tokens = current_ids.shape[1] - prompt_length
        
        total_time += elapsed
        total_tokens += generated_tokens
        total_target_calls += num_target_calls
        sample_times.append(elapsed)
        sample_tokens.append(generated_tokens)
        sample_target_calls.append(num_target_calls)
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'total_target_calls': total_target_calls,
        'sample_times': sample_times,
        'sample_tokens': sample_tokens,
        'sample_target_calls': sample_target_calls,
        'avg_time': total_time / len(sample_times) if sample_times else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
        'avg_target_calls_per_token': total_target_calls / total_tokens if total_tokens > 0 else 0,
    }


def evaluate_generation_drafter(draft_model, target_model, tokenizer, prompts, gen_length, args, template):
    """Evaluate generation with EAGLE3 speculative decoding."""
    print(f"\n{'='*60}")
    print(f"Speculative Generation: {gen_length} tokens")
    print('='*60)
    
    total_time = 0.0
    total_tokens = 0
    total_target_calls = 0
    sample_times = []
    sample_tokens = []
    sample_target_calls = []
    acceptance_rates = []
    
    # Number of tokens to draft ahead (typically 3-5)
    num_draft_tokens = min(args.ttt_length - 1, 4)
    
    for prompt in tqdm(prompts, desc=f"Speculative ({gen_length} tok)"):
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        if template.system_prompt:
            messages = [{"role": "system", "content": template.system_prompt}] + messages
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}"
        
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        prompt_length = inputs['input_ids'].shape[1]
        
        # Speculative generation
        start = time.time()
        
        generated_ids, num_generated, num_calls, avg_acc_rate = speculative_generate(
            draft_model,
            target_model,
            tokenizer,
            inputs['input_ids'],
            gen_length,
            num_draft_tokens,
            args.device,
            tokenizer.eos_token_id,
        )
        
        elapsed = time.time() - start
        
        total_time += elapsed
        total_tokens += num_generated
        total_target_calls += num_calls
        sample_times.append(elapsed)
        sample_tokens.append(num_generated)
        sample_target_calls.append(num_calls)
        acceptance_rates.append(avg_acc_rate)
    
    mean_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'total_target_calls': total_target_calls,
        'sample_times': sample_times,
        'sample_tokens': sample_tokens,
        'sample_target_calls': sample_target_calls,
        'avg_time': total_time / len(sample_times) if sample_times else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
        'avg_target_calls_per_token': total_target_calls / total_tokens if total_tokens > 0 else 0,
        'mean_acceptance_rate': mean_acceptance_rate,
    }


# ==============================================================================
# Helper Functions (Dataset Evaluation - Non-Generation)
# ==============================================================================


def evaluate_baseline(target_model, tokenizer, dataset_samples, args, template):
    """Evaluate baseline (target model only, no drafter)."""
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION (Target Model Only)")
    print('='*60)
    
    total_time = 0.0
    total_tokens = 0
    sample_times = []
    sample_tokens = []
    
    for i, sample in enumerate(tqdm(dataset_samples, desc="Baseline")):
        messages, response = extract_conversation(sample)
        
        if len(messages) < 2:
            continue
        
        # Add system prompt if needed
        if template.system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": template.system_prompt}] + messages
        
        # Tokenize
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}{response}{template.end_of_turn_token}"
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=args.max_length, 
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        num_tokens = input_ids.shape[1]
        
        # Time the forward pass
        start = time.time()
        with torch.no_grad():
            # Just run target model forward pass
            _ = target_model.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elapsed = time.time() - start
        
        total_time += elapsed
        total_tokens += num_tokens
        sample_times.append(elapsed)
        sample_tokens.append(num_tokens)
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'sample_times': sample_times,
        'sample_tokens': sample_tokens,
        'avg_time': total_time / len(sample_times) if sample_times else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
    }


def evaluate_with_drafter(eagle3_model, target_model, tokenizer, dataset_samples, args, template):
    """Evaluate with EAGLE3 drafter."""
    print(f"\n{'='*60}")
    print("DRAFTER EVALUATION (EAGLE3 Draft Model)")
    print('='*60)
    
    all_accs = [[] for _ in range(args.ttt_length)]
    all_losses = [[] for _ in range(args.ttt_length)]
    total_time = 0.0
    total_tokens = 0
    sample_times = []
    sample_tokens = []
    sample_accs = []
    
    for i, sample in enumerate(tqdm(dataset_samples, desc="With Drafter")):
        messages, response = extract_conversation(sample)
        
        if len(messages) < 2:
            continue
        
        # Add system prompt if needed
        if template.system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": template.system_prompt}] + messages
        
        # Tokenize
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
            text = f"{template.user_header}{prompt}{template.end_of_turn_token}{template.assistant_header}{response}{template.end_of_turn_token}"
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            max_length=args.max_length, 
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        offsets = inputs["offset_mapping"][0]
        
        # Build loss mask
        loss_mask = torch.zeros(input_ids.shape[1], dtype=torch.long, device=args.device)
        user_sep = f"{template.end_of_turn_token or ''}{template.user_header}"
        asst_sep = f"{template.end_of_turn_token or ''}{template.assistant_header}"
        assistant_pattern = re.escape(asst_sep) + r"(.*?)(?=" + re.escape(user_sep) + "|$)"
        
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_char = match.start(1)
            end_char = match.end(1)
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end > start_char and tok_start <= end_char:
                    loss_mask[idx] = 1
        
        loss_mask = loss_mask.unsqueeze(0)
        num_tokens = input_ids.shape[1]
        
        # Time the forward pass
        start = time.time()
        with torch.no_grad():
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            
            plosses, _, acces = eagle3_model(
                input_ids=eagle3_data.input_ids,
                attention_mask=eagle3_data.attention_mask,
                loss_mask=eagle3_data.loss_mask,
                target=eagle3_data.target,
                hidden_states=eagle3_data.hidden_states,
            )
        elapsed = time.time() - start
        
        # Collect metrics
        for j, (acc, loss) in enumerate(zip(acces, plosses)):
            all_accs[j].append(acc.item())
            all_losses[j].append(loss.item())
        
        mean_acc = sum(a.item() for a in acces) / len(acces)
        
        total_time += elapsed
        total_tokens += num_tokens
        sample_times.append(elapsed)
        sample_tokens.append(num_tokens)
        sample_accs.append(mean_acc)
    
    per_pos_acc = [sum(accs) / len(accs) for accs in all_accs]
    per_pos_loss = [sum(losses) / len(losses) for losses in all_losses]
    mean_acc = sum(per_pos_acc) / len(per_pos_acc)
    mean_loss = sum(per_pos_loss) / len(per_pos_loss)
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'sample_times': sample_times,
        'sample_tokens': sample_tokens,
        'sample_accs': sample_accs,
        'avg_time': total_time / len(sample_times) if sample_times else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
        'per_position_accuracy': per_pos_acc,
        'per_position_loss': per_pos_loss,
        'mean_accuracy': mean_acc,
        'mean_loss': mean_loss,
    }


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

    # In benchmark mode, always use 100 samples
    num_samples_to_load = 100 if args.benchmark else args.num_samples
    dataset_samples = load_dataset_samples(args.dataset_path, num_samples_to_load)
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
    if args.benchmark:
        print("EAGLE3 BENCHMARK MODE (Baseline vs Drafter)")
    else:
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

    # Load embeddings from target model (checkpoint omits them)
    print(f"   Loading embeddings from target model: {args.target_model_path}")
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    print(f"   Embeddings loaded successfully")

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

    # ==============================================================================
    # GENERATION SWEEP MODE
    # ==============================================================================
    if args.generation_sweep:
        print(f"\n📊 Running generation length sweep...")
        print(f"   Lengths: {args.generation_lengths}")
        print(f"   Samples per length: {args.num_generation_samples}")
        
        # Extract prompts from dataset
        prompts = []
        for sample in dataset_samples[:args.num_generation_samples]:
            messages, _ = extract_conversation(sample)
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            if user_msg:
                prompts.append(user_msg)
        
        if len(prompts) < args.num_generation_samples:
            print(f"⚠️  Warning: Only found {len(prompts)} prompts, requested {args.num_generation_samples}")
        
        # Run sweep
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sweep_results = {
            'timestamp': timestamp,
            'target_model': args.target_model_path,
            'draft_checkpoint': args.draft_checkpoint,
            'samples_per_length': len(prompts),
            'ttt_length': args.ttt_length,
            'generation_lengths': args.generation_lengths,
            'length_results': []
        }
        
        for gen_length in args.generation_lengths:
            print(f"\n{'='*60}")
            print(f"Generation Length: {gen_length} tokens")
            print('='*60)
            
            # Baseline
            baseline_results = evaluate_generation_baseline(
                target_model, tokenizer, prompts, gen_length, args, template
            )
            
            # Drafter (pass draft_model directly, not eagle3_model wrapper)
            drafter_results = evaluate_generation_drafter(
                draft_model, target_model, tokenizer, prompts, gen_length, args, template
            )
            
            speedup = baseline_results['total_time'] / drafter_results['total_time'] if drafter_results['total_time'] > 0 else 0
            
            length_result = {
                'length': gen_length,
                'baseline_total_time': baseline_results['total_time'],
                'baseline_avg_time': baseline_results['avg_time'],
                'baseline_throughput': baseline_results['throughput'],
                'baseline_avg_target_calls_per_token': baseline_results['avg_target_calls_per_token'],
                'drafter_total_time': drafter_results['total_time'],
                'drafter_avg_time': drafter_results['avg_time'],
                'drafter_throughput': drafter_results['throughput'],
                'drafter_avg_target_calls_per_token': drafter_results['avg_target_calls_per_token'],
                'mean_acceptance_rate': drafter_results.get('mean_acceptance_rate', 0),
                'speedup': speedup,
            }
            sweep_results['length_results'].append(length_result)
            
            print(f"\n✓ {gen_length} tokens: Baseline={baseline_results['total_time']:.2f}s, "
                  f"Drafter={drafter_results['total_time']:.2f}s, Speedup={speedup:.2f}x, "
                  f"Acceptance={drafter_results['mean_acceptance_rate']:.1%}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("GENERATION SWEEP RESULTS")
        print("=" * 80)
        
        headers = ["Length", "Baseline (s)", "Drafter (s)", "Speedup", "Throughput", "Calls/Token", "Accept Rate"]
        rows = []
        for lr in sweep_results['length_results']:
            rows.append([
                f"{lr['length']} tok",
                f"{lr['baseline_total_time']:.2f}s",
                f"{lr['drafter_total_time']:.2f}s",
                f"{lr['speedup']:.2f}x",
                f"{lr['drafter_throughput']:.1f} t/s",
                f"B:{lr['baseline_avg_target_calls_per_token']:.2f} D:{lr['drafter_avg_target_calls_per_token']:.2f}",
                f"{lr['mean_acceptance_rate']:.1%}"
            ])
        
        print("\n" + format_table(headers, rows))
        
        # Speedup visualization
        print("\n📈 Speedup by Generation Length:")
        max_speedup = max(lr['speedup'] for lr in sweep_results['length_results'])
        for lr in sweep_results['length_results']:
            bar_len = int((lr['speedup'] / max_speedup) * 40)
            bar = "█" * bar_len
            print(f"   {lr['length']:4d} tokens: {bar} {lr['speedup']:.2f}x")
        
        # Save results
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(args.output_dir, f"generation_sweep_{timestamp_str}.md")
        json_path = os.path.join(args.output_dir, f"generation_sweep_{timestamp_str}.json")
        
        save_generation_sweep_markdown(sweep_results, md_path)
        save_results_json(sweep_results, json_path)
        
        print(f"\n💾 Results saved:")
        print(f"   Markdown: {md_path}")
        print(f"   JSON:     {json_path}")
        
        print("\n" + "=" * 80)
        print("✅ Generation sweep complete!")
        print("=" * 80)
        
        return
    
    # ==============================================================================
    # BENCHMARK MODE
    # ==============================================================================
    if args.benchmark:
        print(f"\n📊 Running benchmark on {num_samples} samples...")
        
        # Run baseline evaluation
        baseline_results = evaluate_baseline(target_model, tokenizer, dataset_samples, args, template)
        
        # Run drafter evaluation
        drafter_results = evaluate_with_drafter(eagle3_model, target_model, tokenizer, dataset_samples, args, template)
        
        # Compute summary metrics
        speedup = baseline_results['total_time'] / drafter_results['total_time'] if drafter_results['total_time'] > 0 else 0
        
        # Prepare results dictionary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = {
            'timestamp': timestamp,
            'target_model': args.target_model_path,
            'draft_checkpoint': args.draft_checkpoint,
            'dataset_path': args.dataset_path,
            'num_samples': num_samples,
            'ttt_length': args.ttt_length,
            'max_length': args.max_length,
            'chat_template': args.chat_template,
            'summary': {
                'baseline_time': baseline_results['total_time'],
                'drafter_time': drafter_results['total_time'],
                'speedup': speedup,
                'baseline_avg_time': baseline_results['avg_time'],
                'drafter_avg_time': drafter_results['avg_time'],
                'baseline_throughput': baseline_results['throughput'],
                'drafter_throughput': drafter_results['throughput'],
                'mean_accuracy': drafter_results['mean_accuracy'],
                'mean_loss': drafter_results['mean_loss'],
            },
            'per_position_accuracy': drafter_results['per_position_accuracy'],
            'per_position_loss': drafter_results['per_position_loss'],
            'per_sample_metrics': []
        }
        
        # Per-sample metrics
        for i in range(min(len(baseline_results['sample_times']), len(drafter_results['sample_times']))):
            sample_speedup = baseline_results['sample_times'][i] / drafter_results['sample_times'][i] if drafter_results['sample_times'][i] > 0 else 0
            results['per_sample_metrics'].append({
                'sample_id': i,
                'baseline_time': baseline_results['sample_times'][i],
                'drafter_time': drafter_results['sample_times'][i],
                'speedup': sample_speedup,
                'tokens': drafter_results['sample_tokens'][i],
                'accuracy': drafter_results['sample_accs'][i] if i < len(drafter_results['sample_accs']) else 0,
            })
        
        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        # Summary table
        headers = ["Metric", "Baseline", "With Drafter", "Speedup"]
        rows = [
            ["Total Time", f"{baseline_results['total_time']:.2f}s", f"{drafter_results['total_time']:.2f}s", f"{speedup:.2f}x"],
            ["Avg Time/Sample", f"{baseline_results['avg_time']:.3f}s", f"{drafter_results['avg_time']:.3f}s", f"{speedup:.2f}x"],
            ["Throughput", f"{baseline_results['throughput']:.1f} tok/s", f"{drafter_results['throughput']:.1f} tok/s", f"{speedup:.2f}x"],
            ["Mean Accuracy", "N/A", f"{drafter_results['mean_accuracy']:.1%}", "-"],
        ]
        print("\n" + format_table(headers, rows))
        
        # Per-position accuracy
        print("\n📊 Per-Position Accuracy:")
        for i, (acc, loss) in enumerate(zip(drafter_results['per_position_accuracy'], drafter_results['per_position_loss'])):
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"   Position {i}: {bar} {acc:.1%}  (loss: {loss:.4f})")
        
        # Save results
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(args.output_dir, f"benchmark_{timestamp_str}.md")
        json_path = os.path.join(args.output_dir, f"benchmark_{timestamp_str}.json")
        
        save_results_markdown(results, md_path)
        save_results_json(results, json_path)
        
        print(f"\n💾 Results saved:")
        print(f"   Markdown: {md_path}")
        print(f"   JSON:     {json_path}")
        
        print("\n" + "=" * 80)
        print("✅ Benchmark complete!")
        print("=" * 80)
        
        return
    
    # ==============================================================================
    # NORMAL EVALUATION MODE
    # ==============================================================================
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
            # Show detailed predictions for assistant tokens only
            print(f"\n📊 Detailed Predictions (ASSISTANT TOKENS ONLY - showing first {args.show_first_n_positions} positions):")
            print(f"\nLegend: TTT=Time-to-Target offset (0 means predicting next token, 1 means +1 position ahead, etc.)")
            print(f"\n{'Pos':<4} {'TTT':<3} {'Context (last 3 tokens)':<50} {'Draft':<20} {'Target':<20} {'✓/✗':<3}")
            print("-" * 120)
            
            # Get target tokens (ground truth from target model logits)
            target_tokens = eagle3_data.target.argmax(dim=-1)[0]  # [seq_len]
            input_tokens = input_ids[0].tolist()  # [seq_len]
            loss_mask_flat = loss_mask[0]  # [seq_len]
            
            # Get assistant token positions (where loss_mask == 1)
            assistant_positions = [i for i in range(len(loss_mask_flat)) if loss_mask_flat[i] == 1]
            
            if len(assistant_positions) == 0:
                print("⚠️  No assistant tokens found in loss mask!")
            else:
                print(f"Evaluating {len(assistant_positions)} assistant token positions out of {len(loss_mask_flat)} total tokens\n")
                
                # Calculate accuracy for the window we're about to show
                num_positions_to_show = min(args.show_first_n_positions, len(assistant_positions))
                window_correct = [0] * args.ttt_length
                window_total = [0] * args.ttt_length
                
                for pos_idx in assistant_positions[:num_positions_to_show]:
                    for ttt_idx in range(args.ttt_length):
                        target_pos = pos_idx + ttt_idx
                        if target_pos < len(target_tokens):
                            draft_pred = draft_predictions[ttt_idx][0, pos_idx].item()
                            target_token = target_tokens[target_pos].item()
                            if draft_pred == target_token:
                                window_correct[ttt_idx] += 1
                            window_total[ttt_idx] += 1
                
                window_accs = [window_correct[i] / window_total[i] if window_total[i] > 0 else 0 
                              for i in range(args.ttt_length)]
                window_mean_acc = sum(window_accs) / len(window_accs)
                
                print(f"⚠️  Window Accuracy (positions {assistant_positions[0]}-{assistant_positions[num_positions_to_show-1]}): {window_mean_acc:.1%}")
                print(f"    Overall Accuracy (all {len(assistant_positions)} assistant positions): {mean_acc:.1%}")
                print(f"    {'='*80}")
                
                positions_shown = 0
                
                for pos_idx in assistant_positions[:num_positions_to_show]:
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
                    
                    positions_shown += 1
                    # Add separator between positions
                    if positions_shown < num_positions_to_show and positions_shown % 5 == 0:
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
    print("RESULTS (ASSISTANT TOKENS ONLY)")
    print("=" * 60)

    print("\n📊 Per-Position Accuracy (on assistant tokens):")
    for i, (acc, loss) in enumerate(zip(per_pos_acc, per_pos_loss)):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"   Position {i}: {bar} {acc:.1%}  (loss: {loss:.4f})")

    print("\n📈 Summary:")
    print(f"   Mean Accuracy:    {mean_acc:.1%} (assistant tokens only)")
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
