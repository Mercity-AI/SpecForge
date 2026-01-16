#!/usr/bin/env python3
"""
Standalone evaluation script for Eagle3 draft models.
Compares baseline (target-only) vs speculative decoding (target + draft).

Metrics:
  - Time to First Token (TTFT)
  - Generation latency (total time)
  - Throughput (tokens/second)
  - Accept rate (for speculative decoding)

Usage:
    python scripts/eval_eagle3_standalone.py \
        --draft-model-path ./outputs/eagle3-mla/epoch_4_step_12000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --num-samples 20
"""

import argparse
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import AutoEagle3DraftModel
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


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


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    name: str
    metrics: List[TimingMetrics] = field(default_factory=list)
    
    @property
    def avg_ttft(self) -> float:
        return sum(m.ttft for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_throughput(self) -> float:
        return sum(m.throughput for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_decode_throughput(self) -> float:
        return sum(m.decode_throughput for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_latency(self) -> float:
        return sum(m.total_time for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_tokens(self) -> float:
        return sum(m.tokens_generated for m in self.metrics) / len(self.metrics) if self.metrics else 0.0
    
    @property
    def avg_accept_rate(self) -> float:
        rates = [m.accept_rate for m in self.metrics if m.draft_count > 0]
        return sum(rates) / len(rates) if rates else 0.0
    
    @property
    def avg_accepted_per_step(self) -> float:
        vals = [m.avg_accepted_per_step for m in self.metrics if m.num_spec_steps > 0]
        return sum(vals) / len(vals) if vals else 0.0
    
    @property
    def total_tokens(self) -> int:
        return sum(m.tokens_generated for m in self.metrics)
    
    @property
    def total_time(self) -> float:
        return sum(m.total_time for m in self.metrics)


# ==============================================================================
# Dataset Configuration
# ==============================================================================

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}

AVAILABLE_DATASETS = ["ultrachat", "sharegpt", "nemotron"]


# ==============================================================================
# Argument Parsing
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Eagle3 Benchmark: Baseline vs Speculative")
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ultrachat"],
        choices=AVAILABLE_DATASETS,
        help="Datasets to evaluate on",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Samples per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="Draft tokens per step")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (target-only) benchmark",
    )
    parser.add_argument(
        "--skip-speculative", 
        action="store_true",
        help="Skip speculative decoding benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    return parser.parse_args()


# ==============================================================================
# Dataset Loading Functions
# ==============================================================================


def process_ultrachat_row(row: Dict) -> Optional[Dict]:
    """HuggingFaceH4/ultrachat_200k - messages format"""
    conversations = []
    for msg in row.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            conversations.append({"role": role, "content": content})
    
    if len(conversations) < 2:
        return None
    return {"id": row.get("prompt_id", str(uuid.uuid4())), "conversations": conversations}


def process_sharegpt_row(row: Dict) -> Optional[Dict]:
    """Aeala/ShareGPT_Vicuna_unfiltered - ShareGPT format"""
    conversations = []
    for msg in row.get("conversations", []):
        from_role = msg.get("from", "")
        value = msg.get("value", "")
        if from_role in ROLE_MAPPING and value:
            role = ROLE_MAPPING[from_role]
            conversations.append({"role": role, "content": value})
    
    if len(conversations) < 2:
        return None
    return {"id": row.get("id", str(uuid.uuid4())), "conversations": conversations}


def process_nemotron_row(row: Dict) -> Optional[Dict]:
    """nvidia/Nemotron-Instruction-Following-Chat-v1"""
    conversations = []
    for msg in row.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"] and content:
            conversations.append({"role": role, "content": content})
    
    if len(conversations) < 2:
        return None
    return {"id": str(uuid.uuid4()), "conversations": conversations}


def load_dataset_samples(dataset_name: str, num_samples: int) -> List[Dict]:
    """Load samples from a dataset using streaming."""
    print(f"Loading {dataset_name} dataset (streaming)...")
    
    if dataset_name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        proc_fn = process_ultrachat_row
    elif dataset_name == "sharegpt":
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        proc_fn = process_sharegpt_row
    elif dataset_name == "nemotron":
        ds = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="chat_if", streaming=True)
        proc_fn = process_nemotron_row
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    samples = []
    for row in ds:
        if len(samples) >= num_samples:
            break
        result = proc_fn(row)
        if result is not None:
            samples.append(result)
    
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
# Baseline Generation (Target Model Only)
# ==============================================================================


def generate_baseline(
    target_model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[List[int], TimingMetrics]:
    """
    Standard autoregressive generation with the target model.
    Returns generated tokens and timing metrics.
    """
    metrics = TimingMetrics()
    device = input_ids.device
    
    generated_tokens = []
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    
    # Prefill phase
    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = target_model(
            input_ids=current_ids,
            attention_mask=current_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
    
    # Sample first token
    if temperature > 0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_logits.argmax(dim=-1, keepdim=True)
    
    prefill_end = time.perf_counter()
    metrics.prefill_time = prefill_end - prefill_start
    metrics.ttft = metrics.prefill_time
    
    generated_tokens.append(next_token.item())
    
    # Decode phase
    decode_start = time.perf_counter()
    
    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        with torch.no_grad():
            outputs = target_model(
                input_ids=next_token,
                attention_mask=torch.ones((1, current_mask.shape[1] + len(generated_tokens)), device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
        
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        
        generated_tokens.append(next_token.item())
    
    decode_end = time.perf_counter()
    metrics.decode_time = decode_end - decode_start
    metrics.total_time = metrics.prefill_time + metrics.decode_time
    metrics.tokens_generated = len(generated_tokens)
    
    return generated_tokens, metrics


# ==============================================================================
# Speculative Decoding Generation
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
        if layer_id < len(hidden_states_tuple):
            aux_hidden.append(hidden_states_tuple[layer_id])
    
    if not aux_hidden:
        # Fallback: use last layer repeated
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
    Speculative decoding using Eagle3's TTT-style draft prediction.
    
    Key insight from training code:
    - Draft model uses TARGET hidden states (not its own output)
    - Uses padding() to shift input_ids for predicting positions t+1, t+2, ...
    - All draft predictions share the same target hidden state context
    
    Returns generated tokens and timing metrics.
    """
    from specforge.utils import padding
    
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
    
    # Get updated hidden states that include the first generated token
    # This is necessary because draft model needs hidden states matching current_ids length
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
        
        # === DRAFT PHASE (TTT-style) ===
        # Generate draft tokens using Eagle3's TTT loop pattern
        # Key: use TARGET hidden states throughout, with padding for each position
        draft_tokens = []
        draft_logits_list = []
        
        with torch.no_grad():
            # Project target hidden states (full sequence, not just last position)
            projected_hidden = draft_model.project_hidden_states(target_hidden)
            
            # Initialize for TTT loop
            draft_input_ids = current_ids.clone()
            draft_attn_mask = current_mask.clone()
            seq_len = draft_input_ids.shape[1]
            
            # Position IDs for full sequence
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            
            # Hidden states for TTT loop (starts with projected target hidden states)
            hidden_states = projected_hidden
            
            for ttt_idx in range(num_draft_tokens):
                # Embed input ids
                input_embeds = draft_model.embed_input_ids(draft_input_ids)
                input_embeds = input_embeds.to(hidden_states.dtype)
                
                # Run draft backbone (matching eval_eagle.py pattern)
                hidden_states_out = draft_model.backbone(
                    input_embeds=input_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=[[], []],
                    attention_mask=draft_attn_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                
                # Update hidden states for next iteration
                hidden_states = hidden_states_out
                
                # Get logits and sample from last position
                draft_logits = draft_model.compute_logits(hidden_states)
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
        
                # Pad for next TTT iteration (matching training pattern)
                # Must pad ALL sequence-length tensors to keep dimensions aligned
                if ttt_idx < num_draft_tokens - 1:
                    draft_input_ids = padding(draft_input_ids, left=False)
                    position_ids = padding(position_ids, left=False)
                    hidden_states = padding(hidden_states, left=False)
                    # Pad attention mask (add 1 for new position)
                    draft_attn_mask = torch.cat([
                        draft_attn_mask, 
                        torch.ones((1, 1), dtype=draft_attn_mask.dtype, device=device)
                    ], dim=1)
        
        if not draft_tokens:
            break
            
        metrics.draft_count += len(draft_tokens)
        
        # === VERIFY PHASE ===
        # Verify all draft tokens in one target model pass
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
                    # Accept
                    generated_tokens.append(draft_token_id)
                    accepted_count += 1
                else:
                    accept_prob = target_prob / draft_prob
                    if torch.rand(1).item() < accept_prob:
                        # Probabilistic accept
                        generated_tokens.append(draft_token_id)
                        accepted_count += 1
                    else:
                        # Reject - sample bonus token from adjusted distribution
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
                    # Reject - target's token is the bonus token
                    bonus_token = target_token
                    break
            
            if draft_token_id == tokenizer.eos_token_id:
                break
        
        # Add bonus token if we rejected (standard spec decoding guarantees 1 token progress)
        if bonus_token is not None and generated_tokens[-1] != tokenizer.eos_token_id:
            generated_tokens.append(bonus_token)
        # If all accepted, get bonus token from last verify position
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
        # Calculate how many new tokens were added this step
        new_tokens = generated_tokens[-(accepted_count + (1 if bonus_token is not None else 0)):]
        new_tokens_tensor = torch.tensor([new_tokens], device=device)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)
        current_mask = torch.cat([current_mask, torch.ones((1, len(new_tokens)), device=device)], dim=1)
        
        # Update target KV cache and hidden states
        # Always recompute to get correct hidden states for next draft phase
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
# Benchmark Runner
# ==============================================================================


def run_benchmark(
    name: str,
    generate_fn,
    samples: List[Dict],
    tokenizer,
    max_prompt_length: int,
    max_new_tokens: int,
    warmup: int = 2,
    verbose: bool = False,
    **generate_kwargs,
) -> BenchmarkResults:
    """Run benchmark on all samples."""
    results = BenchmarkResults(name=name)
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Warmup
    if warmup > 0 and samples:
        print(f"Warming up ({warmup} iterations)...")
        for i in range(min(warmup, len(samples))):
            input_ids, attention_mask = prepare_prompt(
                tokenizer, samples[i]["conversations"], max_prompt_length
            )
            input_ids = input_ids.to(generate_kwargs.get("device", "cuda"))
            attention_mask = attention_mask.to(generate_kwargs.get("device", "cuda"))
            
            with torch.no_grad():
                _ = generate_fn(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(32, max_new_tokens),
                    **{k: v for k, v in generate_kwargs.items() if k != "device"}
                )
        
        # Clear CUDA cache after warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    # Actual benchmark
    for i, sample in enumerate(tqdm(samples, desc=name)):
        input_ids, attention_mask = prepare_prompt(
            tokenizer, sample["conversations"], max_prompt_length
        )
        input_ids = input_ids.to(generate_kwargs.get("device", "cuda"))
        attention_mask = attention_mask.to(generate_kwargs.get("device", "cuda"))
        
        # Sync before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        with torch.no_grad():
            tokens, metrics = generate_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
                **{k: v for k, v in generate_kwargs.items() if k != "device"}
            )
        
        # Sync after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        results.metrics.append(metrics)
        
        if verbose and i < 3:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"\n  Sample {i+1}:")
            print(f"    Tokens: {metrics.tokens_generated}")
            print(f"    TTFT: {metrics.ttft*1000:.1f}ms")
            print(f"    Total: {metrics.total_time*1000:.1f}ms")
            print(f"    Throughput: {metrics.throughput:.1f} tok/s")
            if metrics.draft_count > 0:
                print(f"    Accept rate: {metrics.accept_rate:.1%}")
            print(f"    Output: {text[:100]}...")
    
    return results


def print_results(baseline: Optional[BenchmarkResults], speculative: Optional[BenchmarkResults]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    headers = ["Metric", "Baseline", "Speculative", "Speedup"]
    col_widths = [25, 15, 15, 12]
    
    def fmt_row(cols):
        return " | ".join(f"{c:<{w}}" for c, w in zip(cols, col_widths))
    
    print(fmt_row(headers))
    print("-" * 80)
    
    def fmt_val(val, unit=""):
        if isinstance(val, float):
            if unit == "ms":
                return f"{val*1000:.2f} ms"
            elif unit == "%":
                return f"{val:.1%}"
            else:
                return f"{val:.2f}{unit}"
        return str(val)
    
    def speedup(base, spec):
        if base and spec and base > 0:
            return f"{spec/base:.2f}x"
        return "-"
    
    metrics = [
        ("Samples", lambda r: len(r.metrics), ""),
        ("Avg Tokens Generated", lambda r: r.avg_tokens, ""),
        ("Time to First Token", lambda r: r.avg_ttft, "ms"),
        ("Avg Total Latency", lambda r: r.avg_latency, "ms"),
        ("Avg Throughput", lambda r: r.avg_throughput, " tok/s"),
        ("Decode Throughput", lambda r: r.avg_decode_throughput, " tok/s"),
        ("Total Tokens", lambda r: r.total_tokens, ""),
        ("Total Time", lambda r: r.total_time, "s"),
    ]
    
    for name, fn, unit in metrics:
        base_val = fn(baseline) if baseline else None
        spec_val = fn(speculative) if speculative else None
        
        base_str = fmt_val(base_val, unit) if base_val is not None else "-"
        spec_str = fmt_val(spec_val, unit) if spec_val is not None else "-"
        
        # Speedup calculation
        if name in ["Avg Throughput", "Decode Throughput"]:
            sp = speedup(base_val, spec_val) if (base_val and spec_val) else "-"
        elif name in ["Time to First Token", "Avg Total Latency"]:
            sp = speedup(spec_val, base_val) if (base_val and spec_val) else "-"  # Inverted
        else:
            sp = "-"
        
        print(fmt_row([name, base_str, spec_str, sp]))
    
    # Speculative-specific metrics
    if speculative:
        print("-" * 80)
        print(fmt_row(["Accept Rate", "-", fmt_val(speculative.avg_accept_rate, "%"), "-"]))
        print(fmt_row(["Avg Accepted/Step", "-", f"{speculative.avg_accepted_per_step:.2f}", "-"]))
    
    print("=" * 80)
    
    # Summary
    if baseline and speculative:
        throughput_speedup = speculative.avg_throughput / baseline.avg_throughput if baseline.avg_throughput > 0 else 0
        latency_speedup = baseline.avg_latency / speculative.avg_latency if speculative.avg_latency > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Throughput improvement: {throughput_speedup:.2f}x")
        print(f"   Latency improvement: {latency_speedup:.2f}x")
        print(f"   Accept rate: {speculative.avg_accept_rate:.1%}")
        
        if throughput_speedup >= 2.0:
            print(f"\n   ✅ EXCELLENT - Speculative decoding is {throughput_speedup:.1f}x faster!")
        elif throughput_speedup >= 1.5:
            print(f"\n   ⚠️  GOOD - Speculative decoding is {throughput_speedup:.1f}x faster")
        elif throughput_speedup >= 1.1:
            print(f"\n   ⚠️  MARGINAL - Only {throughput_speedup:.1f}x speedup, consider tuning")
        else:
            print(f"\n   ❌ NO BENEFIT - Speculative decoding is not helping")


# ==============================================================================
# Main
# ==============================================================================


def main():
    args = parse_args()
    
    # Validate checkpoint path
    checkpoint_path = os.path.abspath(args.draft_model_path)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("EAGLE3 SPECULATIVE DECODING BENCHMARK")
    print("=" * 60)
    print(f"Draft model: {checkpoint_path}")
    print(f"Target model: {args.target_model_path}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Draft tokens per step: {args.num_draft_tokens}")
    print(f"Temperature: {args.temperature}")
    
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
    
    # Load draft model (if needed)
    draft_model = None
    if not args.skip_speculative:
        print(f"\n📦 Loading draft model...")
        draft_model = AutoEagle3DraftModel.from_pretrained(
            checkpoint_path,
            attention_backend="sdpa",  # Use SDPA for simplicity
            torch_dtype=torch.bfloat16,
        ).to(args.device).eval()
        
        # Load vocab mapping
        vocab_mapping_path = None
        if os.path.exists(os.path.join(checkpoint_path, "vocab_mapping.pt")):
            vocab_mapping_path = os.path.join(checkpoint_path, "vocab_mapping.pt")
        else:
            for cache_dir in ["./cache/vocab_mapping", "../cache/vocab_mapping"]:
                if os.path.isdir(cache_dir):
                    for f in os.listdir(cache_dir):
                        if f.endswith(".pt"):
                            vocab_mapping_path = os.path.join(cache_dir, f)
                            break
                if vocab_mapping_path:
                    break
        
        if vocab_mapping_path:
            draft_model.load_vocab_mapping(vocab_mapping_path)
            print(f"   Loaded vocab mapping from: {vocab_mapping_path}")
    
    # Load samples
    all_samples = []
    for dataset_name in args.datasets:
        samples = load_dataset_samples(dataset_name, args.num_samples)
        all_samples.extend(samples)
        print(f"Loaded {len(samples)} samples from {dataset_name}")
    
    print(f"\n✅ Loaded {len(all_samples)} total samples")
    
    # Run benchmarks
    baseline_results = None
    speculative_results = None
    
    if not args.skip_baseline:
        baseline_results = run_benchmark(
            name="Baseline (Target Only)",
            generate_fn=lambda **kw: generate_baseline(
                target_model=target_model,
                tokenizer=tokenizer,
                **kw
            ),
            samples=all_samples,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            verbose=args.verbose,
            device=args.device,
            temperature=args.temperature,
        )
    
    if not args.skip_speculative and draft_model is not None:
        speculative_results = run_benchmark(
            name="Speculative (Target + Draft)",
            generate_fn=lambda **kw: generate_speculative(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                num_draft_tokens=args.num_draft_tokens,
                **kw
            ),
            samples=all_samples,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            verbose=args.verbose,
            device=args.device,
            temperature=args.temperature,
        )
    
    # Print results
    print_results(baseline_results, speculative_results)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
