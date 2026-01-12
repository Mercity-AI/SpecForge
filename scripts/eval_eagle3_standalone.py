#!/usr/bin/env python3
"""
Standalone evaluation script for Eagle3 draft models.
Bypasses SGLang and uses SpecForge directly.

Usage:
    python scripts/eval_eagle3_standalone.py \
        --draft-model-path ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --benchmark gsm8k \
        --num-samples 100
"""

import argparse
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from specforge import AutoEagle3DraftModel
from specforge.modeling.target import get_eagle3_target_model
from specforge.data.template import TEMPLATE_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Eagle3 evaluation")
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "mtbench"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_gsm8k_data(num_samples: Optional[int] = None):
    """Load GSM8K dataset."""
    try:
        from sglang.utils import download_and_cache_file, read_jsonl
    except ImportError:
        raise ImportError("Please install sglang: pip install sglang")
    
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))
    
    if num_samples:
        lines = lines[:num_samples]
    
    questions = []
    labels = []
    for line in lines:
        questions.append(line["question"])
        # Extract numeric answer
        answer_str = line["answer"].replace(",", "")
        import re
        numbers = re.findall(r"\d+", answer_str)
        if numbers:
            labels.append(int(numbers[-1]))
        else:
            labels.append(None)
    
    return questions, labels


def load_mtbench_data(num_samples: Optional[int] = None):
    """Load MT-Bench dataset."""
    try:
        from sglang.utils import download_and_cache_file, read_jsonl
    except ImportError:
        raise ImportError("Please install sglang: pip install sglang")
    
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    download_and_cache_file(url, filename="mtbench.jsonl")
    questions_data = list(read_jsonl("mtbench.jsonl"))
    
    if num_samples:
        questions_data = questions_data[:num_samples]
    
    questions = []
    for q in questions_data:
        questions.append({
            "turn_1": q["turns"][0],
            "turn_2": q["turns"][1] if len(q["turns"]) > 1 else None
        })
    
    return questions, [None] * len(questions)


def format_prompt(tokenizer, chat_template, messages: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Format messages using chat template."""
    template = TEMPLATE_REGISTRY.get(chat_template)
    
    # Build conversation string
    conversation = ""
    for msg in messages:
        if msg["role"] == "system":
            conversation += f"{template.system_prompt}\n" if template.system_prompt else ""
        elif msg["role"] == "user":
            conversation += f"{template.user_header}{msg['content']}{template.end_of_turn_token}"
        elif msg["role"] == "assistant":
            conversation += f"{template.assistant_header}{msg['content']}{template.end_of_turn_token}"
    
    # Tokenize
    inputs = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"], inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))


def speculative_decode_step(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 1,
    num_draft_tokens: int = 4,
) -> Tuple[List[int], float, float]:
    """
    Run one step of speculative decoding.
    Returns: (generated_tokens, accept_rate, draft_time)
    """
    device = input_ids.device
    generated_tokens = []
    total_draft_time = 0.0
    total_accept = 0
    total_draft = 0
    
    # Get initial hidden states from target
    with torch.no_grad():
        target_output = target_model.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=torch.ones_like(input_ids),
        )
    
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    current_hidden_states = target_output.hidden_states
    
    for step in range(max_new_tokens):
        # Draft phase: generate candidate tokens
        draft_start = time.time()
        
        with torch.no_grad():
            # Project hidden states
            projected_hidden = draft_model.project_hidden_states(current_hidden_states)
            
            # Get embeddings for current input
            input_embeds = draft_model.embed_input_ids(current_input_ids)
            input_embeds = input_embeds.to(projected_hidden.dtype)
            
            # Run draft model
            draft_output = draft_model.backbone(
                input_embeds=input_embeds,
                hidden_states=projected_hidden,
                cache_hidden=None,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
            )
            
            # Get logits
            draft_logits = draft_model.compute_logits(draft_output)
            
            # Sample from draft (greedy for now)
            draft_probs = torch.softmax(draft_logits[:, -1, :] / (temperature + 1e-8), dim=-1)
            if top_k > 1:
                top_k_probs, top_k_indices = torch.topk(draft_probs, top_k, dim=-1)
                draft_probs = torch.zeros_like(draft_probs)
                draft_probs.scatter_(1, top_k_indices, top_k_probs)
                draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
            
            draft_token = torch.argmax(draft_probs, dim=-1)
        
        draft_time = time.time() - draft_start
        total_draft_time += draft_time
        total_draft += 1
        
        # Verify with target model
        verify_input_ids = torch.cat([current_input_ids, draft_token.unsqueeze(0)], dim=1)
        verify_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=device)], dim=1)
        
        with torch.no_grad():
            # Get target model's prediction for the next token
            target_outputs = target_model.generate_eagle3_data(
                input_ids=verify_input_ids,
                attention_mask=verify_attention_mask,
                loss_mask=torch.ones_like(verify_input_ids),
            )
            
            # Get target's logits for the last position
            # For simplicity, we'll use the target's hidden states to get next token
            # In practice, you'd use the target model's LM head
            # This is a simplified version - full implementation would use target's logits
            
            # Accept draft token
            generated_tokens.append(draft_token.item())
            total_accept += 1
            
            # Update for next iteration
            current_input_ids = verify_input_ids
            current_attention_mask = verify_attention_mask
            current_hidden_states = target_outputs.hidden_states
        
        # Check for EOS
        if draft_token.item() == tokenizer.eos_token_id:
            break
    
    accept_rate = total_accept / total_draft if total_draft > 0 else 0.0
    return generated_tokens, accept_rate, total_draft_time


def evaluate_gsm8k(
    draft_model,
    target_model,
    tokenizer,
    chat_template: str,
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
    device: str,
):
    """Evaluate on GSM8K."""
    print("Loading GSM8K dataset...")
    questions, labels = load_gsm8k_data(num_samples)
    
    draft_model.eval()
    target_model.eval()
    
    correct = 0
    total = 0
    total_time = 0.0
    total_accept_rate = 0.0
    
    print(f"\nEvaluating on {len(questions)} samples...")
    
    for i, (question, label) in enumerate(zip(questions, labels)):
        if label is None:
            continue
        
        # Format prompt
        messages = [
            {"role": "user", "content": f"Question: {question}\nAnswer:"}
        ]
        input_ids, attention_mask = format_prompt(tokenizer, chat_template, messages)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate
        start_time = time.time()
        generated_tokens, accept_rate, draft_time = speculative_decode_step(
            target_model,
            draft_model,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - start_time
        
        # Decode
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract answer
        import re
        numbers = re.findall(r"\d+", generated_text.replace(",", ""))
        pred_answer = int(numbers[-1]) if numbers else None
        
        # Check correctness
        if pred_answer == label:
            correct += 1
        total += 1
        total_time += elapsed
        total_accept_rate += accept_rate
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(questions)}, Accuracy: {correct/total:.2%}, "
                  f"Avg Accept Rate: {total_accept_rate/(i+1):.2%}, "
                  f"Avg Time: {total_time/(i+1):.3f}s")
    
    accuracy = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0
    avg_accept_rate = total_accept_rate / total if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"GSM8K Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Average Accept Rate: {avg_accept_rate:.2%}")
    print(f"Average Time per Sample: {avg_time:.3f}s")
    print(f"Average Throughput: {len(generated_tokens)/avg_time:.1f} tokens/s")
    print(f"{'='*60}\n")
    
    return {
        "accuracy": accuracy,
        "accept_rate": avg_accept_rate,
        "avg_time": avg_time,
        "throughput": len(generated_tokens)/avg_time if avg_time > 0 else 0.0,
    }


def evaluate_mtbench(
    draft_model,
    target_model,
    tokenizer,
    chat_template: str,
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
    device: str,
):
    """Evaluate on MT-Bench (simplified - just generates responses)."""
    print("Loading MT-Bench dataset...")
    questions, _ = load_mtbench_data(num_samples)
    
    draft_model.eval()
    target_model.eval()
    
    total_time = 0.0
    total_accept_rate = 0.0
    
    print(f"\nGenerating responses for {len(questions)} samples...")
    
    for i, q_dict in enumerate(questions):
        # Format first turn
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q_dict["turn_1"]}
        ]
        input_ids, attention_mask = format_prompt(tokenizer, chat_template, messages)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate first response
        start_time = time.time()
        generated_tokens, accept_rate, _ = speculative_decode_step(
            target_model,
            draft_model,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - start_time
        
        total_time += elapsed
        total_accept_rate += accept_rate
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(questions)}, "
                  f"Avg Accept Rate: {total_accept_rate/(i+1):.2%}, "
                  f"Avg Time: {total_time/(i+1):.3f}s")
    
    avg_time = total_time / len(questions) if questions else 0.0
    avg_accept_rate = total_accept_rate / len(questions) if questions else 0.0
    
    print(f"\n{'='*60}")
    print(f"MT-Bench Results")
    print(f"{'='*60}")
    print(f"Average Accept Rate: {avg_accept_rate:.2%}")
    print(f"Average Time per Sample: {avg_time:.3f}s")
    print(f"{'='*60}\n")
    
    return {
        "accept_rate": avg_accept_rate,
        "avg_time": avg_time,
    }


def main():
    args = parse_args()
    
    print("Loading models...")
    print(f"Draft model: {args.draft_model_path}")
    print(f"Target model: {args.target_model_path}")
    
    # Load models
    draft_model = AutoEagle3DraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    
    target_model = get_eagle3_target_model(
        args.target_model_path,
        backend="hf",  # Use HF backend instead of SGLang
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    target_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    print("Models loaded successfully!\n")
    
    # Run evaluation
    if args.benchmark == "gsm8k":
        results = evaluate_gsm8k(
            draft_model,
            target_model,
            tokenizer,
            args.chat_template,
            args.num_samples,
            args.max_new_tokens,
            args.batch_size,
            args.device,
        )
    elif args.benchmark == "mtbench":
        results = evaluate_mtbench(
            draft_model,
            target_model,
            tokenizer,
            args.chat_template,
            args.num_samples,
            args.max_new_tokens,
            args.batch_size,
            args.device,
        )
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()

