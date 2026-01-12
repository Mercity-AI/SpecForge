#!/usr/bin/env python3
"""
Simple evaluation script to test draft model quality without full speculative decoding.
Measures position-wise accuracy which directly correlates with acceptance rate.

Usage:
    python scripts/eval_draft_quality.py \
        --draft-model-path ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --eval-data-path ./cache/dataset/eval.jsonl \
        --num-samples 1000
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoEagle3DraftModel
from specforge.modeling.target import get_eagle3_target_model
from specforge.data import build_eagle3_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate draft model quality")
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, required=True)
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate_draft_quality(
    draft_model,
    target_model,
    eval_dataset,
    num_samples: Optional[int],
    batch_size: int,
    device: str,
):
    """Evaluate draft model quality by measuring position-wise accuracy."""
    draft_model.eval()
    target_model.eval()
    
    # Statistics
    position_accuracies = [[] for _ in range(7)]  # Assuming TTT length of 7
    position_losses = [[] for _ in range(7)]
    total_samples = 0
    
    print(f"\nEvaluating draft model quality...")
    print(f"Dataset size: {len(eval_dataset)}")
    if num_samples:
        print(f"Evaluating on {num_samples} samples")
    
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    for batch_idx, data in enumerate(tqdm(dataloader, desc="Evaluating")):
        if num_samples and total_samples >= num_samples:
            break
        
        # Move to device
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        loss_mask = data["loss_mask"].to(device)
        
        # Get target model's hidden states and targets
        with torch.no_grad():
            target_output = target_model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
        
        # Run draft model forward (simulating training forward)
        with torch.no_grad():
            # This simulates what happens during training
            # We'll use a simplified version that just checks accuracy at each position
            
            # Project hidden states
            projected_hidden = draft_model.project_hidden_states(target_output.hidden_states)
            
            # Get embeddings
            input_embeds = draft_model.embed_input_ids(input_ids)
            input_embeds = input_embeds.to(projected_hidden.dtype)
            
            # Run draft model backbone
            draft_output = draft_model.backbone(
                input_embeds=input_embeds,
                hidden_states=projected_hidden,
                cache_hidden=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
            )
            
            # Get logits
            draft_logits = draft_model.compute_logits(draft_output)
            
            # Compute accuracy at each position
            # This is a simplified version - full implementation would use TTT loop
            target_ids = target_output.target
            
            # For each position in the sequence where we have a target
            seq_len = input_ids.shape[1]
            for pos in range(min(seq_len - 1, 7)):  # Check up to 7 positions
                if pos < draft_logits.shape[1] - 1:
                    # Get predictions for position pos+1
                    pred_logits = draft_logits[:, pos, :]
                    pred_token = torch.argmax(pred_logits, dim=-1)
                    
                    # Get target token at position pos+1
                    if pos + 1 < target_ids.shape[1]:
                        target_token = target_ids[:, pos + 1]
                        
                        # Check if prediction matches target
                        correct = (pred_token == target_token).float()
                        
                        # Only count where loss_mask is active
                        if pos + 1 < loss_mask.shape[1]:
                            mask = loss_mask[:, pos + 1]
                            correct = correct * mask
                            
                            if mask.sum() > 0:
                                accuracy = correct.sum() / mask.sum()
                                position_accuracies[pos].append(accuracy.item())
        
        total_samples += batch_size
    
    # Compute average accuracies
    avg_accuracies = []
    for pos_accs in position_accuracies:
        if pos_accs:
            avg_accuracies.append(sum(pos_accs) / len(pos_accs))
        else:
            avg_accuracies.append(0.0)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Draft Model Quality Evaluation")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"\nPosition-wise Accuracy:")
    for i, acc in enumerate(avg_accuracies):
        print(f"  Position {i}: {acc:.2%}")
    
    # Compute average across positions
    overall_avg = sum(avg_accuracies) / len(avg_accuracies) if avg_accuracies else 0.0
    print(f"\nOverall Average Accuracy: {overall_avg:.2%}")
    
    # Estimate acceptance rate (simplified: higher accuracy = higher acceptance)
    # In practice, acceptance rate depends on the verification mechanism
    print(f"\nEstimated Acceptance Rate: ~{overall_avg*100:.1f}% (simplified estimate)")
    print(f"{'='*60}\n")
    
    return {
        "position_accuracies": avg_accuracies,
        "overall_accuracy": overall_avg,
        "total_samples": total_samples,
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
        backend="hf",
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    target_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    print("Loading evaluation dataset...")
    from datasets import load_dataset
    eval_dataset_raw = load_dataset("json", data_files=args.eval_data_path)["train"]
    
    if args.num_samples:
        eval_dataset_raw = eval_dataset_raw.select(range(min(args.num_samples, len(eval_dataset_raw))))
    
    # Build Eagle3 dataset (same as training)
    eval_dataset = build_eagle3_dataset(
        eval_dataset_raw,
        tokenizer,
        args.chat_template,
        args.max_length,
        is_vlm=False,
        is_preformatted=False,
    )
    
    print("Models and dataset loaded successfully!\n")
    
    # Run evaluation
    results = evaluate_draft_quality(
        draft_model,
        target_model,
        eval_dataset,
        args.num_samples,
        args.batch_size,
        args.device,
    )
    
    print("Evaluation complete!")
    print(f"\nResults summary:")
    print(f"  Overall accuracy: {results['overall_accuracy']:.2%}")
    print(f"  Position 0 accuracy: {results['position_accuracies'][0]:.2%}")
    print(f"  Position 1-3 average: {sum(results['position_accuracies'][1:4])/3:.2%}")


if __name__ == "__main__":
    main()

