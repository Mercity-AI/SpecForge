#!/usr/bin/env python3
"""
Generate Vocab Mapping for EAGLE3

This creates the vocab mapping file (t2d and d2t tensors) needed for evaluation.
The vocab mapping maps between the target model's full vocabulary and the draft
model's reduced vocabulary.

Requirements:
    1. A dataset (JSONL with conversations) - same format used for training
    2. Target model path (for tokenizer and vocab size)
    3. Draft model checkpoint OR draft vocab size

Usage:
    # Using draft checkpoint (gets vocab sizes from config)
    python scripts/generate_vocab_mapping.py \
        --data-path ./cache/dataset/train.jsonl \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --draft-checkpoint ./outputs/qwen-8b-eagle3/epoch_0_step_2000 \
        --output-path ./cache/vocab_mapping/my_vocab_mapping.pt

    # Specifying vocab sizes manually
    python scripts/generate_vocab_mapping.py \
        --data-path ./cache/dataset/train.jsonl \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --target-vocab-size 152064 \
        --draft-vocab-size 60000 \
        --output-path ./vocab_mapping.pt

    # Using a HuggingFace dataset directly
    python scripts/generate_vocab_mapping.py \
        --hf-dataset HuggingFaceH4/ultrachat_200k \
        --hf-split train_sft \
        --target-model-path Qwen/Qwen2.5-7B-Instruct \
        --draft-vocab-size 60000 \
        --num-samples 10000
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from specforge.data.template import TEMPLATE_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Generate vocab mapping for EAGLE3")
    
    # Data source (one of these required)
    data_group = parser.add_argument_group("Data Source (choose one)")
    data_group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (JSONL with conversations)",
    )
    data_group.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., HuggingFaceH4/ultrachat_200k)",
    )
    data_group.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HuggingFace dataset split (default: train)",
    )
    
    # Model config
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Target model path (for tokenizer)",
    )
    model_group.add_argument(
        "--draft-checkpoint",
        type=str,
        default=None,
        help="Draft model checkpoint (to get vocab sizes from config)",
    )
    model_group.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help="Target vocab size (auto-detected from target model if not provided)",
    )
    model_group.add_argument(
        "--draft-vocab-size",
        type=int,
        default=60000,
        help="Draft vocab size (default: 60000)",
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--chat-template",
        type=str,
        default="qwen",
        help="Chat template (default: qwen)",
    )
    proc_group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length for tokenization (default: 2048)",
    )
    proc_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    
    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-path",
        type=str,
        default="./vocab_mapping.pt",
        help="Output path for vocab mapping (default: ./vocab_mapping.pt)",
    )
    
    return parser.parse_args()


def load_conversations_from_jsonl(data_path: str, num_samples: int = None):
    """Load conversations from JSONL file."""
    from datasets import load_dataset
    
    print(f"Loading data from: {data_path}")
    dataset = load_dataset("json", data_files=data_path)["train"]
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    return dataset


def load_conversations_from_hf(dataset_name: str, split: str, num_samples: int = None):
    """Load conversations from HuggingFace dataset."""
    from datasets import load_dataset
    
    print(f"Loading HuggingFace dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Convert streaming to list with limit
    samples = []
    limit = num_samples or 50000  # Default to 50k samples
    
    for i, sample in enumerate(tqdm(dataset, total=limit, desc="Loading")):
        if i >= limit:
            break
        samples.append(sample)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def tokenize_and_count(
    data,
    tokenizer,
    chat_template: str,
    max_length: int,
) -> Counter:
    """Tokenize conversations and count token frequencies."""
    
    template = TEMPLATE_REGISTRY.get(chat_template)
    token_counter = Counter()
    
    # Role mapping for different dataset formats
    role_map = {
        "human": "user",
        "gpt": "assistant", 
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }
    
    print("Tokenizing and counting tokens...")
    
    for sample in tqdm(data, desc="Processing"):
        try:
            # Extract conversations - handle different formats
            conversations = None
            if "conversations" in sample:
                conversations = sample["conversations"]
            elif "messages" in sample:
                conversations = sample["messages"]
            elif "conversation" in sample:
                conversations = sample["conversation"]
            
            if not conversations:
                continue
            
            # Normalize format
            normalized = []
            for msg in conversations:
                if "role" in msg and "content" in msg:
                    role = role_map.get(msg["role"], msg["role"])
                    normalized.append({"role": role, "content": msg["content"]})
                elif "from" in msg and "value" in msg:
                    role = role_map.get(msg["from"], msg["from"])
                    normalized.append({"role": role, "content": msg["value"]})
            
            if len(normalized) < 2:
                continue
            
            # Tokenize
            try:
                text = tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback: manual formatting
                text = ""
                for msg in normalized:
                    if msg["role"] == "user":
                        text += f"{template.user_header}{msg['content']}{template.end_of_turn_token}"
                    elif msg["role"] == "assistant":
                        text += f"{template.assistant_header}{msg['content']}{template.end_of_turn_token}"
            
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
            token_counter.update(tokens)
            
        except Exception as e:
            continue
    
    print(f"Counted {sum(token_counter.values()):,} total tokens")
    print(f"Found {len(token_counter):,} unique tokens")
    
    return token_counter


def generate_vocab_mapping(
    token_counter: Counter,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> dict:
    """Generate t2d and d2t mapping tensors."""
    
    print(f"\nGenerating vocab mapping...")
    print(f"  Target vocab size: {target_vocab_size:,}")
    print(f"  Draft vocab size: {draft_vocab_size:,}")
    
    # Ensure we have enough tokens
    if len(token_counter) < draft_vocab_size:
        print(f"  Adding {draft_vocab_size - len(token_counter)} missing tokens...")
        existing = set(token_counter.keys())
        for token_id in range(draft_vocab_size):
            if token_id not in existing:
                token_counter[token_id] = 0
            if len(token_counter) >= draft_vocab_size:
                break
    
    # Get top N most frequent tokens
    total_freq = sum(token_counter.values())
    top_n = token_counter.most_common(draft_vocab_size)
    top_n_freq = sum(freq for _, freq in top_n)
    
    coverage = top_n_freq / total_freq if total_freq > 0 else 0
    print(f"  Top {draft_vocab_size:,} tokens cover {coverage:.1%} of all tokens")
    
    # Create mappings
    used_tokens = sorted([token_id for token_id, _ in top_n])
    
    # d2t: draft index -> offset to get target index
    # target_token_id = draft_idx + d2t[draft_idx]
    d2t = torch.tensor([used_tokens[i] - i for i in range(len(used_tokens))], dtype=torch.int64)
    
    # t2d: boolean mask - True if target token is in draft vocab
    t2d = torch.tensor([i in used_tokens for i in range(target_vocab_size)], dtype=torch.bool)
    
    print(f"  d2t shape: {d2t.shape}")
    print(f"  t2d shape: {t2d.shape}")
    print(f"  Tokens in draft vocab: {t2d.sum().item():,}")
    
    return {"d2t": d2t, "t2d": t2d}


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("GENERATE VOCAB MAPPING")
    print("=" * 60)
    
    # Validate inputs
    if not args.data_path and not args.hf_dataset:
        raise ValueError("Either --data-path or --hf-dataset must be provided")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.target_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    # Get vocab sizes
    if args.draft_checkpoint:
        from specforge import AutoEagle3DraftModel
        print(f"Loading draft config from: {args.draft_checkpoint}")
        draft_model = AutoEagle3DraftModel.from_pretrained(args.draft_checkpoint)
        target_vocab_size = draft_model.config.vocab_size
        draft_vocab_size = draft_model.config.draft_vocab_size
        del draft_model
    else:
        # Get target vocab size from config
        if args.target_vocab_size:
            target_vocab_size = args.target_vocab_size
        else:
            target_config = AutoConfig.from_pretrained(args.target_model_path)
            target_vocab_size = target_config.vocab_size
        draft_vocab_size = args.draft_vocab_size
    
    print(f"  Target vocab: {target_vocab_size:,}")
    print(f"  Draft vocab: {draft_vocab_size:,}")
    
    # Load data
    if args.data_path:
        data = load_conversations_from_jsonl(args.data_path, args.num_samples)
    else:
        data = load_conversations_from_hf(args.hf_dataset, args.hf_split, args.num_samples)
    
    # Count tokens
    token_counter = tokenize_and_count(
        data=data,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
    )
    
    # Generate mapping
    vocab_mapping = generate_vocab_mapping(
        token_counter=token_counter,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vocab_mapping, output_path)
    
    print(f"\n✅ Saved vocab mapping to: {output_path}")
    print("\nYou can now use this with eval_eagle.py:")
    print(f"  python scripts/eval_eagle.py --vocab-mapping-path {output_path} ...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

