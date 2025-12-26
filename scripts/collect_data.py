import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import uuid

from datasets import load_dataset, Dataset
from tqdm import tqdm


ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}

DATASETS = [
    "nemotron",
    "science-qna", 
    "roleplay",
    "lmsys",
    "sharegpt",
    "ultrachat",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="./processed_data")
    parser.add_argument("--sample-size", type=int, default=None, help="Total samples across all datasets")
    parser.add_argument("--split-eval", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    return parser.parse_args()


# =============================================================================
# Processing functions for each dataset
# =============================================================================

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


def process_science_qna_row(row: Dict) -> Optional[Dict]:
    """169Pi/Science-QnA - prompt/response format"""
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    
    if not prompt or not response:
        return None
    
    conversations = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"id": str(uuid.uuid4()), "conversations": conversations}


def process_roleplay_row(row: Dict) -> Optional[Dict]:
    """agentlans/combined-roleplay - from/value format with system prompt"""
    conversations = []
    
    # Handle both list format and dict with 'conversations' key
    if isinstance(row, dict) and "conversations" in row:
        raw_convos = row["conversations"]
    elif isinstance(row, list):
        raw_convos = row
    else:
        return None
    
    for msg in raw_convos:
        from_role = msg.get("from", "")
        value = msg.get("value", "")
        
        if from_role == "system":
            # Extract character info but keep it shorter
            conversations.append({"role": "system", "content": value[:2000]})  # Truncate long system prompts
        elif from_role in ROLE_MAPPING:
            role = ROLE_MAPPING[from_role]
            # Clean up {{user}} placeholders
            value = value.replace("{{user}}", "User")
            conversations.append({"role": role, "content": value})
    
    if len(conversations) < 2:
        return None
    return {"id": str(uuid.uuid4()), "conversations": conversations}


def process_lmsys_row(row: Dict) -> Optional[Dict]:
    """lmsys/lmsys-chat-1m - already in role/content format"""
    conversations = []
    
    raw_convos = row.get("conversation", [])
    for msg in raw_convos:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            conversations.append({"role": role, "content": content})
    
    if len(conversations) < 2:
        return None
    return {"id": row.get("conversation_id", str(uuid.uuid4())), "conversations": conversations}


def process_sharegpt_row(row: Dict) -> Optional[Dict]:
    """ShareGPT format - from/value"""
    conversations = []
    
    raw_convos = row.get("conversations", [])
    for msg in raw_convos:
        from_role = msg.get("from", "")
        value = msg.get("value", "")
        
        if from_role in ROLE_MAPPING and value:
            conversations.append({"role": ROLE_MAPPING[from_role], "content": value})
    
    if len(conversations) < 2:
        return None
    return {"id": row.get("id", str(uuid.uuid4())), "conversations": conversations}


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


# =============================================================================
# Dataset loading
# =============================================================================

def load_and_process_dataset(name: str, sample_size: int) -> List[Dict]:
    """Load dataset and process to unified format."""
    print(f"\n[INFO] Loading {name} ({sample_size} samples)...")
    
    processed = []
    
    if name == "nemotron":
        ds = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="train", streaming=True)
        proc_fn = process_nemotron_row
        
    elif name == "science-qna":
        ds = load_dataset("169Pi/Science-QnA", split="train", streaming=True)
        proc_fn = process_science_qna_row
        
    elif name == "roleplay":
        ds = load_dataset("agentlans/combined-roleplay", split="train", streaming=True)
        proc_fn = process_roleplay_row
        
    elif name == "lmsys":
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        proc_fn = process_lmsys_row
        
    elif name == "sharegpt":
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        proc_fn = process_sharegpt_row
        
    elif name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        proc_fn = process_ultrachat_row
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Process with progress bar
    count = 0
    skipped = 0
    pbar = tqdm(ds, total=sample_size, desc=f"Processing {name}")
    
    for row in pbar:
        if count >= sample_size:
            break
        
        result = proc_fn(row)
        if result is not None:
            processed.append(result)
            count += 1
        else:
            skipped += 1
        
        pbar.set_postfix({"processed": count, "skipped": skipped})
    
    print(f"[INFO] {name}: {len(processed)} samples processed, {skipped} skipped")
    return processed


def save_jsonl(data: List[Dict], path: Path):
    """Save data to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(data)} samples to {path}")


def main():
    args = parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per dataset
    if args.sample_size is None:
        samples_per_dataset = 10000  # Default
    else:
        samples_per_dataset = args.sample_size // len(DATASETS)
    
    print(f"[INFO] Total sample size: {args.sample_size or samples_per_dataset * len(DATASETS)}")
    print(f"[INFO] Samples per dataset: {samples_per_dataset}")
    print(f"[INFO] Datasets: {DATASETS}")
    
    # Process each dataset
    all_data = []
    for ds_name in DATASETS:
        try:
            data = load_and_process_dataset(ds_name, samples_per_dataset)
            all_data.extend(data)
        except Exception as e:
            print(f"[ERROR] Failed to process {ds_name}: {e}")
            continue
    
    print(f"\n[INFO] Total samples collected: {len(all_data)}")
    
    # Shuffle
    import random
    random.shuffle(all_data)
    
    # Split if needed
    if args.split_eval:
        split_idx = int(len(all_data) * (1 - args.eval_ratio))
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]
        
        save_jsonl(train_data, output_path / "train.jsonl")
        save_jsonl(eval_data, output_path / "eval.jsonl")
    else:
        save_jsonl(all_data, output_path / "train.jsonl")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Total samples: {len(all_data)}")
    if args.split_eval:
        print(f"Train samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")
    
    # Sample preview
    print("\n[INFO] Sample preview:")
    if all_data:
        sample = all_data[0]
        print(f"  ID: {sample['id']}")
        print(f"  Turns: {len(sample['conversations'])}")
        for i, turn in enumerate(sample['conversations'][:3]):
            content_preview = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
            print(f"  [{i}] {turn['role']}: {content_preview}")


if __name__ == "__main__":
    main()