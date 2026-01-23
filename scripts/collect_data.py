import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import uuid

from datasets import load_dataset
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
    "soda",
    "lmsys",
    "ultrachat",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Total samples across all datasets")
    parser.add_argument("--split-eval", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    return parser.parse_args()


# =============================================================================
# Processing functions for each dataset
# =============================================================================

def process_nemotron_row(row: Dict) -> Tuple[Optional[Dict], int]:
    """nvidia/Nemotron-Instruction-Following-Chat-v1"""
    conversations = []
    skipped = 0
    for msg in row.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"] and content:
            conversations.append({"role": role, "content": content})
        else:
            skipped += 1
    
    if len(conversations) < 2:
        return None, skipped
    return {"id": str(uuid.uuid4()), "conversations": conversations}, skipped


def process_science_qna_row(row: Dict) -> Tuple[Optional[Dict], int]:
    """169Pi/Science-QnA - prompt/response format"""
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    
    if not prompt or not response:
        return None, 1
    
    conversations = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"id": str(uuid.uuid4()), "conversations": conversations}, 0


def process_roleplay_row(row: Dict) -> Tuple[Optional[Dict], int]:
    """agentlans/combined-roleplay - from/value format with system prompt"""
    conversations = []
    skipped = 0
    
    if isinstance(row, dict) and "conversations" in row:
        raw_convos = row["conversations"]
    elif isinstance(row, list):
        raw_convos = row
    else:
        return None, 1
    
    for msg in raw_convos:
        from_role = msg.get("from", "")
        value = msg.get("value", "")
        
        if from_role == "system":
            conversations.append({"role": "system", "content": value[:2000]})
        elif from_role in ROLE_MAPPING:
            role = ROLE_MAPPING[from_role]
            value = value.replace("{{user}}", "User")
            conversations.append({"role": role, "content": value})
        else:
            skipped += 1
    
    if len(conversations) < 2:
        return None, skipped
    return {"id": str(uuid.uuid4()), "conversations": conversations}, skipped


def process_lmsys_row(row: Dict) -> Tuple[Optional[Dict], int]:
    """lmsys/lmsys-chat-1m - already in role/content format"""
    conversations = []
    skipped = 0
    
    raw_convos = row.get("conversation", [])
    for msg in raw_convos:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            conversations.append({"role": role, "content": content})
        else:
            skipped += 1
    
    if len(conversations) < 2:
        return None, skipped
    return {"id": row.get("conversation_id", str(uuid.uuid4())), "conversations": conversations}, skipped


def process_ultrachat_row(row: Dict) -> Tuple[Optional[Dict], int]:
    """HuggingFaceH4/ultrachat_200k - messages format"""
    conversations = []
    skipped = 0
    
    for msg in row.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            conversations.append({"role": role, "content": content})
        else:
            skipped += 1
    
    if len(conversations) < 2:
        return None, skipped
    return {"id": row.get("prompt_id", str(uuid.uuid4())), "conversations": conversations}, skipped


# =============================================================================
# Dataset loading and processing
# =============================================================================

def get_dataset_and_proc_fn(name: str):
    """Return dataset and processing function for a given dataset name."""
    if name == "nemotron":
        ds = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="chat_if", streaming=True)
        return ds, process_nemotron_row
        
    elif name == "science-qna":
        ds = load_dataset("169Pi/Science-QnA", split="train", streaming=True)
        return ds, process_science_qna_row
        
    elif name == "soda":
        ds = load_dataset("agentlans/combined-roleplay", "soda", split="train", streaming=True)
        return ds, process_roleplay_row
        
    elif name == "lmsys":
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        return ds, process_lmsys_row
        
    elif name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        return ds, process_ultrachat_row
        
    else:
        raise ValueError(f"Unknown dataset: {name}")


def process_and_save_ds(output_path: Path, dataset_name: str, sample_size: int, split_eval: bool, eval_ratio: float):
    """Process dataset and save to JSONL files."""
    
    train_output_path = output_path / f"{dataset_name}_train.jsonl"
    if train_output_path.exists():
        print(f"[INFO] {dataset_name} already processed at {train_output_path}, skipping...")
        return
    
    print(f"\n[INFO] Loading {dataset_name} ({sample_size} samples)...")
    
    try:
        ds, proc_fn = get_dataset_and_proc_fn(dataset_name)
    except Exception as e:
        print(f"[ERROR] Failed to load {dataset_name}: {e}")
        return
    
    # Collect samples
    all_samples = []
    total_skipped = 0
    count = 0
    
    pbar = tqdm(ds, total=sample_size, desc=f"Processing {dataset_name}")
    for row in pbar:
        if count >= sample_size:
            break
        
        result, skipped = proc_fn(row)
        total_skipped += skipped
        
        if result is not None:
            all_samples.append(result)
            count += 1
        
        pbar.set_postfix({"processed": count, "skipped": total_skipped})
    
    if total_skipped > 0:
        print(f"[INFO] Skipped {total_skipped} messages for {dataset_name}")
    
    # Split if needed
    if split_eval and len(all_samples) > 0:
        import random
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1 - eval_ratio))
        train_samples = all_samples[:split_idx]
        eval_samples = all_samples[split_idx:]
        
        # Save train
        with open(train_output_path, "w", encoding="utf-8") as f:
            for item in train_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved {len(train_samples)} train samples to {train_output_path}")
        
        # Save eval
        eval_output_path = output_path / f"{dataset_name}_test.jsonl"
        with open(eval_output_path, "w", encoding="utf-8") as f:
            for item in eval_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved {len(eval_samples)} eval samples to {eval_output_path}")
    else:
        # Save all as train
        with open(train_output_path, "w", encoding="utf-8") as f:
            for item in all_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved {len(all_samples)} samples to {train_output_path}")


def concat_all_datasets(output_path: Path, split_eval: bool):
    """Concatenate all individual dataset files into combined train/test files."""
    
    train_files = list(output_path.glob("*_train.jsonl"))
    test_files = list(output_path.glob("*_test.jsonl"))
    
    # Concat train
    combined_train_path = output_path / "train.jsonl"
    with open(combined_train_path, "w", encoding="utf-8") as outf:
        for fpath in train_files:
            with open(fpath, "r", encoding="utf-8") as inf:
                for line in inf:
                    outf.write(line)
    
    train_count = sum(1 for _ in open(combined_train_path, "r", encoding="utf-8"))
    print(f"[INFO] Combined train: {train_count} samples -> {combined_train_path}")
    
    # Concat test if exists
    if split_eval and test_files:
        combined_test_path = output_path / "eval.jsonl"
        with open(combined_test_path, "w", encoding="utf-8") as outf:
            for fpath in test_files:
                with open(fpath, "r", encoding="utf-8") as inf:
                    for line in inf:
                        outf.write(line)
        
        test_count = sum(1 for _ in open(combined_test_path, "r", encoding="utf-8"))
        print(f"[INFO] Combined eval: {test_count} samples -> {combined_test_path}")


def main():
    args = parse_args()
    
    # Set output path
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        output_path = root_path / "cache" / "dataset"
    else:
        output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per dataset
    if args.sample_size is None:
        samples_per_dataset = 10000
    else:
        samples_per_dataset = args.sample_size // len(DATASETS)
    
    print(f"[INFO] Total sample size: {args.sample_size or samples_per_dataset * len(DATASETS)}")
    print(f"[INFO] Samples per dataset: {samples_per_dataset}")
    print(f"[INFO] Datasets: {DATASETS}")
    print(f"[INFO] Output path: {output_path}")
    
    # Process each dataset
    for ds_name in DATASETS:
        try:
            process_and_save_ds(output_path, ds_name, samples_per_dataset, args.split_eval, args.eval_ratio)
        except Exception as e:
            print(f"[ERROR] Failed to process {ds_name}: {e}")
            continue
    
    # Concatenate all datasets
    print("\n[INFO] Concatenating all datasets...")
    concat_all_datasets(output_path, args.split_eval)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Individual files: {list(output_path.glob('*_train.jsonl'))}")
    print(f"Combined train: {output_path / 'train.jsonl'}")
    if args.split_eval:
        print(f"Combined eval: {output_path / 'eval.jsonl'}")


if __name__ == "__main__":
    main()