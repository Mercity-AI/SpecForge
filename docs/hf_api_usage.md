# HuggingFace API Usage Guide

## Authentication / Login

### Method 1: Command Line (Recommended)
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login using token (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Or login with token directly
huggingface-cli login --token YOUR_TOKEN_HERE
```

### Method 2: Python API
```python
from huggingface_hub import login

# Login interactively (will prompt for token)
login()

# Or login with token directly
login(token="YOUR_TOKEN_HERE")

# Or use environment variable
import os
os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"
```

### Method 3: Environment Variable
```bash
# Set token as environment variable
export HF_TOKEN="YOUR_TOKEN_HERE"

# Or in Python
import os
os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"
```

---

## Pulling Models (HuggingFace Hub API)

### Method 1: Using `huggingface_hub` library
```python
from huggingface_hub import snapshot_download, hf_hub_download

# Download entire repository
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_type="model",
    local_dir="./models/qwen2.5-7b-instruct",
    token=True  # Uses token from login or HF_TOKEN env var
)

# Download specific file
hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    filename="config.json",
    local_dir="./models/qwen2.5-7b-instruct",
    token=True
)

# Download with revision/branch
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="main",  # or specific commit hash
    local_dir="./models/qwen2.5-7b-instruct",
    token=True
)
```

### Method 2: Using `transformers` library (auto-downloads)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Automatically downloads and caches model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=True,  # Uses token from login
    cache_dir="./cache/models"  # Optional: specify cache directory
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=True
)
```

### Method 3: Using `HfApi` class
```python
from huggingface_hub import HfApi

api = HfApi(token=True)  # Uses token from login or HF_TOKEN

# Get repository info
repo_info = api.repo_info(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_type="model"
)

# List files in repository
files = api.list_repo_files(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_type="model"
)

# Download specific file
api.hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    filename="config.json",
    local_dir="./models/qwen2.5-7b-instruct"
)
```

---

## Pulling Datasets (datasets library)

### Method 1: Using `datasets` library
```python
from datasets import load_dataset

# Load public dataset (no auth needed)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Load private/gated dataset (requires login)
dataset = load_dataset(
    "private-org/private-dataset",
    split="train",
    token=True  # Uses token from login or HF_TOKEN env var
)

# Load with streaming (for large datasets)
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    streaming=True,
    token=True
)

# Load from specific revision
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    revision="main",  # or commit hash
    token=True
)

# Load and save locally
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    token=True
)
dataset.save_to_disk("./cache/datasets/ultrachat")

# Load from local disk later
dataset = load_dataset_from_disk("./cache/datasets/ultrachat")
```

### Method 2: Using `huggingface_hub` for dataset files
```python
from huggingface_hub import snapshot_download

# Download entire dataset repository
snapshot_download(
    repo_id="HuggingFaceH4/ultrachat_200k",
    repo_type="dataset",
    local_dir="./cache/datasets/ultrachat",
    token=True
)
```

### Method 3: Load from JSON/JSONL files
```python
from datasets import load_dataset

# Load from JSON file in repository
dataset = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/org/dataset/resolve/main/data.jsonl",
    token=True
)

# Or from local file
dataset = load_dataset("json", data_files="./data.jsonl")
```

---

## Complete Example: Pull Model and Dataset

```python
from huggingface_hub import login, snapshot_download
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Login
login(token="YOUR_TOKEN_HERE")  # Or use huggingface-cli login

# 2. Download model
print("Downloading model...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_type="model",
    local_dir="./models/qwen2.5-7b-instruct",
    token=True
)

# 3. Load model using transformers
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-7b-instruct",  # or use repo_id directly
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./models/qwen2.5-7b-instruct")

# 4. Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    streaming=True,
    token=True
)

# Process dataset
for sample in dataset:
    print(sample)
    break  # Just show first sample
```

---

## Common Use Cases

### Check if logged in
```python
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception:
    print("Not logged in")
```

### Download with progress bar
```python
from huggingface_hub import snapshot_download
from tqdm import tqdm

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/qwen2.5-7b-instruct",
    token=True,
    tqdm_class=tqdm  # Shows progress bar
)
```

### Resume interrupted download
```python
from huggingface_hub import snapshot_download

# Automatically resumes if interrupted
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/qwen2.5-7b-instruct",
    token=True,
    resume_download=True  # Default is True
)
```

### Download specific files only
```python
from huggingface_hub import hf_hub_download

# Download only config and tokenizer files
files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
for filename in files:
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        filename=filename,
        local_dir="./models/qwen2.5-7b-instruct",
        token=True
    )
```

---

## Troubleshooting

### Token Issues
```python
# Check if token is set
import os
print("HF_TOKEN set:", "HF_TOKEN" in os.environ)

# Or check from huggingface_hub
from huggingface_hub import HfFolder
print("Token cached:", HfFolder.get_token() is not None)
```

### Private Repository Access
```python
# For private repos, ensure you have access and are logged in
from huggingface_hub import login, snapshot_download

login(token="YOUR_TOKEN_HERE")
snapshot_download(
    repo_id="private-org/private-model",
    token=True,  # Required for private repos
    local_dir="./models/private-model"
)
```

### Gated Models/Datasets
```python
# For gated models, you need to:
# 1. Request access on HuggingFace website
# 2. Login with token
# 3. Use token=True when loading

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gated-org/gated-model",
    token=True  # Required for gated repos
)
```

