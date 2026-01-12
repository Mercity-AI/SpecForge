# Understanding SGLang Requirements and Testing Alternatives

## What SGLang Server Needs

### 1. Model Loading Mechanism

SGLang server loads draft models using **Transformers' `AutoModel.from_pretrained()`**. This requires:

1. **`config.json`** with:
   - `architectures`: List containing the model class name (e.g., `["EagleForCausalLM"]`)
   - `auto_map`: Dictionary telling Transformers where to find the custom model class

2. **Model class availability**:
   - Either a **local file** in the checkpoint directory (e.g., `modeling_eagle.py`)
   - Or an **installed package** that can be imported (e.g., `specforge.modeling.draft.eagle3`)

### 2. The `auto_map` Format

```json
{
  "auto_map": {
    "AutoConfig": "transformers.LlamaConfig",
    "AutoModel": "modeling_eagle.EagleForCausalLM"  // or "specforge.modeling.draft.eagle3--EagleForCausalLM"
  }
}
```

**Why it fails**: Transformers' `get_class_from_dynamic_module()` tries to:
1. First look for a local file in the checkpoint directory
2. If not found, try to import from an installed package
3. If both fail → Error

### 3. Current Issue

Your checkpoint has:
- ✅ `architectures: ["EagleForCausalLM"]`
- ✅ `auto_map` pointing to `specforge.modeling.draft.eagle3--EagleForCausalLM`
- ❌ But Transformers is looking for `transformers.py` file in checkpoint directory

**Solution**: Create local wrapper files (`modeling_eagle.py` and `transformers.py`) that import from SpecForge.

---

## Do You Need SGLang Server?

### Short Answer: **Only for the provided benchmark infrastructure**

### What the Benchmarks Do

The benchmark scripts (`bench_eagle3.py`) are **tightly coupled** to SGLang:

1. **Launch SGLang server** with your draft model
2. **Connect via HTTP** (host/port) to the server
3. **Send prompts** using SGLang's client API (`sgl.function` and `run_batch`)
4. **Collect metrics** like:
   - Latency
   - Throughput (tokens/sec)
   - Accept length (speculative decoding metric)
   - Accuracy (for some benchmarks)

### Alternatives to SGLang Benchmarking

#### Option 1: Use Training Metrics (Already Available) ✅

Your training script already tracks the most important metrics:

- **Position-wise accuracy** (`train/acc_0` through `train/acc_6`)
  - Position 0: ~60-70% = Good
  - Positions 1-3: ~40-50% = Good
  - Later positions: Decreasing is normal

- **Position-wise loss** (`train/ploss_0` through `train/ploss_6`)
  - Lower = Better
  - Should decrease during training

**Where to check**: Your wandb dashboard

**What it tells you**: 
- How well your draft model predicts tokens at each position
- This directly correlates with acceptance rate in speculative decoding

#### Option 2: Custom Evaluation Script (Using SpecForge Directly)

You can write a simple script to test your model without SGLang:

```python
from specforge import AutoEagle3DraftModel
from specforge.modeling.target import get_eagle3_target_model
from transformers import AutoTokenizer

# Load models
draft_model = AutoEagle3DraftModel.from_pretrained("./outputs/qwen-8b-eagle3/epoch_0_step_2000")
target_model = get_eagle3_target_model("Qwen/Qwen2.5-7B-Instruct", backend="hf")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Test on sample prompts
prompts = ["Hello, how are you?", "What is 2+2?"]

for prompt in prompts:
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Get target hidden states
    with torch.no_grad():
        target_output = target_model.generate_eagle3_data(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            loss_mask=torch.ones_like(inputs["input_ids"]),
        )
    
    # Run draft model
    draft_logits = draft_model(
        input_ids=target_output.input_ids,
        hidden_states=target_output.hidden_states,
        ...
    )
    
    # Compare predictions with target
    # Compute acceptance rate, etc.
```

#### Option 3: Fix SGLang Compatibility (For Full Benchmarking)

If you want to use the provided benchmarks, you need to:

1. **Add wrapper files** to each checkpoint:
   - `modeling_eagle.py`
   - `transformers.py`

2. **Update `config.json`** with correct `auto_map`

3. **Or modify `save_checkpoints()`** in `train_eagle3.py` to auto-generate these files

---

## Recommendation

1. **For immediate evaluation**: Use your **wandb training metrics** - they tell you everything about model quality
2. **For production benchmarking**: Fix SGLang compatibility (add wrapper files) to use the benchmark suite
3. **For custom testing**: Write a simple evaluation script using SpecForge directly

The training metrics (accuracy at each position) are actually the **most direct** measure of draft model quality and correlate strongly with real-world acceptance rates.

