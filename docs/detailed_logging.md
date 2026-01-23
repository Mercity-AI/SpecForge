# Detailed Training Logging

This guide explains the enhanced logging features in EAGLE3 training that help you observe and understand how the drafter model improves during training.

## Overview

The enhanced logging system provides detailed insights into:
- **Drafter predictions**: What tokens the drafter model predicts at each position
- **Target comparisons**: How the drafter's predictions compare to the target model
- **Confidence scores**: How confident the model is in its predictions
- **Training progress**: Whether the model is actually learning and improving

## New Command-Line Arguments

```bash
--log-samples-interval 200    # Log detailed predictions every N steps (default: 200)
--num-samples-to-log 5        # Number of samples to log in detail (default: 5)
--verbose                     # Enable verbose debug output
```

## What Gets Logged

### 1. Prediction Tables (WandB)

A table showing token-by-token comparisons:

| step | sample | context | draft_position | accuracy | predicted | target | match | pred_conf | target_conf |
|------|--------|---------|----------------|----------|-----------|--------|-------|-----------|-------------|
| 200  | 0      | "Hello..." | 0           | 85%      | "world"   | "world"| ✓     | 0.892     | 0.892       |
| 200  | 0      | "Hello..." | 1           | 78%      | "today"   | "there"| ✗     | 0.645     | 0.234       |

### 2. Confidence Metrics

Track how confident the model becomes over time:

- `predictions/overall_pred_confidence`: Average confidence in what the drafter predicts
- `predictions/overall_target_confidence`: Average probability the drafter assigns to correct tokens
- `predictions/confidence_gap`: Difference between target and predicted confidence
  - **Positive gap**: Model assigns higher probability to correct tokens (good!)
  - **Negative gap**: Model is more confident in wrong predictions (needs training)

### 3. Per-Position Metrics

For each draft position (0 to ttt_length-1):
- `predictions/pos_X_pred_confidence`: Confidence at position X
- `predictions/pos_X_target_confidence`: Target confidence at position X

This helps identify which positions the drafter learns fastest/slowest.

### 4. Hidden State Statistics

Monitor the internal representations:
- `hidden_states/mean`: Average activation
- `hidden_states/std`: Standard deviation of activations
- `hidden_states/min`, `hidden_states/max`: Activation range

### 5. Detailed Text Logs

Human-readable logs showing:

```
================================================================================
Step 200 - Detailed Predictions
================================================================================

Model Flow:
  1. Target model generates hidden states from input_ids
  2. Draft model receives: input_ids + hidden_states
  3. Draft model predicts next tokens at 7 positions
  4. Predictions compared against target model's outputs

[Sample 0]
Context: The quick brown fox jumps over the lazy dog...

  Position 0 (Acc: 85%) ✓
    Predicted: and (conf: 0.892)
    Target:    and (conf: 0.892)

  Position 1 (Acc: 78%) ✗
    Predicted: today (conf: 0.645)
    Target:    there (conf: 0.234)
    ⚠️  High confidence in wrong prediction!

[Confidence Summary]
  Avg confidence in predictions: 0.768
  Avg confidence in targets: 0.563
  Gap (target - pred): -0.205
  → Model needs more training
```

## How to Interpret the Logs

### Signs of Good Training

1. **Increasing accuracy over time**: Check `train/acc_X` metrics
2. **Decreasing loss over time**: Check `train/ploss_X` metrics
3. **Confidence gap approaching zero or positive**: The model assigns higher probability to correct tokens
4. **Fewer "high confidence wrong" warnings**: Model becomes better calibrated

### Red Flags

1. **Stagnant accuracy**: Model not learning
2. **Large negative confidence gap**: Model is confidently wrong
3. **All loss_mask sums are 0**: Data formatting issue (check the warning in training logs)

## Example Usage

### Minimal Example (5 samples, quick debugging)

```bash
python scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2.5-3B-Instruct \
  --train-data-path ./data/train.jsonl \
  --output-dir ./outputs/test \
  --batch-size 1 \
  --max-length 512 \
  --num-epochs 1 \
  --log-samples-interval 100 \
  --num-samples-to-log 3 \
  --report-to wandb \
  --verbose
```

### Full Training with Detailed Monitoring

```bash
python scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2.5-7B-Instruct \
  --train-data-path ./data/train.jsonl \
  --eval-data-path ./data/eval.jsonl \
  --output-dir ./outputs/eagle3_full \
  --batch-size 4 \
  --max-length 2048 \
  --num-epochs 10 \
  --log-interval 50 \
  --log-samples-interval 200 \
  --num-samples-to-log 5 \
  --eval-interval 1000 \
  --save-interval 2000 \
  --report-to wandb \
  --wandb-project eagle3-production \
  --wandb-run-name qwen-7b-run1
```

## Viewing Logs in WandB

1. Go to your WandB project
2. Select your run
3. Navigate to different sections:
   - **Charts**: View `predictions/*` metrics over time
   - **Tables**: View `predictions/samples_table` for detailed examples
   - **Text**: View `predictions/detailed_text` for formatted logs

## Best Practices

1. **Start with frequent logging** (`--log-samples-interval 100`) to catch issues early
2. **Use small sample counts** (`--num-samples-to-log 3-5`) to avoid cluttering logs
3. **Enable verbose mode** during initial debugging
4. **Compare confidence trends** across different hyperparameters
5. **Monitor all draft positions** - later positions are typically harder to learn

## Troubleshooting

### "WARNING: Empty loss_mask at step X!"

Your data is not being formatted correctly. The loss mask should indicate which tokens to learn from. Check:
- Your chat template matches your model
- You're using `--is-preformatted` if data is already formatted
- Your JSONL has proper conversation structure

### "wandb not available, skipping detailed predictions logging"

Install wandb: `pip install wandb`

### Logs show all zeros

This typically means:
1. Gradients aren't flowing (check optimizer)
2. Model is frozen (check `requires_grad`)
3. Data issue (check loss_mask)

Enable `--verbose` to see debug output for investigation.

## Performance Impact

Detailed logging has minimal impact:
- Runs only every N steps (configurable)
- Uses a fixed batch (no extra data loading)
- Evaluation mode (no gradient computation)
- Typical overhead: <5% when logging every 200 steps

For production training, you can increase `--log-samples-interval` to 500-1000.

