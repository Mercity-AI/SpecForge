#!/bin/bash
# Example: Train EAGLE3 with detailed prediction logging
# This script demonstrates the new logging features for observing drafter improvement

# Set your paths
TARGET_MODEL="Qwen/Qwen2.5-3B-Instruct"
TRAIN_DATA="./cache/dataset/ultrachat_train.jsonl"
OUTPUT_DIR="./outputs/eagle3_detailed_logging"

# IMPORTANT: Chat template must match your model!
# Qwen models use "chatml" template (<|im_start|>, <|im_end|>)
# Llama models use "llama3" template (<|start_header_id|>, <|eot_id|>)
# If you see "WARNING: Empty loss_mask", your chat template is wrong!

# Run training with detailed logging enabled
torchrun --nproc_per_node=1 scripts/train_eagle3.py \
  --target-model-path $TARGET_MODEL \
  --train-data-path $TRAIN_DATA \
  --output-dir $OUTPUT_DIR \
  --batch-size 1 \
  --max-length 1024 \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --ttt-length 7 \
  --chat-template chatml \
  --log-interval 50 \
  --num-samples-to-log 2 \
  --save-interval 1000 \
  --eval-interval 500 \
  --verbose

# Note: Removed --log-samples-interval so it logs at end of each epoch
# Note: --num-samples-to-log set to 2 for concise output
# Note: Removed WandB (--report-to, --wandb-project, --wandb-name)

# Key parameters for detailed logging:
# --num-samples-to-log 2        : Log 2 samples at end of each epoch
# --log-samples-interval N      : Optional - log every N steps (omit for end-of-epoch only)
# --verbose                     : Enable verbose debug output
# --chat-template chatml        : IMPORTANT - must match your model!

# What gets logged (printed to console):
# 1. Model internals (tensor shapes, loss mask sum)
# 2. Token-by-token predictions vs targets for each draft position
# 3. Confidence scores (how confident the model is in predictions)
# 4. Per-position accuracy and confidence
# 5. Overall confidence gap (are we learning?)
# 6. Warnings for confident-but-wrong predictions

# The output shows:
# - Context text being completed
# - What the drafter predicts at each position (0 to ttt_length-1)
# - What the target model actually outputs
# - Confidence scores for both
# - Whether predictions match (✓/✗)

