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
  --log-samples-interval 100 \
  --num-samples-to-log 5 \
  --save-interval 1000 \
  --eval-interval 500 \
  --report-to wandb \
  --wandb-project eagle3-training \
  --wandb-name detailed-logging-test \
  --verbose

# Key parameters for detailed logging:
# --log-samples-interval 200  : Log detailed predictions every 200 steps
# --num-samples-to-log 5      : Log 5 samples each time
# --verbose                   : Enable verbose debug output
# --report-to wandb           : Use Weights & Biases for logging

# What gets logged:
# 1. Prediction tables showing drafter vs target for each draft position
# 2. Confidence scores (how confident the model is in its predictions)
# 3. Hidden state statistics
# 4. Token-level comparisons with decoded text
# 5. Improvement metrics (confidence gap over time)
# 6. Per-position accuracy trends

# In WandB, you'll see:
# - predictions/samples_table: Table with all predictions
# - predictions/detailed_text: Formatted text log
# - predictions/pos_X_pred_confidence: Confidence at each position
# - predictions/pos_X_target_confidence: Target confidence at each position
# - predictions/overall_pred_confidence: Overall prediction confidence
# - predictions/overall_target_confidence: Overall target confidence
# - predictions/confidence_gap: Gap between target and prediction confidence
# - hidden_states/mean, std, min, max: Hidden state statistics

