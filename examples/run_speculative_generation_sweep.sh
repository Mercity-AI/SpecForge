#!/bin/bash
# Example: Run true speculative generation benchmark with length sweep
#
# This script demonstrates TRUE speculative decoding with EAGLE3:
# - Draft model predicts 3-4 tokens ahead
# - Target model verifies all predictions in ONE forward pass
# - Measures real acceptance rates and speedup
#
# NOT a proxy - this is the actual speculative generation algorithm!

python scripts/eval_eagle3_standalone.py \
    --draft-checkpoint ./outputs/epoch_1_step_12528 \
    --target-model-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-path ./cache/dataset/train.jsonl \
    --chat-template qwen \
    --ttt-length 7 \
    --attention-backend flex_attention_mla \
    --generation-sweep \
    --generation-lengths 50 100 200 500 \
    --num-generation-samples 20 \
    --output-dir ./benchmark_results

# What this measures:
# - Baseline: Standard autoregressive generation (1 token per target call)
# - Speculative: EAGLE3 generation (~2-3 tokens per target call)
# - Real speedup from actual generation, not simulation!
# - Acceptance rate: how many draft tokens are accepted
# - Target calls per token: efficiency metric (lower = faster)

