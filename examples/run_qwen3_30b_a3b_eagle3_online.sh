#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=${1:-4}
TP_SIZE=${2:-4}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
  --standalone \
  --nproc_per_node $NUM_GPUS \
  $ROOT_DIR/scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2.5-32B-Instruct \
  --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
  --eval-data-path $ROOT_DIR/cache/dataset/ultrachat_test.jsonl \
  --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
  --output-dir $ROOT_DIR/outputs/qwen3-baseline \
  --num-epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --chat-template qwen \
  --cache-dir $ROOT_DIR/cache \
  --embedding-key model.embed_tokens.weight \
  --tp-size $TP_SIZE \
  --target-model-backend sglang \
  --report-to wandb \
  --log-interval 5 \
  --ttt-length 7 \
  --save-interval 2000 \
  --eval-interval 1000 \
  --attention-backend flex_attention
