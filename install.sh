apt update
apt install libnuma1 numactl

pip install uv

git clone https://github.com/Mercity-AI/SpecForge.git
cd SpecForge
git checkout pranav/fixes

# create a new virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install specforge
uv pip install -v . --prerelease=allow

## datset scripts
python scripts/prepare_data.py --sample-size 1500 --split-eval --dataset ultrachat
python scripts/collect_data.py --sample-size 150000 --split-eval --eval-ratio 0.05

torchrun --nproc_per_node=2 scripts/train_eagle3.py   --target-model-path Qwen/Qwen2.5-7B-Instruct   --train-data-path ./cache/dataset/train.jsonl   --eval-data-path ./cache/dataset/eval.jsonl   --output-dir ./outputs/eagle3-mla   --attention-backend flex_attention_mla   --batch-size 8   --max-length 1024   --num-epochs 3   --learning-rate 1e-4   --ttt-length 7   --save-interval 10000   --eval-interval 1000   --log-interval 50   --chat-template qwen   --report-to wandb

# ==============================================================================
# EVALUATION SCRIPTS
# ==============================================================================
# Replace CHECKPOINT_PATH with your actual checkpoint (e.g., ./outputs/eagle3-mla/epoch_0_step_10000)

# 1. Quick evaluation on test prompts (simple accuracy check)
# python scripts/eval_eagle.py \
#     --draft-checkpoint CHECKPOINT_PATH \
#     --target-model-path Qwen/Qwen2.5-7B-Instruct \
#     --chat-template qwen \
#     --num-samples 10 \
#     --attention-backend flex_attention_mla \
#     --ttt-length 7 \
#     --max-length 1024 \
#     --verbose

# 2. Comprehensive benchmark (baseline vs speculative decoding with timing)
# python scripts/eval_eagle3_standalone.py \
#     --draft-model-path CHECKPOINT_PATH \
#     --target-model-path Qwen/Qwen2.5-7B-Instruct \
#     --chat-template qwen \
#     --num-samples 20 \
#     --max-new-tokens 128 \
#     --max-prompt-length 512 \
#     --num-draft-tokens 4 \
#     --temperature 0.0

# 3. Draft quality evaluation on eval dataset (position-wise accuracy)
# python scripts/eval_draft_quality.py \
#     --draft-model-path CHECKPOINT_PATH \
#     --target-model-path Qwen/Qwen2.5-7B-Instruct \
#     --eval-data-path ./cache/dataset/eval.jsonl \
#     --chat-template qwen \
#     --max-length 1024 \
#     --num-samples 1000 \
#     --batch-size 1
