
apt update
apt install libnuma1 numactl

pip install uv

git clone https://github.com/Mercity-AI/SpecForge.git
cd SpecForge

# create a new virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install specforge
uv pip install -v . --prerelease=allow

# training script
# torchrun --nproc_per_node=1 scripts/eval_eagle.py     --draft-checkpoint ./outputs/eagle3-mla/epoch_9_step_3000     --target-model-path Qwen/Qwen2.5-7B-Instruct     --chat-template qwen     --num-samples 5     --attention-backend flex_attention_mla     --ttt-length 7     --verbose

# eval script
# torchrun --nproc_per_node=1 scripts/eval_eagle.py     --draft-checkpoint ./outputs/eagle3-mla/epoch_9_step_3000     --target-model-path Qwen/Qwen2.5-7B-Instruct     --chat-template qwen     --num-samples 5     --attention-backend flex_attention_mla     --ttt-length 7     --verbose
