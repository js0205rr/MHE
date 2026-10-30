#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the parent of LF-AmazonTitles-131K}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the parent of jina-embeddings-v3-hf}"
: "${RUN_TAG:?Set a unique RUN_TAG so checkpoints and logs cannot collide}"
test -f "$MODEL_ROOT/jina-embeddings-v3-hf/config.json" || {
  echo "Missing local Jina checkpoint under MODEL_ROOT" >&2
  exit 1
}

cd "$(dirname "$0")/src"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec python -u main.py \
  --dataset lfamazontitles131k \
  --data_path "$DATA_PATH" \
  --bert jinaai/jina-embeddings-v3-hf \
  --bert_path "$MODEL_ROOT" \
  --lr 5e-5 \
  --epoch 15 \
  --batch 16 \
  --max_len 32 \
  --num_group 512 \
  --group_y_candidate_topk 10 \
  --hidden_dim 300 \
  --seed 6088 \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.1 \
  --bf16 \
  --valid \
  --run_tag "$RUN_TAG"
