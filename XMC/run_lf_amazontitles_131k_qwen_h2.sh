#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec python -u main.py \
  --dataset lfamazontitles131k \
  --data_path /notebook/dataset \
  --bert Qwen/Qwen3-Embedding-0.6B \
  --bert_path /notebook/workspace/NLP-Model \
  --lr 5e-5 \
  --epoch 15 \
  --batch 16 \
  --max_len 32 \
  --num_group 512 \
  --group_y_candidate_topk 10 \
  --hidden_dim 300 \
  --seed 6088 \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --bf16 \
  --valid
