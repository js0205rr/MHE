#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
data_path="${DATA_PATH:-$workspace/data}"
model_root="${MODEL_ROOT:-$workspace/pretrained}"
run_dir="$workspace/runs/feature_dropout_20261108"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$run_dir"
cd "$workspace/src"

for spec in 'd1 0.3' 'd2 0.1'; do
  read -r run_id feature_dropout <<< "$spec"
  tag="qwen_feature_dropout_${run_id}_seed6088_20261108"
  output="$run_dir/$tag.out"
  if [[ -e "$output" || -e "log/log_lfamazontitles131k_Qwen_Qwen3-Embedding-0.6B_512_$tag" ]]; then
    echo "Existing output for $tag; refusing to append or overwrite" >&2
    exit 1
  fi

  printf '%s START %s feature_dropout=%s\n' "$(date -Is)" "$tag" "$feature_dropout" | tee -a "$run_dir/status.txt"
  "$python_bin" -u main.py \
    --dataset lfamazontitles131k \
    --data_path "$data_path" \
    --bert Qwen/Qwen3-Embedding-0.6B \
    --bert_path "$model_root" \
    --lr 5e-5 \
    --epoch 5 \
    --batch 16 \
    --max_len 32 \
    --pooling mean \
    --num_group 512 \
    --group_y_candidate_topk 10 \
    --hidden_dim 300 \
    --seed 6088 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --feature_dropout "$feature_dropout" \
    --bf16 \
    --valid \
    --no_save_checkpoint \
    --run_tag "$tag" > "$output" 2>&1
  printf '%s DONE %s\n' "$(date -Is)" "$tag" | tee -a "$run_dir/status.txt"
done
