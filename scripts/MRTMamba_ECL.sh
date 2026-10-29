#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

data="ECL"
seq_len=512
comment="main"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

mkdir -p "./logs/${data}"
exec > >(tee -a "$log_file") 2>&1

model_name=MRT-Mamba
train_epochs=100
learning_rate=1e-4

master_port=29500
num_process=1
batch_size=384

small_patch=16
small_stride=8
large_patch=48
large_stride=8

d_model=32
d_ff=128
llm_layers=24

enc_in=321

run_ecl() {
  local pred_len="$1"

  accelerate launch --mixed_precision bf16 --num_processes "$num_process" --main_process_port "$master_port" run_main.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model "$model_name" \
    --data ECL \
    --features M \
    --freq h \
    --seq_len 512 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --train_epochs "$train_epochs" \
    --dropout 0.2 \
    --llm_layers "$llm_layers" \
    --small_patch "$small_patch" \
    --small_stride "$small_stride" \
    --large_patch "$large_patch" \
    --large_stride "$large_stride" \
    --model_comment "$comment"
}

run_ecl 96
run_ecl 192
run_ecl 336
run_ecl 720
