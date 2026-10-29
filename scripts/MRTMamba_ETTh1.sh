#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

data="ETTh1"
seq_len=512
comment="main"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

mkdir -p "./logs/${data}"

exec > >(tee -a "$log_file") 2>&1
export PYTHONUNBUFFERED=1

model_name=MRT-Mamba
train_epochs=100
learning_rate=0.01
weight_decay=0.01

master_port=29500
num_process=1
batch_size=24

small_patch=8
small_stride=4
large_patch=24
large_stride=4

d_model=32
d_ff=128
llm_layers=12

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj type1 \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj type1 \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.001 \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj COS \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 7 \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj type1 \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"
