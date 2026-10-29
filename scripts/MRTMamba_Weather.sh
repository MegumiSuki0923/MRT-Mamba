#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

data="Weather"
seq_len=512
comment="main"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

mkdir -p "./logs/${data}"

model_name=MRT-Mamba
train_epochs=100
learning_rate=0.001

master_port=29500
num_process=1
batch_size=32

small_patch=12
small_stride=4
large_patch=24
large_stride=12

d_model=32
d_ff=128
llm_layers=24

enc_in=21

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in $enc_in \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in $enc_in \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in $enc_in \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in $enc_in \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"
