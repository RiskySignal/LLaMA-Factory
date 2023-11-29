#!/usr/bin/env bash

# train the pretrain model using full-parameters
deepspeed --include localhost:4,5,6,7 --master_port=9902 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage pt \
    --model_name_or_path "/dockerdata/jiangszhang/model/opt/opt-125m" \
    --do_train \
    --dataset quizlet_mixer \
    --finetuning_type full \
    --output_dir "/dockerdata/jiangszhang/tmp/model_train/opt-125m-quizlet-mixer91-0.42b-1129" \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 \
    --cutoff_len 2048 \
    --flash_attn \
    --overwrite_output_dir

# train the sft model using lora
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path "/dockerdata/jiangszhang/model/llama-2/llama-2-7b-hf" \
    --do_train \
    --dataset zhuque \
    --template alpaca \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir "/dockerdata/jiangszhang/tmp/model_train/llama-2-7b-1031" \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --bf16 \
    --cutoff_len 2048

# export the model
python src/export_model.py \
    --model_name_or_path "/dockerdata/jiangszhang/model/llama-2/llama-2-7b-hf" \
    --template alpaca \
    --finetuning_type lora \
    --checkpoint_dir "/dockerdata/jiangszhang/tmp/model_train/llama-2-7b-1031/checkpoint-3000" \
    --export_dir "/dockerdata/jiangszhang/tmp/model_train/llama-2-7b-1031/checkpoint-3000-hf"

# start cli client for inference
python src/cli_demo.py \
    --model_name_or_path "/dockerdata/jiangszhang/tmp/model_train/llama-2-7b/checkpoint-2000-hf" \
    --template alpaca \
    --max_length 2048 \
    --do_sample false


# Run Benchmark Test
python src/run_benchmark.py \
    --model_name_or_path "/dockerdata/jiangszhang/tmp/model_train/llama-2-7b-1030/checkpoint-7000-hf" \
    --template alpaca \
    --max_length 2048 \
    --do_sample false