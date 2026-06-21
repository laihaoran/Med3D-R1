#!/bin/bash
# Training script for 3D Medical VQA with GRPO
# Reward functions: accuracy, format, discretized RM

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

RUN_NAME="3D-VL-GRPO-VQA"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

# Set paths -- adjust these to your environment
MODEL_PATH="/base_model"
REWARD_MODEL_PATH="/RM-Mistral-7B"
OUTPUT_DIR="./output/$RUN_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    src/open_r1/grpo_3d.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name data_config/rec.yaml \
    --reward_funcs accuracy format rm \
    --reward_model_path $REWARD_MODEL_PATH \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --dataloader_num_workers 8 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 10000 \
    --save_only_model true

# To train without the RM reward (only accuracy + format):
# --reward_funcs accuracy format
