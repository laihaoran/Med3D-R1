#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

GPUS="${GPUS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-25011}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
DATA_ROOT="${DATA_ROOT:-./Data/ct-rate/train_fixed_256_128_high}"
OUTPUT_DIR="${OUTPUT_DIR:-./LaMed/output/LaMed-pretrain-release}"
VISION_WEIGHT="${VISION_WEIGHT-./temp_model/PretrainWeight/BrgSA_vision_encoder.pth}"
MEAN_PROMPT_TEMPLATE="${MEAN_PROMPT_TEMPLATE-./mean_prompt_template_qwen2.5.pt}"

CAP_DATA_PATH="${CAP_DATA_PATH:-./Data/ct-rate/medical_reports.json}"
VQA_DATA_TRAIN_PATH="${VQA_DATA_TRAIN_PATH:-./Data/ct-rate/ctrate_vqa_train_open.json}"
VQA_YN_DATA_TRAIN_PATH="${VQA_YN_DATA_TRAIN_PATH:-./Data/ct-rate/ctrate_vqa_train_close.json}"
DETECTION_DATA_PATH="${DETECTION_DATA_PATH:-./Data/ct-rate/bboxes.json}"
SINGLE_CHOICE_DATA_PATH="${SINGLE_CHOICE_DATA_PATH:-./Data/ct-rate/ctrate_vqa_train_close_first_image.json}"
SEG_INDEX_DATA_PATH="${SEG_INDEX_DATA_PATH:-./Data/ct-rate/patch_indices.json}"

VISION_ARGS=()
if [[ -n "${VISION_WEIGHT:-}" ]]; then
    VISION_ARGS=(--pretrain_vision_model "${VISION_WEIGHT}")
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

MEAN_PROMPT_ARGS=()
if [[ -n "${MEAN_PROMPT_TEMPLATE:-}" ]]; then
    MEAN_PROMPT_ARGS=(--mean_prompt_template_path "${MEAN_PROMPT_TEMPLATE}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --vision_tower vitb3d \
    --model_max_length 2048 \
    --set_proj_num 343 \
    --use_random False \
    --random_num 2000 \
    --mm_projector_type convreshape \
    --proj_layer_type linear \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 0 \
    --report_to none \
    --data_root "${DATA_ROOT}" \
    --detection_data_path "${DETECTION_DATA_PATH}" \
    --single_choice_data_path "${SINGLE_CHOICE_DATA_PATH}" \
    --seg_index_data_path "${SEG_INDEX_DATA_PATH}" \
    --cap_data_path "${CAP_DATA_PATH}" \
    --vqa_data_train_path "${VQA_DATA_TRAIN_PATH}" \
    --vqa_yn_data_train_path "${VQA_YN_DATA_TRAIN_PATH}" \
    "${VISION_ARGS[@]}" \
    "${DEEPSPEED_ARGS[@]}" \
    "${MEAN_PROMPT_ARGS[@]}"
