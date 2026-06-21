# Med3D-R1: Mitigating Narrative Bias and Enhancing Reasoning Consistency in 3D Medical Vision-Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/TODO)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

This repository contains the official training code for **Med3D-R1**, a two-stage training framework for 3D medical image understanding:

- **Stage 1 – Supervised Fine-Tuning (SFT):** Trains a 3D multimodal large language model on CT-RATE covering captioning, VQA, detection, and segmentation tasks.
- **Stage 2 – Reinforcement Learning (RL):** Further optimizes the SFT model on multiple-choice VQA using Group Relative Policy Optimization (GRPO) with accuracy, format, and reward-model signals.

## Repository Structure

```
Med3D-R1/
├── stage1_sft/           # Stage 1: Supervised Fine-Tuning
│   ├── LaMed/
│   │   ├── script/train.sh       # SFT training entry point
│   │   └── src/
│   │       ├── config/           # DeepSpeed configs (stage1.json, stage2.json)
│   │       ├── dataset/          # Dataset loaders and prompt templates
│   │       ├── model/            # Model architecture (encoder, projector, LLM)
│   │       ├── train/            # Trainer and training loop
│   │       └── utils/            # Utility scripts
│   ├── Data/                     # Dataset directory (see Data Preparation)
│   └── temp_model/PretrainWeight/ # Pretrained vision encoder (see below)
└── stage2_rl/            # Stage 2: Reinforcement Learning (GRPO)
    ├── src/open_r1/
    │   ├── grpo_3d.py            # GRPO training script with reward functions
    │   ├── trainer/              # Custom GRPO trainer for Qwen2-VL
    │   └── prompt_templates.py   # Task-specific prompt templates
    ├── data/sample_data.json     # Sample VQA data (format reference)
    ├── data_config/rec.yaml      # Dataset config pointing to VQA JSON
    ├── local_scripts/zero3.json  # DeepSpeed ZeRO-3 config
    └── run_grpo_3d.sh            # RL training entry point
```

## Requirements

### Environment

Create a conda environment using the provided specification:

```bash
conda env create -f stage1_sft/environment.yaml
conda activate e3d
```

Key dependencies:
- Python 3.10
- PyTorch 2.6.0 + CUDA 12.6
- Transformers 4.51.3
- DeepSpeed 0.16.3
- TRL (for GRPO)
- flash-attn 2.7.4
- MONAI 1.2.0, nibabel (for 3D medical image processing)

## Data Preparation

### CT-RATE Dataset

Both stages use the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset. Download and organize as follows.

**Stage 1 SFT data** — place under `stage1_sft/Data/ct-rate/`:

```
stage1_sft/Data/ct-rate/
├── train_fixed_256_128_high/      # 3D CT volumes (.nii.gz)
├── medical_reports.json           # Radiology reports (captioning)
├── ctrate_vqa_train_open.json     # Open-ended VQA
├── ctrate_vqa_train_close.json    # Yes/No VQA
├── ctrate_vqa_train_close_first_image.json  # Single-choice VQA
├── bboxes.json                    # Detection annotations
└── patch_indices.json             # Segmentation patch indices
```

**Stage 2 RL data** — update `stage2_rl/data_config/rec.yaml` to point to your VQA JSON:

```yaml
datasets:
  - json_path: /path/to/ctrate_vqa_train_close_first_image.json
```

Each JSON entry must contain `VolumeName`, `question`, `options`, and `answer` fields (see `stage2_rl/data/sample_data.json` for the format).

**Video frames for Stage 2** — the GRPO trainer loads CT volumes as video frame sequences. Pre-extract slices from `.nii.gz` files into a parallel directory:

```
train_fixed_256_128_high/          # original volumes
train_fixed_256_128_high_video_frames/  # extracted JPG frames per volume
```

### Pretrained Weights

Download the BrgSA 3D vision encoder checkpoint and place it at:

```
stage1_sft/temp_model/PretrainWeight/BrgSA_vision_encoder.pth
```

Download the mean prompt template tensor for the `convreshape` projector and place it at:

```
stage1_sft/mean_prompt_template_qwen2.5.pt
```

Links for these files are provided on the project page.

## Stage 1: Supervised Fine-Tuning

The SFT stage trains a 3D multimodal model based on [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) with a 3D ViT encoder and a `convreshape` multimodal projector.

### Basic Training (single GPU)

```bash
cd stage1_sft

MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
VISION_WEIGHT=./temp_model/PretrainWeight/BrgSA_vision_encoder.pth \
MEAN_PROMPT_TEMPLATE=./mean_prompt_template_qwen2.5.pt \
DATA_ROOT=./Data/ct-rate/train_fixed_256_128_high \
bash LaMed/script/train.sh
```

### Multi-GPU Training

```bash
cd stage1_sft

GPUS=0,1,2,3 NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
VISION_WEIGHT=./temp_model/PretrainWeight/BrgSA_vision_encoder.pth \
MEAN_PROMPT_TEMPLATE=./mean_prompt_template_qwen2.5.pt \
DATA_ROOT=./Data/ct-rate/train_fixed_256_128_high \
bash LaMed/script/train.sh
```

### With DeepSpeed

```bash
cd stage1_sft

DEEPSPEED_CONFIG=./LaMed/src/config/stage1.json \
GPUS=0,1,2,3 NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
bash LaMed/script/train.sh
```

The trained checkpoint is saved to `stage1_sft/LaMed/output/LaMed-pretrain-release/` by default (override with `OUTPUT_DIR`).

## Stage 2: Reinforcement Learning

The RL stage uses GRPO to optimize the SFT model on multiple-choice VQA. Three reward signals are supported:

| Reward | Description |
|--------|-------------|
| `accuracy` | 1.0 if `<answer>` matches ground-truth letter; 0.5 for partial match |
| `format` | 1.0 if response has exactly one `<think>...</think><answer>...</answer>` block |
| `rm` | Discretized score from an external reward model (e.g., RM-Mistral-7B) |

The model is prompted to produce chain-of-thought reasoning inside `<think>` tags followed by a single-letter answer inside `<answer>` tags.

### Training (8 GPUs)

```bash
cd stage2_rl

# Update paths in run_grpo_3d.sh first:
#   MODEL_PATH      — path to SFT checkpoint from Stage 1
#   REWARD_MODEL_PATH — path to RM-Mistral-7B (required when using --reward_funcs rm)
#   OUTPUT_DIR      — desired output directory

bash run_grpo_3d.sh
```

### Training without Reward Model (accuracy + format only)

Edit `run_grpo_3d.sh` and change:

```bash
--reward_funcs accuracy format
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_generations` | 4 | Number of rollouts per prompt (GRPO group size) |
| `--per_device_train_batch_size` | 1 | Batch size per GPU |
| `--gradient_accumulation_steps` | 2 | Effective batch = GPUs × batch × accum |
| `--max_prompt_length` | 1024 | Max tokens for the input prompt |
| `--max_completion_length` | 512 | Max tokens for model completion |
| `--num_train_epochs` | 1 | Training epochs |

## Citation

If you find this work useful, please cite:

```bibtex
@article{lai2026med3d,
  title={Med3D-R1: Incentivizing Clinical Reasoning in 3D Medical Vision-Language Models for Abnormality Diagnosis},
  author={Lai, Haoran and Jiang, Zihang and Zhang, Kun and Yao, Qingsong and Wang, Rongsheng and He, Zhiyang and Tao, Xiaodong and Wei, Wei and Zhou, Shaohua Kevin},
  journal={arXiv preprint arXiv:2602.01200},
  year={2026}
}
```

## Acknowledgements
- Base language model: [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).
- Dataset: [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
