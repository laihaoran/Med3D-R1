# Stage 1: Supervised Fine-Tuning (SFT)

This directory contains the SFT training code for Med3D-R1. See the [top-level README](../README.md) for the full pipeline overview.

## Contents

- `LaMed/src/`: training, dataset, model, projector, and utility code.
- `LaMed/script/train.sh`: release-friendly training entrypoint.
- `Data/`: dataset directory skeleton only. Downloaded data should be placed here.
- `temp_model/PretrainWeight/`: placeholder for downloaded pretrained vision weights.
- `environment.yaml`: reference conda environment.

## Prepare Files

GitHub does not include datasets, model checkpoints, or template tensors. Download them from the external links provided by the project, then place them at the expected paths below or override the paths with environment variables.

Place CT-RATE volumes under:

```text
Data/ct-rate/train_fixed_256_128_high/
```

Place annotation json files under:

```text
Data/ct-rate/
```

Optionally place the vision encoder checkpoint at:

```text
temp_model/PretrainWeight/BrgSA_vision_encoder.pth
```

Place the mean prompt template used by the `convreshape` projector at:

```text
mean_prompt_template_qwen2.5.pt
```

These files are intentionally excluded from GitHub by `.gitignore`. If you store them elsewhere, set `DATA_ROOT`, `VISION_WEIGHT`, and `MEAN_PROMPT_TEMPLATE` when launching training.

## Train

Update paths through environment variables, then run:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
VISION_WEIGHT=./temp_model/PretrainWeight/BrgSA_vision_encoder.pth \
MEAN_PROMPT_TEMPLATE=./mean_prompt_template_qwen2.5.pt \
DATA_ROOT=./Data/ct-rate/train_fixed_256_128_high \
bash LaMed/script/train.sh
```

For multi-GPU training:

```bash
GPUS=0,1,2,3 NPROC_PER_NODE=4 bash LaMed/script/train.sh
```

To enable DeepSpeed:

```bash
DEEPSPEED_CONFIG=./LaMed/src/config/stage1.json bash LaMed/script/train.sh
```
