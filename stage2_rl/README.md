# Stage 2: Reinforcement Learning with GRPO

This directory contains the GRPO-based RL training code for Med3D-R1. See the [top-level README](../README.md) for the full pipeline overview.

## Structure

```
stage2_rl/
├── src/open_r1/
│   ├── grpo_3d.py              # Main training script: dataset, rewards, main()
│   ├── trainer/
│   │   ├── grpo_trainer_qwen.py  # Custom GRPO trainer for Qwen2-VL
│   │   └── grpo_config.py        # GRPOConfig dataclass
│   └── prompt_templates.py       # Caption / detection / segmentation templates
├── data/sample_data.json         # Sample data (format reference only)
├── data_config/rec.yaml          # Dataset config (update json_path)
├── local_scripts/zero3.json      # DeepSpeed ZeRO-3 config
└── run_grpo_3d.sh                # Training entry point
```

## Data Format

Each entry in your VQA JSON must follow:

```json
{
  "VolumeName": "/path/to/train_fixed_256_128_high/.../scan.nii.gz",
  "question": "What finding is present? A) ... B) ... C) ... D) ...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A"
}
```

Update `data_config/rec.yaml` to point to your JSON file:

```yaml
datasets:
  - json_path: /path/to/ctrate_vqa_train_close_first_image.json
```

## Pre-extract Video Frames

The GRPO trainer loads CT volumes as sequences of JPG slices. Create a parallel frame directory for each volume:

```bash
# Example: extract axial slices from each .nii.gz into a sibling directory
# train_fixed_256_128_high/  ->  train_fixed_256_128_high_video_frames/
```

Frames are discovered at runtime from the path produced by replacing `train_fixed_256_128_high` with `train_fixed_256_128_high_video_frames` and stripping `.nii.gz`.

## Quick Start

Edit `run_grpo_3d.sh` to set `MODEL_PATH`, `REWARD_MODEL_PATH`, and `OUTPUT_DIR`, then:

```bash
cd stage2_rl
bash run_grpo_3d.sh
```

To run without the external reward model (accuracy + format rewards only):

```bash
# In run_grpo_3d.sh, change the reward_funcs line to:
#   --reward_funcs accuracy format
bash run_grpo_3d.sh
```

## Reward Functions

| Name | Signal |
|------|--------|
| `accuracy` | 1.0 / 0.5 / 0.0 based on whether `<answer>` matches ground truth |
| `format` | 1.0 if exactly one `<think>…</think><answer>…</answer>` block present |
| `rm` | Discretized score {0, 0.2, 0.4, 0.6, 0.8, 1.0} from RM-Mistral-7B |
