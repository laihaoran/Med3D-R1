# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import math
import random
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
import transformers
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig


# ---------------------------------------------------------------------------
# Fix flash attention bug in current version of transformers
# ---------------------------------------------------------------------------
def custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
        cos = cos.to(torch.float)
        sin = sin.to(torch.float)
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO 3D medical VQA training."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "rm"],
        metadata={
            "help": (
                "List of reward functions. "
                "Possible values: 'accuracy', 'format', 'rm'."
            )
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model (RM-Mistral-7B or similar)"},
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LazyVQAChoiseSupervisedDataset(Dataset):
    """
    Dataset for 3D CT-RATE multiple-choice VQA.

    Each JSON entry must have:
      - VolumeName: path to the .nii.gz file (used to derive video-frame folder)
      - question: full question string with options, e.g. "What is ... A) ... B) ..."
      - options: dict {"A": "...", "B": "...", "C": "...", "D": "..."}
      - answer: correct option letter, e.g. "A"

    Video frames should be pre-extracted to a parallel directory:
      train_fixed_256_128_high  ->  train_fixed_256_128_high_video_frames
    """

    COT_PROMPT = (
        "\nYour task:\n"
        "1. Think through the question step by step, and enclose your reasoning "
        "process in exactly one <think>...</think> tag.\n"
        "2. Then provide the correct single-letter choice (A, B, C, D, ...) inside "
        "exactly one <answer>...</answer> tag.\n"
        "3. Ensure that no extra information, explanations, or text appear outside "
        "of these two tags.\n"
        "4. There should be only one <think> tag and one <answer> tag in the entire "
        "response.\n"
        "5. If there is any extra content, or if the tags are not used correctly, "
        "the response will be considered invalid."
    )

    def __init__(self, data_path: str, script_args, mode: str = "train"):
        super().__init__()
        self.script_args = script_args
        self.mode = mode
        self.list_data_dict = []
        self._load_dataset_config(data_path)

    def _load_dataset_config(self, data_path: str):
        with open(data_path, "r") as f:
            config = yaml.safe_load(f)
        for dataset in config["datasets"]:
            self._process_single_dataset(dataset)

    def _process_single_dataset(self, dataset_config: dict):
        with open(dataset_config["json_path"]) as f:
            full_data = json.load(f)
        self.list_data_dict.extend(full_data)

    def __len__(self):
        return len(self.list_data_dict)

    def _build_conversation(self, example: dict) -> list:
        question = example["question"] + self.COT_PROMPT
        jpg_path = (
            example["VolumeName"]
            .replace("train_fixed_256_128_high", "train_fixed_256_128_high_video_frames")
            .replace(".nii.gz", "")
        )
        list_jpg_path = sorted([os.path.join(jpg_path, n) for n in os.listdir(jpg_path)])
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": list_jpg_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def __getitem__(self, idx: int) -> dict:
        example = self.list_data_dict[idx]
        return {
            "problem": example["question"],
            "solution": example["options"][example["answer"]],
            "prompt": self._build_conversation(example),
        }


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs) -> list[float]:
    """Reward 1.0 if the response has exactly one <think>...</think> and one
    <answer>...</answer> with non-empty content and no other tags."""

    def is_valid(content: str) -> bool:
        think_matches = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if len(think_matches) != 1 or len(answer_matches) != 1:
            return False
        if not think_matches[0].strip() or not answer_matches[0].strip():
            return False
        all_tags = re.findall(r"<(/?\w+)>", content)
        allowed = {"think", "/think", "answer", "/answer"}
        if any(t not in allowed for t in all_tags):
            return False
        pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
        return bool(re.fullmatch(pattern, content, re.DOTALL))

    completion_contents = [c[0]["content"] for c in completions]
    return [1.0 if is_valid(c) else 0.0 for c in completion_contents]


def accuracy_reward(completions, solution, **kwargs) -> list[float]:
    """
    Reward for multiple-choice VQA:
      - 1.0 if the answer exactly matches the correct option letter (e.g. "A")
      - 0.5 if the answer starts with the correct letter followed by ':' or ')'
      - 0.0 otherwise
    """
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    contents = [c[0]["content"] for c in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            match = re.search(answer_tag_pattern, content, re.DOTALL)
            if match:
                predicted = match.group(1).strip()
                if predicted == sol:
                    reward = 1.0
                elif ")" in predicted and predicted.split(")", 1)[0].strip() == sol:
                    reward = 0.5
                elif ":" in predicted and predicted.split(":", 1)[0].strip() == sol:
                    reward = 0.5
        except Exception:
            pass
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"--- {current_time} Accuracy reward: {reward} ---\n")
                    f.write(f"Content: {content}\nSolution: {sol}\n")
    return rewards


# Global RM model (loaded once per process)
_rm = None
_rm_tokenizer = None


def _load_rm(reward_model_path: str):
    global _rm, _rm_tokenizer
    if _rm is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        _rm = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
        ).to(device)
        _rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)


def _discretize_rm_score(score: float) -> float:
    """Quantization operator Q(·): map a continuous RM logit to the ordinal set
    {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, as described in the paper."""
    if score < 0.0:
        return 0.0
    elif score < 0.2:
        return 0.2
    elif score < 0.5:
        return 0.4
    elif score < 1.0:
        return 0.6
    elif score < 1.6:
        return 0.8
    else:
        return 1.0


def rm_reward(completions, solution, **kwargs) -> list[float]:
    """
    Discretized reward-model reward using an external RM (e.g. RM-Mistral-7B).

    The RM scores the conversation [user question, assistant answer, correct answer],
    and the raw logit is discretized into four levels: 0.0 / 0.3 / 0.6 / 1.0.

    Requires the global RM to have been loaded via _load_rm() during main().
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    prompts = kwargs.get("prompts", [])
    rewards = []

    for completion_item, sol, pt_item in zip(completions, solution, prompts):
        content = completion_item[0]["content"] if completion_item else ""
        # Extract the text part of the user prompt (strip image tokens)
        user_pt = pt_item[0]["content"][1]["text"] if pt_item else ""
        match = re.search(r"<\|im_end\|>(.*)", user_pt, re.DOTALL)
        pt_wo_img = match.group(1).strip() if match else user_pt

        conversation = [
            {"role": "user", "content": pt_wo_img},
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"The correct answer is {sol}"},
        ]
        conv_tokenized = _rm_tokenizer.apply_chat_template(
            conversation, tokenize=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            score = _rm(conv_tokenized).logits[0][0].item()

        rewards.append(_discretize_rm_score(score))
        del conv_tokenized
        torch.cuda.empty_cache()

    return rewards


# ---------------------------------------------------------------------------
# Reward registry
# ---------------------------------------------------------------------------
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "rm": rm_reward,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(script_args, training_args, model_args):
    # Load RM if needed
    if "rm" in script_args.reward_funcs:
        assert script_args.reward_model_path is not None, (
            "Please set --reward_model_path when using the 'rm' reward function."
        )
        _load_rm(script_args.reward_model_path)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    dataset = LazyVQAChoiseSupervisedDataset(script_args.dataset_name, script_args)
    print(f"Loaded {len(dataset)} training samples.")

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        model_args=model_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
