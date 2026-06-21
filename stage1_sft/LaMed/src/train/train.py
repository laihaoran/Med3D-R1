import os
import deepspeed
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
# from LaMed.src.dataset.multi_dataset_template import UniDatasets, CapDataset, TextDatasets, VQADataset
# from LaMed.src.dataset.multi_dataset_qwen2 import UniDatasets, CapDataset, TextDatasets, VQADataset
# from LaMed.src.dataset.multi_dataset_qwen2vl import UniDatasets, CapDataset, TextDatasets, VQADataset
# from LaMed.src.dataset.multi_dataset_gemma import UniDatasets, CapDataset, TextDatasets, VQADataset
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedDeepseekForCausalLM, LamedQwen2ForCausalLM
# LamedQwen3ForCausalLM, LamedGemma3ForCausalLM
# , LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
from transformers.modeling_utils import unwrap_model
from torch.nn.utils.rnn import pad_sequence

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "Supported values: llama, deepseek, qwen2"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # # image  # original parameters
    # image_channel: int = field(default=1)
    # image_size: tuple = field(default=(32, 256, 256))
    # patch_size: tuple = field(default=(4, 16, 16))

    # image for mae vit
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(224, 224, 112))
    patch_size: tuple = field(default=(16, 16, 8))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)
    mean_prompt_template_path: Optional[str] = field(
        default="./mean_prompt_template_qwen2.5.pt",
        metadata={"help": "Path to the externally downloaded mean prompt template tensor for the convreshape projector."},
    )

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})
    set_proj_num: int = field(default=343, metadata={"help": "Size of pooling in projector."})
    use_random: bool = field(default=False, metadata={"help": "Used random projection in projector."})
    random_num: int = field(default=1000, metadata={"help": "iteraction."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})



@dataclass
class DataArguments:
   # for CT-RATE
    data_root: str = field(default="./Data/ct-rate/train_fixed_256_128_high", metadata={"help": "Root directory for CT-RATE volume files."})

    # caption data
    cap_data_path: str = field(default="./Data/ct-rate/medical_reports.json", metadata={"help": "Path to caption data."})

    # caption data
    cap_data_2_split_path: str = field(default="./Data/ct-rate/medical_reports_complete_no_uncertain.json", metadata={"help": "Path to caption data."})

    cap_data_4_split_path: str = field(default="./Data/ct-rate/medical_reports_complete_no_uncertain.json", metadata={"help": "Path to caption data."})

    # VQA data
    vqa_data_train_path: str = field(default="./Data/ct-rate/ctrate_vqa_train_open.json", metadata={"help": "Path to training VQA data."})
    vqa_data_val_path: str = field(default="./Data/ct-rate/ctrate_vqa_valid_open.json", metadata={"help": "Path to validation VQA data."})
    vqa_data_test_path: str = field(default="./Data/ct-rate/ctrate_vqa_test_open.json", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="./Data/ct-rate/ctrate_vqa_train_close.json", metadata={"help": "Path to training VQA Yes or No data."})

    # detection data
    detection_data_path: str = field(default="./Data/ct-rate/bboxes.json", metadata={"help": "Path to detection data."})

    # single choice data
    single_choice_data_path: str = field(default="./Data/ct-rate/ctrate_vqa_train_close_first_image.json", metadata={"help": "Path to training single choice data."})

    deepseek_vqa_data_train_path: str = field(default="./Data/ct-rate/medical_reports_with_vqa.json", metadata={"help": "Path to training single choice data."})

    deepseek_vqa_data_train_shenli_path: str = field(default="./Data/shenli/mapped_vqa_to_image_shenli.json", metadata={"help": "Path to Shenli training data."})
    shenli_data_root: str = field(default="./Data/shenli", metadata={"help": "Root directory for Shenli image files."})


    # segmentation data
    seg_index_data_path: str = field(default="./Data/ct-rate/patch_indices.json", metadata={"help": "Path to segmentation index data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="./Data/data/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
    refseg_data_train_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})





@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")
    # optim: str = field(default="adamw_hf")

#    #    # DeepSpeed config. Pass --deepspeed explicitly when needed.
    deepspeed: Optional[str] = None

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"

   

    # evaluation_strategy: str = "steps"
    # eval_accumulation_steps: int = 1
    # eval_steps: float = 0.04
    # load_best_model_at_end=True,  # Load the best model at the end of training
    # metric_for_best_model='eval_loss'  # Metric to determine the best model
    # greater_is_better=False  # For loss, smaller is better


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
#     to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return

def get_mm_projector_state_maybe_zero_3(state_dict, keys_to_match):
    to_return = {k: state_dict[k] for k in state_dict if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens', 'lm_head']
        # weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.state_dict(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    # print("Save pretrained")
    # trainer.model.config.save_pretrained(output_dir)
    # trainer.model.save_pretrained(output_dir)
    # trainer.tokenizer.save_pretrained(output_dir)



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable, max_length, pad_token_id):
        self.seg_enable = seg_enable
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    # def __call__(self, batch: list) -> dict:
    #     if self.seg_enable:
    #         images, input_ids, labels, attention_mask, segs = tuple(
    #             [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

    #         images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
    #         input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
    #         labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
    #         attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

    #         for i, seg in enumerate(segs):
    #             if seg.sum() == 0:
    #                 segs[i] = torch.zeros((1, 1, 32, 256, 256))
    #             else:
    #                 segs[i] = seg.unsqueeze(0)
    #         segs = torch.cat(segs, dim=0)

    #         return_dict = dict(
    #             images=images,
    #             input_ids=input_ids,
    #             labels=labels,
    #             attention_mask=attention_mask,
    #             segs=segs,
    #         )
    #     else:
    #         images, input_ids, labels, attention_mask, token_weights= tuple(
    #             [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'token_weight')) #'token_weight'
    #         # token_weights
    #         images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
    #         input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
    #         labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
    #         attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
    #         # if token_weights is not None:
    #         token_weights = torch.cat([_.unsqueeze(0) for _ in token_weights], dim=0)

    #         return_dict = dict(
    #             images=images,
    #             input_ids=input_ids,
    #             labels=labels,
    #             attention_mask=attention_mask,
    #             token_weights=token_weights,
    #         )

    #     return return_dict
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg')
            )
        else:
            images, input_ids, labels, attention_mask, token_weights = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'token_weight')
            )

        # 计算 batch 中的最大长度（但不超过 max_length）
        batch_max_len = min(max(len(x) for x in input_ids), self.max_length)

        def truncate_pad(tensors, pad_val):
            return pad_sequence(
                [x[:batch_max_len] for x in tensors], batch_first=True, padding_value=pad_val
            )

        # 动态 pad + 最大长度裁剪
        input_ids = truncate_pad(input_ids, self.pad_token_id)
        labels = truncate_pad(labels, -100)
        attention_mask = truncate_pad(attention_mask, 0)

        images = torch.stack(images)

        if self.seg_enable:
            segs_processed = []
            for seg in segs:
                segs_processed.append(seg.unsqueeze(0) if seg.sum() > 0 else torch.zeros((1, 1, 32, 256, 256)))
            segs = torch.cat(segs_processed, dim=0)

            return {
                "images": images,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "segs": segs,
            }
        else:
            token_weights = truncate_pad(token_weights, 1.0)
            return {
                "images": images,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "token_weights": token_weights,
            }


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    # , "<|vision_start|>", "<|vision_end|>"
    # Define and add special tokens
    # special_token = {"additional_special_tokens": ["<im_patch>", "<|im_start|>", "<|im_end|>"]}
    special_token = {"additional_special_tokens": ["<im_patch>"]}
    num_added_tokens = tokenizer.add_special_tokens(
        special_token
    )
    num_added_tokens += tokenizer.add_tokens("[SEG]")

    for token in special_token["additional_special_tokens"]:
        print(f"Token: {token} - Index: {tokenizer.convert_tokens_to_ids(token)}")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    print("Tokenizer.pad_token: ", tokenizer.pad_token)
    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    # print the token of end
    print("EOS token:", repr(tokenizer.decode(tokenizer.eos_token_id)))
    print("EOS token idx:" , tokenizer.eos_token_id)


    rank0_print("="*20 + " Model preparation " + "="*20)
    model_type = (model_args.model_type or "").lower()
    if model_args.vision_tower is not None:
        if model_type in ("llama", "llama2"):
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
            )
        elif model_type in ("deepseek", "deepseek-v3"):
            model = LamedDeepseekForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
            )
        elif model_type in ("qwen", "qwen2", "qwen2.5"):
            model = LamedQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
            )
        else:
            raise ValueError(f"Unknown model_type '{model_args.model_type}'. Use one of: llama, deepseek, qwen2.")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.config.seg_token_id = model_args.seg_token_id
    model.config.img_token_id = model_args.img_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for name, p in model.get_model().mm_projector.named_parameters():
                if 'mean_prompt_template' not in name:
                        p.requires_grad = True
    else:
        model.requires_grad_(False)
        for name, p in model.get_model().mm_projector.named_parameters():
                if 'mean_prompt_template' not in name:
                        p.requires_grad = True

        # 2. 打开 LLM 最后一个 block 的参数
        for name, p in model.get_model().layers[-1].named_parameters():
            p.requires_grad = True

          # 2. 打开 LLM 最后一个 block 的参数
        for name, p in model.get_model().layers[-2].named_parameters():
            p.requires_grad = True



    model_args.num_new_tokens = num_added_tokens
    print('num_new_tokens: ', model_args.num_new_tokens)
    model.initialize_vision_tokenizer(model_args, tokenizer) 

    # 3. 打开 lm_head 的参数
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False
        elif "embed_tokens" in name:
            p.requires_grad = False




    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        # lora_config = LoraConfig(
        #     r=training_args.lora_r,
        #     lora_alpha=training_args.lora_alpha,
        #     target_modules=find_all_linear_names(model),
        #     lora_dropout=training_args.lora_dropout,
        #     bias=training_args.lora_bias,
        #     task_type="CAUSAL_LM",
        # )

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "v_proj",  "o_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        #    
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        # for n, p in model.named_parameters():
        #     if any(
        #             [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens',  'seg_projector', 'seg_module']]
        #     ):
        #         p.requires_grad = True

        for name, param in model.named_parameters():
            if any(x in name for x in ['mean_prompt_template']):
                param.requires_grad = False
            elif any(x in name for x in ['mm_projector']):
                param.requires_grad = True
        model.print_trainable_parameters()

    # ckpt = torch.load("PATH", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)


    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")

    if model_args.tune_mm_mlp_adapter:
        train_dataset = TextDatasets(data_args, tokenizer, mode='train')
    else:
        train_dataset = UniDatasets(data_args, tokenizer, mode='train')

    # eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable, data_args.max_length, tokenizer.pad_token_id)
    rank0_print("="*20 + " Training " + "="*20)

    # for n, p in model.named_parameters():
    #     if any(
    #             [x in n for x in [ 'embed_tokens']]
    #     ):
    #         p.requires_grad = True



    # for n, p in model.named_parameters():
    #     if any(
    #             [x in n for x in ['lm_head']]
    #     ):
    #         p.requires_grad = False

    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                      )
    
    
                            #    eval_dataset=eval_dataset,
                            # compute_metrics=compute_metrics,
                            # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    
    # if training_args.lora_enable:
    #     if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    #         from peft.utils.other import fsdp_auto_wrap_policy

    #         fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    #         fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # for p in model.get_input_embeddings().parameters():
    #     p.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    print(f'train embed_tokens:{model.model.embed_tokens.weight.requires_grad}')
    print(f'train lm head:{model.lm_head.weight.requires_grad}')

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
    #     from torch.distributed.fsdp import (
    # FullyShardedDataParallel as FSDP,
    # MixedPrecision,
    # BackwardPrefetch,
    # ShardingStrategy,
    # FullStateDictConfig,
    # StateDictType,
    #     )
    #     from torch.distributed.fsdp.wrap import (
    #         transformer_auto_wrap_policy,
    #         enable_wrap,
    #         wrap,
    #     )

    #     unwrapped_model = unwrap_model(model)
    #     # 配置保存策略
    #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    #     # 使用 FSDP 的状态字典类型和保存策略
    #     with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, save_policy):
    #         state_dict_with_lora = unwrapped_model.state_dict()

    #     # lora_params = {k: v for k, v in state_dict_with_lora.items() if 'lora' in k}
    #     # 选择保存 LoRA 参数和其他指定的参数
    #     parameters_to_save = ['lora', 'mm_projector', 'embed_tokens', 'lm_head']
    #     selected_params = {k: v for k, v in state_dict_with_lora.items() if any(param in k for param in parameters_to_save)}

        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    


if __name__ == "__main__":
    main()
