import random
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates,Finding_templates, Impression_template
from .term_dictionary import term_dict
from .prompt_templates import generate_mmvaq_prompt, generate_yesno_prompt, generate_open_prompt, generate_region_prompt, generate_vqa_prompt, generate_finding_prompt, Think_finding_prefix, generate_vqa_wo_tag_prompt, generate_report_prompt


def resolve_ct_rate_image_path(data_root, image_path):
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    marker = "train_fixed_256_128_high/"
    if os.path.isabs(image_path) and marker in image_path:
        return os.path.join(data_root, image_path.split(marker, 1)[1])

    if os.path.isabs(image_path):
        return image_path

    parts = image_path.split("_")
    if len(parts) >= 3 and image_path.endswith(".nii.gz"):
        base_name = f"{parts[0]}_{parts[1]}"
        second_base_name = f"{base_name}_{parts[2]}"
        return os.path.join(data_root, base_name, second_base_name, image_path)

    return os.path.join(data_root, image_path)


def resolve_shenli_image_path(args, image_path):
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(getattr(args, "shenli_data_root", "./Data/shenli"), image_path)



class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

        # train_transform = mtf.Compose(
        #     [
        #         # mtf.ToTensor(dtype=torch.float),
        #         # mtf.AddChannel(),
        #         # Randomly crop the 3D image to a fixed size
        #         mtf.Resize(spatial_size=(32, 256, 256)),
        #         # Normalize intensity of the image from -1000~1000 to -1~1
        #         mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),


        #         mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        #         mtf.RandFlip(prob=0.10, spatial_axis=0),
        #         mtf.RandFlip(prob=0.10, spatial_axis=1),
        #         mtf.RandFlip(prob=0.10, spatial_axis=2),
        #         mtf.RandScaleIntensity(factors=0.1, prob=0.5),
        #         mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

        #         mtf.ToTensor(dtype=torch.float),
        #     ]
        # )

        # val_transform = mtf.Compose(
        #         [
           
        #             # mtf.ToTensor(dtype=torch.float),
        #             # mtf.AddChannel(),
        #         #    # Randomly crop the 3D image to a fixed size
        #             mtf.Resize(spatial_size=(32, 256, 256)),
        #             # Normalize intensity of the image from -1000~1000 to -1~1
        #             mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        #             mtf.ToTensor(dtype=torch.float),

        #         ]

        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                # mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                # Normalize intensity of the image from -1000~1000 to -1~1

                # mtf.RandRotate90(prob=0.50, spatial_axes=(0, 1)),
                # mtf.RandFlip(prob=0.10, spatial_axis=0),
                # mtf.RandFlip(prob=0.10, spatial_axis=1),
                # mtf.RandFlip(prob=0.10, spatial_axis=2),
                # mtf.RandScaleIntensity(factors=0.1, prob=0.1),
                # mtf.RandShiftIntensity(offsets=0.1, prob=0.1),

                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    # mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                    mtf.CenterSpatialCrop(roi_size=(224, 224, 112)),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                    # mtf.ToTensor(dtype=torch.float),

                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                # image_abs_path = os.path.join(self.data_root, image_path)

                image_abs_path = image_path
                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = nib.load(image_abs_path).get_fdata()

                # image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # necessary for original process
                
                image = self.transform(image)

                # image = image.permute(0, 3, 2, 1)  # necessary for original process

                # text_path = data["text"]
                # text_abs_path = os.path.join(self.data_root, text_path)
                # with open(text_abs_path, 'r') as text_file:
                #     raw_text = text_file.read()
                raw_text = data["conversations"][1]['value']

                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                # question = self.image_tokens + prompt_question
                image_prompt = 'This is a CT scan of a patient as follow: '
                medium_prompt = 'According to the image, please answer the question.'


                question = 'user\n'+ image_prompt + '<|im_start|>' + self.image_tokens + '<|im_end|>'  + medium_prompt + prompt_question + '\nassistant\n'

                text_tensor = self.tokenizer(
                    question + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    # 'token_weight': None,
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class WeightCapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"
        
        # 预存特殊标记的 ID
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")


        with open(args.cap_data_2_split_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])

        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def format_tag_block_with_weight(self, tokenizer, tag, content_list):
        if not content_list:
            return [], []

        start_tag = f"<|{tag}|>"
        end_tag = f"<|/{tag}|>"

        token_ids = []
        weights = []

        # Start tag
        start_tokens = tokenizer.encode(start_tag, add_special_tokens=False)
        token_ids.extend(start_tokens)
        weights.extend([1.00] * len(start_tokens))

        for idx, item in enumerate(content_list):
            text = item["text"].strip()
            if not text:
                continue

            if idx > 0:
                # Add space between sentences
                space_tokens = tokenizer.encode(" ", add_special_tokens=False)
                token_ids.extend(space_tokens)
                weights.extend([1.0] * len(space_tokens))

            # Encode current sentence
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_ids.extend(tokens)
            token_weight = 1.10 if item.get("type", "") == "abnormal" else 1.00
            weights.extend([token_weight] * len(tokens))

        # End tag
        end_tokens = tokenizer.encode(end_tag, add_special_tokens=False)
        token_ids.extend(end_tokens)
        weights.extend([1.00] * len(end_tokens))

        return token_ids, weights


    def build_chatml_prompt_with_weights(self, tokenizer, image_tokens, prompt_question, finding_list, impression_list):
        # 构建 system 和 user
        task_lines = []
        # if finding_list:
        #     task_lines.append("Provide detailed findings enclosed in <|finding|>...<|/finding|>.")
        # if impression_list:
        #     task_lines.append("Provide concise impressions enclosed in <|impression|>...<|/impression|>.")
        # task_lines.append("Ensure your response is medically accurate, logically consistent, and well-structured.")

        # system_prompt = "You are a medical AI assistant that analyzes CT scans and generates radiology reports.\n" + "\n".join(
        #     f"{i + 1}. {line}" for i, line in enumerate(task_lines)
        # )
        system_prompt = generate_report_prompt(finding_list=finding_list, impression_list=impression_list)
        user_prompt = f"{image_tokens}{prompt_question.strip()}"

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 构建 token 索引和权重
        finding_tokens, finding_weights = self.format_tag_block_with_weight(tokenizer, "finding", finding_list)
        impression_tokens, impression_weights = self.format_tag_block_with_weight(tokenizer, "impression", impression_list)
        newline_token = tokenizer.encode("\n", add_special_tokens=False)
        im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        assistant_token_ids = (
            finding_tokens +
            newline_token +
            impression_tokens +
            im_end_token
        )

        assistant_weights = (
            finding_weights +
            [1.0] * len(newline_token) +
            impression_weights +
            [1.0] * len(im_end_token)
        )

        return conversation, assistant_token_ids, assistant_weights


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = resolve_ct_rate_image_path(self.args.data_root, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)

                finding = data["conversations"][1]['value']['finding']
                impression = data["conversations"][1]['value']['impression']
                prompt_question = random.choice(self.caption_prompts)

                
                conversation, assistant_token_ids, assistant_weights = self.build_chatml_prompt_with_weights(
                    tokenizer=self.tokenizer,
                    image_tokens=self.image_tokens,
                    prompt_question=prompt_question,
                    finding_list=finding,
                    impression_list=impression
                )
                assistant_tokens = torch.tensor(assistant_token_ids, dtype=torch.long)

                encoded = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=False
                    )
            

                # 获取 input_ids 中的 <|im_start|> 位置
                input_ids = encoded["input_ids"][0]
                assistant_start = len(input_ids)


                # 合并所有 tokens
                input_ids = torch.cat([input_ids, assistant_tokens])
                attention_mask = torch.ones_like(input_ids)

                # 构建 assistant_mask
                assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                assistant_mask[assistant_start:] = True  # 从 assistant 开始到结尾

                # 处理 labels 和 weights
                labels = input_ids.clone()
                labels[~assistant_mask] = -100

                weights = torch.ones_like(input_ids, dtype=torch.float16)
                weights[~assistant_mask] = 0.0
                

                # 在合并权重前检查长度一致性
                assert len(assistant_weights) == len(assistant_tokens), \
                    f"Weights长度 {len(assistant_weights)} 与 Tokens长度 {len(assistant_tokens)} 不匹配"


                weights[assistant_start:] = torch.tensor(assistant_weights, dtype=torch.float16)



                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': conversation[1]['content'],
                    'answer': self.tokenizer.decode(assistant_tokens, skip_special_tokens=True),
                    'question_type': "Caption",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class WeightFindingCapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"
        
        # 预存特殊标记的 ID
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")


        with open(args.cap_data_2_split_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Finding_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])

        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def format_tag_block_with_weight(self, tokenizer, tag, content_list):
        if not content_list:
            return [], []

        token_ids = []
        weights = []

        # Start tag

        for idx, item in enumerate(content_list):
            text = item["text"].strip()
            if not text:
                continue

            if idx > 0:
                # Add space between sentences
                space_tokens = tokenizer.encode(" ", add_special_tokens=False)
                token_ids.extend(space_tokens)
                weights.extend([1.0] * len(space_tokens))

            # Encode current sentence
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_ids.extend(tokens)
            token_weight = 1.10 if item.get("type", "") == "abnormal" else 1.00
            weights.extend([token_weight] * len(tokens))


        return token_ids, weights


    def build_chatml_prompt_with_weights(self, tokenizer, image_tokens, prompt_question, finding_list, impression_list):
        # 构建 system 和 user
        # task_lines = []
        # task_lines.append("Ensure your response is medically accurate, logically consistent, and well-structured.")

        # system_prompt = "You are a medical AI assistant that analyzes CT scans and generates radiology findings.\n" + "\n".join(
        #     f"{i + 1}. {line}" for i, line in enumerate(task_lines)
        # )

        system_prompt = generate_finding_prompt()

        user_prompt = f"{image_tokens}{prompt_question.strip()}"

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 构建 token 索引和权重
        finding_tokens, finding_weights = self.format_tag_block_with_weight(tokenizer, "finding", finding_list)
        im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        assistant_token_ids = (
            finding_tokens +
            im_end_token
        )

        assistant_weights = (
            finding_weights + [1.0] * len(im_end_token)
        )

        return conversation, assistant_token_ids, assistant_weights


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = resolve_ct_rate_image_path(self.args.data_root, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)

                finding = data["conversations"][1]['value']['finding']
                impression = data["conversations"][1]['value']['impression']
                prompt_question = random.choice(self.caption_prompts)

                if len(finding) == 0:
                    idx = random.randint(0, len(self.data_list) - 1)
                    continue
                conversation, assistant_token_ids, assistant_weights = self.build_chatml_prompt_with_weights(
                    tokenizer=self.tokenizer,
                    image_tokens=self.image_tokens,
                    prompt_question=prompt_question,
                    finding_list=finding,
                    impression_list=impression
                )
                assistant_tokens = torch.tensor(assistant_token_ids, dtype=torch.long)

                encoded = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=False
                    )
            

                # 获取 input_ids 中的 <|im_start|> 位置
                input_ids = encoded["input_ids"][0]
                assistant_start = len(input_ids)


                # 合并所有 tokens
                input_ids = torch.cat([input_ids, assistant_tokens])
                attention_mask = torch.ones_like(input_ids)

                # 构建 assistant_mask
                assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                assistant_mask[assistant_start:] = True  # 从 assistant 开始到结尾

                # 处理 labels 和 weights
                labels = input_ids.clone()
                labels[~assistant_mask] = -100

                weights = torch.ones_like(input_ids, dtype=torch.float16)
                weights[~assistant_mask] = 0.0
                

                # 在合并权重前检查长度一致性
                assert len(assistant_weights) == len(assistant_tokens), \
                    f"Weights长度 {len(assistant_weights)} 与 Tokens长度 {len(assistant_tokens)} 不匹配"


                weights[assistant_start:] = torch.tensor(assistant_weights, dtype=torch.float16)

                # # Truncate & pad
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # if pad_len > 0:
                #     pad_id = self.tokenizer.pad_token_id
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': conversation[1]['content'],
                    'answer': self.tokenizer.decode(assistant_tokens, skip_special_tokens=True),
                    'question_type': "Caption",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

think_rate = 0


class ReportVQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"
        
        # 预存特殊标记的 ID
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")


        with open(args.cap_data_2_split_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Impression_template

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])

        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def format_tag_block_with_weight(self, tokenizer, tag, content_list):
        if not content_list:
            return [], []

        start_tag = f"<|{tag}|>"
        end_tag = f"<|/{tag}|>"

        token_ids = []
        weights = []

        # Start tag
        start_tokens = tokenizer.encode(start_tag, add_special_tokens=False)
        token_ids.extend(start_tokens)
        weights.extend([1.00] * len(start_tokens))

        for idx, item in enumerate(content_list):
            text = item["text"].strip()
            if not text:
                continue

            if idx > 0:
                # Add space between sentences
                space_tokens = tokenizer.encode(" ", add_special_tokens=False)
                token_ids.extend(space_tokens)
                weights.extend([1.0] * len(space_tokens))

            # Encode current sentence
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_ids.extend(tokens)
            token_weight = 1.10 if item.get("type", "") == "abnormal" else 1.00
            weights.extend([token_weight] * len(tokens))

        # End tag
        end_tokens = tokenizer.encode(end_tag, add_special_tokens=False)
        token_ids.extend(end_tokens)
        weights.extend([1.00] * len(end_tokens))

        return token_ids, weights


    def build_chatml_prompt_with_weights(self, tokenizer, image_tokens, prompt_question, finding_list, impression_list):
        # 构建 system 和 user
        task_lines = []
        if finding_list:
            task_lines.append("Provide detailed findings enclosed in <|finding|>...<|/finding|>.")
        if impression_list:
            task_lines.append("Provide concise impressions enclosed in <|impression|>...<|/impression|>.")
        task_lines.append("Ensure your response is medically accurate, logically consistent, and well-structured.")

        system_prompt = "You are a medical AI assistant that analyzes CT scans and generates radiology reports.\n" + "\n".join(
            f"{i + 1}. {line}" for i, line in enumerate(task_lines)
        )
        user_prompt = f"{image_tokens}{prompt_question.strip()}"

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 构建 token 索引和权重
        finding_tokens, finding_weights = self.format_tag_block_with_weight(tokenizer, "finding", finding_list)
        impression_tokens, impression_weights = self.format_tag_block_with_weight(tokenizer, "impression", impression_list)
        newline_token = tokenizer.encode("\n", add_special_tokens=False)
        im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        assistant_token_ids = (
            finding_tokens +
            newline_token +
            impression_tokens +
            im_end_token
        )

        assistant_weights = (
            finding_weights +
            [1.0] * len(newline_token) +
            impression_weights +
            [1.0] * len(im_end_token)
        )

        return conversation, assistant_token_ids, assistant_weights

    def build_vqa_prompt_with_weights(self, tokenizer, image_tokens, prompt_question, finding_list, impression_list):
        # 构造固定问题
        user_prompt = f"{image_tokens}{prompt_question}"

    #     # system message 可固定或灵活配置
    #     system_prompt = (
    #     "You are a medical visual question answering (VQA) assistant. "
    #     "You will be given a CT image and a user question.\n\n"
    #     "Your task:\n"
    #     "1. Carefully observe the image and think step-by-step in a <think>...</think> tag.\n"
    #     "2. Then, answer the user's question in an <answer>...</answer> tag.\n\n"
    #     "Base your reasoning only on what is visible in the image. "
    #     "Ensure the logic is clear and medically sound."
    # )

        # 0.7概率为True，0.3概率为False
        use_think = random.random() < think_rate
        system_prompt = generate_vqa_prompt(use_think=use_think)
        

        # 构造 <think> 内容
        finding_text = " ".join([item["text"].strip() for item in finding_list if item["text"].strip()])
        impression_text = " ".join([item["text"].strip() for item in impression_list if item["text"].strip()])
        prefix = random.choice(Think_finding_prefix)
        if use_think:
            think_answer_text = f"<think>{prefix} {finding_text}</think>\n{impression_text}"
        else:
            think_answer_text = f"{impression_text}"

        # # 编码输出部分（只需 encode 一次）
        # assistant_token_ids = tokenizer.encode(think_answer_text, add_special_tokens=False)
        # assistant_weights = [1.0] * len(assistant_token_ids)

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": think_answer_text}
        ]

        return conversation, use_think


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = resolve_ct_rate_image_path(self.args.data_root, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)

                finding = data["conversations"][1]['value']['finding']
                impression = data["conversations"][1]['value']['impression']
                prompt_question = random.choice(self.caption_prompts)

                # 如果 finding 或 impression 缺失，则换一个 idx 重试
                if len(impression) == 0 or len(finding) == 0:
                    idx = random.randint(0, len(self.data_list) - 1)
                    continue

                conversation, use_think = self.build_vqa_prompt_with_weights(
                    tokenizer=self.tokenizer,
                    image_tokens=self.image_tokens,
                    prompt_question=prompt_question,
                    finding_list=finding,
                    impression_list=impression
                )

                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': conversation[1]['content'],
                    'answer': conversation[2]['content'],
                    'question_type': "Caption",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                # print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MMVQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"


        with open(args.deepseek_vqa_data_train_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = os.path.join(self.args.data_root, base_name, second_base_name, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)
                
                value = random.randint(0, 1)

                question = data['vqa']['MMVQA'][value]['question']
                options = data['vqa']['MMVQA'][value]['options']
                reasoning = data['vqa']['MMVQA'][value]['reasoning']
                answer = data['vqa']['MMVQA'][value]['answer']
                # 构造对话内容
    #             system_prompt = (
    #     "You are a medical image analysis expert. Your task is to analyze a chest CT scan and answer a multiple-choice "
    #     "question based solely on the visible information in the image.\n\n"
    #     "You must:\n"
    #     "1. Carefully examine the image for relevant visual features.\n"
    #     "2. Consider all the options and reason which one best matches the image findings.\n"
    #     "3. Explain your reasoning step-by-step in a <think>...</think> tag.\n"
    #     "4. Select the best option and output the final answer in an <answer>...</answer> tag using the option letter (e.g., A, B, C, or D).\n\n"
    #     "Base everything strictly on what is visible."
    # )

                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_mmvaq_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{self.format_prompt_with_options(question, options)}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                # -- 构造 full input_ids（用于训练） --
                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]


                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class YESNODataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = os.path.join(self.args.data_root, base_name, second_base_name, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)
                
                value = random.randint(0, 1)

                question = data['vqa']['YESNO'][value]['question']
                reasoning = data['vqa']['YESNO'][value]['reasoning']
                answer = data['vqa']['YESNO'][value]['answer']
                # 构造对话内容
    #             system_prompt =  (
    #     "You are a medical image analysis expert. Your task is to analyze a chest CT scan and answer a Yes/No question "
    #     "based solely on the visible information in the image.\n\n"
    #     "You must:\n"
    #     "1. Examine the image to find visual evidence related to the question.\n"
    #     "2. Provide your reasoning step-by-step inside a <think>...</think> tag.\n"
    #     "3. Output your answer in an <answer>Yes</answer> or <answer>No</answer> format.\n\n"
    #     "Do not guess. If the image does not provide enough evidence, explain this clearly."
    # )

                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_yesno_prompt(use_think=use_think)



                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class REGIONDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = os.path.join(self.args.data_root, base_name, second_base_name, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)
                
                # value = random.randint(0, 1)

                question = data['vqa']['REGION']['question']
                reasoning = data['vqa']['REGION']['reasoning']
                answer = data['vqa']['REGION']['answer']
                # 构造对话内容
    #             system_prompt =  (
    #     "You are a medical image analysis expert. Your task is to describe the visual abnormality observed in a specific "
    #     "region of the chest CT scan.\n\n"
    #     "You must:\n"
    #     "1. Focus only on the specified anatomical region in the question.\n"
    #     "2. Identify any visible abnormality in that area and explain your reasoning in a <think>...</think> tag.\n"
    #     "3. Provide a precise and concise description of the abnormality in an <answer>...</answer> tag.\n\n"
    #     "Avoid general observations. Only include findings from the region mentioned in the question."
    # )

                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_region_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )              # 如果conversation 完整的话，此时enable_thinking会根据后面是否包含think，来决定是否加入think tag，与enable_thinking的设计没有关系
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                    
                    # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

       
                prefix_len = encoded_prefix["input_ids"].shape[-1]
                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class OPENDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_path, 'r') as file:
            self.data_list = json.load(file)

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                base_name = "_".join(data["image"].split('_')[:2])
                second_base_name = base_name + '_' + data["image"].split('_')[2]
                image_path = os.path.join(self.args.data_root, base_name, second_base_name, data["image"])
                image = nib.load(image_path).get_fdata()
                image = self.transform(image)
                
                # value = random.randint(0, 1)

                question = data['vqa']['OPEN']['question']
                reasoning = data['vqa']['OPEN']['reasoning']
                answer = data['vqa']['OPEN']['answer']
                # 构造对话内容
    #             system_prompt =  (
    #     "You are a medical image analysis expert. Your task is to analyze the CT image and provide the most likely diagnosis "
    #     "based solely on visible findings.\n\n"
    #     "You must:\n"
    #     "1. Identify and connect all relevant visual features in the image.\n"
    #     "2. Explain the reasoning leading to your diagnosis inside a <think>...</think> tag.\n"
    #     "3. Provide the final diagnosis inside an <answer>...</answer> tag.\n\n"
    #     "Your answer should reflect only what can be concluded from the image."
    # )

                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_open_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class MMVQASLDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"


        with open(args.deepseek_vqa_data_train_shenli_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = resolve_shenli_image_path(self.args, data["image"])
                image = nib.load(image_path).get_fdata().astype('float32') / 10.0
                image = self.transform(image)
                
                value = random.randint(0, 1)

                question = data['vqa']['MMVQA'][value]['question']
                options = data['vqa']['MMVQA'][value]['options']
                reasoning = data['vqa']['MMVQA'][value]['reasoning']
                answer = data['vqa']['MMVQA'][value]['answer']
                # 构造对话内容
                                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_mmvaq_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{self.format_prompt_with_options(question, options)}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                 # -- 构造 full input_ids（用于训练） --
                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                    

                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]


                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class YESNOSLDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_shenli_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = resolve_shenli_image_path(self.args, data["image"])
                image = nib.load(image_path).get_fdata().astype('float32') / 10.0
                image = self.transform(image)
                
                value = random.randint(0, 1)

                question = data['vqa']['YESNO'][value]['question']
                reasoning = data['vqa']['YESNO'][value]['reasoning']
                answer = data['vqa']['YESNO'][value]['answer']

                 # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_yesno_prompt(use_think=use_think)



                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                prefix_len = encoded_prefix["input_ids"].shape[-1]
                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class REGIONSLDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_shenli_path, 'r') as file:
            self.data_list = json.load(file)

        self.caption_prompts = Caption_templates

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = resolve_shenli_image_path(self.args, data["image"])
                image = nib.load(image_path).get_fdata().astype('float32') / 10.0
                image = self.transform(image)
                
                # value = random.randint(0, 1)

                question = data['vqa']['REGION']['question']
                reasoning = data['vqa']['REGION']['reasoning']
                answer = data['vqa']['REGION']['answer']
                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_region_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )              # 如果conversation 完整的话，此时enable_thinking会根据后面是否包含think，来决定是否加入think tag，与enable_thinking的设计没有关系
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                    
                    # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )

                
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

       
                prefix_len = encoded_prefix["input_ids"].shape[-1]
                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class OPENSLDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        with open(args.deepseek_vqa_data_train_shenli_path, 'r') as file:
            self.data_list = json.load(file)

        transform = mtf.Compose([
            mtf.ToTensor(dtype=torch.float),
            mtf.AddChannel(),
            mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
            mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
            mtf.NormalizeIntensity(subtrahend=0.4978, divisor=0.2449)
        ])
        set_track_meta(False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def format_prompt_with_options(self, question: str, options: dict) -> str:
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        return f"{question.strip()}\n{options_str}"


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = resolve_shenli_image_path(self.args, data["image"])
                image = nib.load(image_path).get_fdata().astype('float32') / 10.0
                image = self.transform(image)
                
                # value = random.randint(0, 1)

                question = data['vqa']['OPEN']['question']
                reasoning = data['vqa']['OPEN']['reasoning']
                answer = data['vqa']['OPEN']['answer']
                # 0.7概率为True，0.3概率为False
                use_think = random.random() < think_rate

                system_prompt = generate_open_prompt(use_think=use_think)


                user_prompt = f"{self.image_tokens}{question}"

                if use_think:
                    assistant_response = f"{reasoning}\n{answer}"
                else:
                    assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                if use_think:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=True
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                    )
                else:
                    # -- 构造 full input_ids（用于训练） --
                    encoded_full = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                             # -- 构造前缀（用于截断位置） --
                    encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                prefix_len = encoded_prefix["input_ids"].shape[-1]


                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "MMVQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"

        if mode == "train":
            with open(args.vqa_data_train_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            with open(args.vqa_data_val_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            with open(args.vqa_data_test_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        # train_transform = mtf.Compose(
        #     [
        #         # mtf.ToTensor(dtype=torch.float),
        #         # mtf.AddChannel(),
        #         # Randomly crop the 3D image to a fixed size
        #         mtf.Resize(spatial_size=(32, 256, 256)),
        #         # Normalize intensity of the image from -1000~1000 to -1~1
        #         mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),


        #         mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        #         mtf.RandFlip(prob=0.10, spatial_axis=0),
        #         mtf.RandFlip(prob=0.10, spatial_axis=1),
        #         mtf.RandFlip(prob=0.10, spatial_axis=2),
        #         mtf.RandScaleIntensity(factors=0.1, prob=0.5),
        #         mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

        #         # mtf.ToTensor(dtype=torch.float),
        #     ]
        # )

        # val_transform = mtf.Compose(
        #         [
           
        #             # mtf.ToTensor(dtype=torch.float),
        #             # mtf.AddChannel(),
        #         #    # Randomly crop the 3D image to a fixed size
        #             mtf.Resize(spatial_size=(32, 256, 256)),
        #             # Normalize intensity of the image from -1000~1000 to -1~1
        #             mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        #             mtf.ToTensor(dtype=torch.float),

        #         ]
        
        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                # Normalize intensity of the image from -1000~1000 to -1~1

                # mtf.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                # mtf.RandFlip(prob=0.10, spatial_axis=0),
                # mtf.RandFlip(prob=0.10, spatial_axis=1),
                # mtf.RandFlip(prob=0.10, spatial_axis=2),
                # mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                # mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                    # mtf.ToTensor(dtype=torch.float),

                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                # data = self.data_list.iloc[idx]
                data = self.data_list[idx]
                # image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
                image_abs_path = resolve_ct_rate_image_path(self.args.data_root, data["Image Path"])

                image = nib.load(image_abs_path).get_fdata()

                # image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # necessary for original process
                
                image = self.transform(image)

                # image = image.permute(0, 3, 2, 1)  # necessary for original process

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    question = question + ' ' + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])


                # question = self.image_tokens + ' ' + question

                system_prompt = generate_vqa_wo_tag_prompt()
                
                user_prompt = f"{self.image_tokens}{question}"
                assistant_response = f"{answer}"
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                # -- 构造 full input_ids（用于训练） --
                encoded_full = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=False,
                    enable_thinking=False
                )
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                # -- 构造前缀（用于截断位置） --
                encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "VQA",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret



class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<|vision_start|>" + ("<im_patch>" * args.proj_out_num) + "<|vision_end|>"
        if mode == "train":
            with open(args.vqa_yn_data_train_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            with open(args.vqa_yn_data_val_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            with open(args.vqa_yn_data_test_path, 'r') as file:
                self.data_list = json.load(file)
            # self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        # train_transform = mtf.Compose(
        #     [
        #         # mtf.ToTensor(dtype=torch.float),
        #         # mtf.AddChannel(),
        #         # Randomly crop the 3D image to a fixed size
        #         mtf.Resize(spatial_size=(32, 256, 256)),
        #         # Normalize intensity of the image from -1000~1000 to -1~1
        #         mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),


        #         mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        #         mtf.RandFlip(prob=0.10, spatial_axis=0),
        #         mtf.RandFlip(prob=0.10, spatial_axis=1),
        #         mtf.RandFlip(prob=0.10, spatial_axis=2),
        #         mtf.RandScaleIntensity(factors=0.1, prob=0.5),
        #         mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

        #         # mtf.ToTensor(dtype=torch.float),
        #     ]
        # )

        # val_transform = mtf.Compose(
        #         [
           
        #             # mtf.ToTensor(dtype=torch.float),
        #             # mtf.AddChannel(),
        #         #    # Randomly crop the 3D image to a fixed size
        #             mtf.Resize(spatial_size=(32, 256, 256)),
        #             # Normalize intensity of the image from -1000~1000 to -1~1
        #             mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        #             mtf.ToTensor(dtype=torch.float),

        #         ]
            
        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                # Normalize intensity of the image from -1000~1000 to -1~1

                # mtf.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                # mtf.RandFlip(prob=0.10, spatial_axis=0),
                # mtf.RandFlip(prob=0.10, spatial_axis=1),
                # mtf.RandFlip(prob=0.10, spatial_axis=2),
                # mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                # mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                    # mtf.ToTensor(dtype=torch.float),

                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                # data = self.data_list.iloc[idx]
                data = self.data_list[idx]
                # image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
                image_abs_path = data["Image Path"]

                image = nib.load(image_abs_path).get_fdata()

                # image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # necessary for original process
                
                image = self.transform(image)

                # image = image.permute(0, 3, 2, 1)  # necessary for original process

                question = data["Question"]
                answer = str(data["Answer"])

                system_prompt = generate_vqa_wo_tag_prompt()

                user_prompt = f"{self.image_tokens}{question}"
                assistant_response = f"{answer}"
    
                # --- ChatML 对话构造 ---
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]

                # -- 构造 full input_ids（用于训练） --
                encoded_full = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=False,
                    enable_thinking=False
                )
                input_ids = encoded_full["input_ids"][0]
                attention_mask = encoded_full["attention_mask"][0]

                # -- 构造前缀（用于截断位置） --
                encoded_prefix = self.tokenizer.apply_chat_template(
                    conversation[:-1],  # 不含 assistant
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prefix_len = encoded_prefix["input_ids"].shape[-1]

                # -- 构造 labels 和 token_weight --
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100
                weights = torch.ones_like(input_ids, dtype=torch.float16)

                # # -- Padding & Truncation --
                # max_len = self.args.max_length
                # pad_len = max_len - len(input_ids)
                # pad_id = self.tokenizer.pad_token_id

                # if pad_len > 0:
                #     input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                #     attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                #     labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                #     weights = torch.cat([weights, torch.zeros(pad_len, dtype=torch.float16)])
                # else:
                #     input_ids = input_ids[:max_len]
                #     attention_mask = attention_mask[:max_len]
                #     labels = labels[:max_len]
                #     weights = weights[:max_len]

                ret = {
                    'image': image,
                    'input_id': input_ids,
                    'attention_mask': attention_mask,
                    'label': labels,
                    'token_weight': weights,
                    'question': question,
                    'answer': answer,
                    'question_type': "VQAYESNO",
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(image)

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret


class DetectionDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.detection_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file

        self.location_map = {
            "lung_upper_lobe_left": [
                "The upper lobe of the left lung.",
                # "Located in the upper section of the left lung.",
                # "This area is positioned in the upper left lung.",
                # "In the upper part of the left lung.",
                # "Situated at the top of the left lung.",
                # "This region belongs to the superior section of the left lung.",
                # "Anatomically found in the uppermost part of the left lung.",
                # "Present in the upper pulmonary region of the left lung."
            ],
            "lung_lower_lobe_left": [
                "The lower lobe of the left lung.",
                # "Located in the lower lobe of the left lung.",
                # "This area is found in the lower part of the left lung.",
                # "Situated in the lower section of the left lung.",
                # "The lower portion of the left lung.",
                # "Positioned in the inferior part of the left lung.",
                # "Present in the bottom lobe of the left lung.",
                # "This structure lies in the basal region of the left lung.",
                # "Situated in the deepest section of the left lung."
            ],
            "lung_upper_lobe_right": [
                "The upper lobe of the right lung.",
                # "This region is in the upper lobe of the right lung.",
                # "Anatomically positioned in the upper right lung.",
                # "Found in the upper section of the right lung.",
                # "Located in the superior lobe of the right lung.",
                # "This area is present in the upper part of the right lung.",
                # "Situated at the top of the right lung.",
                # "The superior pulmonary region of the right lung.",
                # "In the uppermost section of the right lung."
            ],
            "lung_middle_lobe_right": [
                 "The middle lobe of the right lung.",
                # "In the middle lobe of the right lung.",
                # "The middle section of the right lung.",
                # "Found in the central lobe of the right lung.",
                # "Located in the middle part of the right lung.",
                # "This area is positioned in the median region of the right lung.",
                # "Situated between the upper and lower lobes of the right lung.",
                # "This lobe lies in the midsection of the right lung.",
                # "Present in the intermediate portion of the right lung."
            ],
            "lung_lower_lobe_right": [
                "The lower lobe of the right lung.",
                # "Situated in the lower lobe of the right lung.",
                # "Present in the lowest section of the right lung.",
                # "Located in the inferior lobe of the right lung.",
                # "This area is found in the lower part of the right lung.",
                # "Anatomically positioned at the base of the right lung.",
                # "The lower pulmonary region of the right lung.",
                # "Found in the basal lobe of the right lung.",
                # "Situated in the deepest part of the right lung."
            ],
            "heart": [
                "The heart.",
            ]
        }

        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    mtf.CenterSpatialCrop(roi_size=(224, 224, 112)),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)
    
    def get_random_location(self, box):
        """随机选择一个解剖部位的键，并返回一个对应的随机描述"""
        location_key = random.choice(list(box.keys()))
        description = random.choice(self.location_map[location_key])
        return location_key, description


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                example = self.data_list[idx]

                base_name = example["file_name"].split('_')[0] + '_' + example["file_name"].split('_')[1]
                second_base_name = base_name + '_' + example["file_name"].split('_')[2]
                image_path = resolve_ct_rate_image_path(self.args.data_root, example["file_name"])

                image_abs_path = image_path

                image = nib.load(image_abs_path).get_fdata()

                # image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # necessary for original process
                
                image = self.transform(image)

                location_keys, object_location = self.get_random_location(example["bboxes"])

                image_prompt = 'This is a 3D CT scan of a patient as follow: '
                medium_prompt = 'Based on the aforementioned 3D CT scan, please answer the question:'
                
                QUESTION_TEMPLATE = "{Question} \n1. Please output the final answer in <answer> </answer> tags. The final answer must be <answer>[X_min, Y_min, Z_min, X_max, Y_max, Z_max]</answer> tags, where all values are integers. \n2. No extra information or text outside of these tags."

                question = "The 3D image has a shape of (width=224, height==224, depth=112). Assume the coordinate origin is at the top-left-front of the image. Please provide the coordinates of the smallest 3D bounding box that fully encloses the region described in the following sentence: {sentences}"

                question = 'user\n'+  image_prompt + '<|im_start|>' + self.image_tokens + '<|im_end|>' + ' ' + medium_prompt + ' '+ QUESTION_TEMPLATE.format(Question=question.format(sentences=object_location))  + '\nassistant\n'
                
                answer = f'<answer>{str(example["bboxes"][location_keys])}</answer>'


                text_tensor = self.tokenizer(
                    question + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                # write token weight of label not equal to -100 as 1.0
                token_weight = torch.zeros_like(input_id, dtype=torch.float16)
                token_weight[label != -100] = 1.0

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    'token_weight': token_weight
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class SingleChoiceDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.single_choice_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file

        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    mtf.CenterSpatialCrop(roi_size=(224, 224, 112)),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                example = self.data_list[idx]

                image = nib.load(example["VolumeName"]).get_fdata()
                
                image = self.transform(image)

                image_prompt = 'This is a 3D CT scan of a patient as follow: '
                medium_prompt = 'Based on the aforementioned 3D CT scan, please answer the question.'
                CoTprompt = 'Your task:\n1. Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.\n2. No extra information or text outside of these tags.'

                """构建符合多模态对话格式的prompt"""
                question = 'user\n'+  image_prompt + '<|im_start|>' + self.image_tokens + '<|im_end|>' + medium_prompt + example['question'] + ' ' + CoTprompt + '\nassistant\n'
                answer = f"<answer>{example['answer']}</answer>"

                text_tensor = self.tokenizer(
                    question + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                            # write token weight of label not equal to -100 as 1.0
                token_weight = torch.zeros_like(input_id, dtype=torch.float16)
                token_weight[label != -100] = 1.0
                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    'token_weight': token_weight
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class SegIndexDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.seg_index_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file

        self.location_map = {
            "lung_upper_lobe_left": [
                "The upper lobe of the left lung.",
                # "Located in the upper section of the left lung.",
                # "This area is positioned in the upper left lung.",
                # "In the upper part of the left lung.",
                # "Situated at the top of the left lung.",
                # "This region belongs to the superior section of the left lung.",
                # "Anatomically found in the uppermost part of the left lung.",
                # "Present in the upper pulmonary region of the left lung."
            ],
            "lung_lower_lobe_left": [
                "The lower lobe of the left lung.",
                # "Located in the lower lobe of the left lung.",
                # "This area is found in the lower part of the left lung.",
                # "Situated in the lower section of the left lung.",
                # "The lower portion of the left lung.",
                # "Positioned in the inferior part of the left lung.",
                # "Present in the bottom lobe of the left lung.",
                # "This structure lies in the basal region of the left lung.",
                # "Situated in the deepest section of the left lung."
            ],
            "lung_upper_lobe_right": [
                "The upper lobe of the right lung.",
                # "This region is in the upper lobe of the right lung.",
                # "Anatomically positioned in the upper right lung.",
                # "Found in the upper section of the right lung.",
                # "Located in the superior lobe of the right lung.",
                # "This area is present in the upper part of the right lung.",
                # "Situated at the top of the right lung.",
                # "The superior pulmonary region of the right lung.",
                # "In the uppermost section of the right lung."
            ],
            "lung_middle_lobe_right": [
                 "The middle lobe of the right lung.",
                # "In the middle lobe of the right lung.",
                # "The middle section of the right lung.",
                # "Found in the central lobe of the right lung.",
                # "Located in the middle part of the right lung.",
                # "This area is positioned in the median region of the right lung.",
                # "Situated between the upper and lower lobes of the right lung.",
                # "This lobe lies in the midsection of the right lung.",
                # "Present in the intermediate portion of the right lung."
            ],
            "lung_lower_lobe_right": [
                "The lower lobe of the right lung.",
                # "Situated in the lower lobe of the right lung.",
                # "Present in the lowest section of the right lung.",
                # "Located in the inferior lobe of the right lung.",
                # "This area is found in the lower part of the right lung.",
                # "Anatomically positioned at the base of the right lung.",
                # "The lower pulmonary region of the right lung.",
                # "Found in the basal lobe of the right lung.",
                # "Situated in the deepest part of the right lung."
            ],
            "heart": [
                "The heart.",
            ]
        }

        train_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.AddChannel(),
                # Randomly crop the 3D image to a fixed size
                mtf.RandSpatialCrop(roi_size=(224, 224, 112), random_size=False),
                mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449)
                # mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
           
                    mtf.ToTensor(dtype=torch.float),
                    mtf.AddChannel(),
                #    # Randomly crop the 3D image to a fixed size
                    mtf.CenterSpatialCrop(roi_size=(224, 224, 112)),
                    # Normalize intensity of the image from -1000~1000 to -1~1
                    mtf.ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=-1, b_max=1, clip=True),
                    mtf.NormalizeIntensity(
                        subtrahend=0.4978, 
                        divisor=0.2449),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)
    
    def get_random_organ(self, box):
        """随机选择一个解剖部位的键，并返回一个对应的随机描述"""
        location_key = random.choice(list(box.keys()))
        return location_key


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                example = self.data_list[idx]

                base_name = example["file_name"].split('_')[0] + '_' + example["file_name"].split('_')[1]
                second_base_name = base_name + '_' + example["file_name"].split('_')[2]
                image_path = resolve_ct_rate_image_path(self.args.data_root, example["file_name"])

                image_abs_path = image_path

                image = nib.load(image_abs_path).get_fdata()
                
                image = self.transform(image)

                location_keys = self.get_random_organ(example["patch_indices"])

                image_prompt = 'This is a 3D CT scan of a patient as follow: '
                medium_prompt = 'Based on the aforementioned 3D CT scan, please answer the question:'
                
                QUESTION_TEMPLATE = "{Question} 1. Please output the final answer in <answer> </answer> tags. The final answer must be a list of patch indices in the format:\n<answer>[patch_index_0, patch_index_1, ..., patch_index_n]</answer>\nwhere each patch_index is an integer representing the linear index of a patch.\n2. No extra information or text outside of these tags."

                question = "The 3D image has a shape of (width=224, height=224, depth=112). \nThe patch size is (width=16, height=16, depth=8). \nThe patch embedding maps to a shape of (14, 14, 14), where:\n- The total number of patches is 14 (width) * 14 (height) * 14 (depth) = 2744.\n- Each patch is assigned a unique linear index, calculated as:\n  index = z * (14 * 14) + y * 14 + x\n  where (x, y, z) are the patch coordinates in the range [0, 13].\nAssume the coordinate origin is at the top-left-front of the image. \nPlease provide the linear indices of all patches that contain the region described in the following sentence: {sentences}"

                question = 'user\n'+  image_prompt + '<|im_start|>' + self.image_tokens + '<|im_end|>' + ' ' + medium_prompt + ' '+ QUESTION_TEMPLATE.format(Question=question.format(sentences=self.location_map[location_keys]))  + '\nassistant\n'
                
                answer = f'<answer>{str(example["patch_indices"][location_keys])}</answer>'

                text_tensor = self.tokenizer(
                    question + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class PosRECDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = PosREC_templates["cls_questions"]
        self.des_qustions = PosREC_templates["des_questions"]
        self.cls_answers = PosREC_templates["cls_answers"]
        self.des_answers = PosREC_templates["des_answers"]
        self.cls_no_answers = PosREC_templates["cls_no_answers"]
        self.des_no_answers = PosREC_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.cls_answers).format(box_text)
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], box_text)
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REC",
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class PosREGDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = PosREG_templates["cls_questions"]
        self.des_questions = PosREG_templates["des_questions"]
        self.cls_answers = PosREG_templates["cls_answers"]
        self.des_answers = PosREG_templates["des_answers"]

        self.cls_no_questions = PosREC_templates["cls_questions"]
        self.des_no_questions = PosREC_templates["des_questions"]

        self.cls_no_answers = PosREG_templates["cls_no_answers"]
        self.des_no_answers = PosREG_templates["des_no_answers"]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)
                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers).format(cls_list[cls_id])
                    else:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]]))
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_no_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_no_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REG",
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class MultiPosDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class PosSegDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosSegDatasets, self).__init__()
        self.ds_list = [
            MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            # WeightCapDataset(args, tokenizer, mode),

            WeightFindingCapDataset(args, tokenizer, mode),
            ReportVQADataset(args, tokenizer, mode),

            MMVQADataset(args, tokenizer, mode),
            YESNODataset(args, tokenizer, mode),
            REGIONDataset(args, tokenizer, mode),
            OPENDataset(args, tokenizer, mode),

            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode=mode),


            MMVQASLDataset(args, tokenizer, mode),
            YESNOSLDataset(args, tokenizer, mode),
            REGIONSLDataset(args, tokenizer, mode),
            OPENSLDataset(args, tokenizer, mode),
            # WeightSplitCapDataset(args, tokenizer, mode),

            # DetectionDataset(args, tokenizer, mode),
            # SingleChoiceDataset(args, tokenizer, mode),
            # SegIndexDataset(args, tokenizer, mode),
        ]
        
        # 设置每个数据集的采样比例
        # sample_ratios = [0.25, 0.25, 0.25, 0.25, 0.1, 0.25]  # 示例比例，根据需要调整
        # sample_ratios = [0.5, 0.5, 0.5, 0.5, 0.5]  # 示例比例，根据需要调整
        sample_ratios = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
        # sample_ratios = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 
        # sample_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

        

        # 设置随机种子，保证可复现
        np.random.seed(42)
        torch.manual_seed(42)

        # 对每个数据集按比例采
        sampled_datasets = []
        for dataset, ratio in zip(self.ds_list, sample_ratios):
            sample_size = int(len(dataset) * ratio)  # 计算当前数据集的采样数量
            indices = np.random.choice(len(dataset), sample_size, replace=False)  # 随机采样索引
            sampled_dataset = Subset(dataset, indices)  # 创建子集
            sampled_datasets.append(sampled_dataset)

        # 合并所有采样后的数据集
        self.dataset = ConcatDataset( sampled_datasets )


        # VQADataset(args, tokenizer, close_ended=True, mode=mode),
        # VQADataset(args, tokenizer, close_ended=False, mode=mode),

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            # WeightCapDataset(args, tokenizer, mode),
            # WeightSplitCapDataset(args, tokenizer, mode),
            WeightFindingCapDataset(args, tokenizer, mode),
            ReportVQADataset(args, tokenizer, mode),

            MMVQADataset(args, tokenizer, mode),
            YESNODataset(args, tokenizer, mode),
            REGIONDataset(args, tokenizer, mode),
            OPENDataset(args, tokenizer, mode),

            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode=mode),

            # DetectionDataset(args, tokenizer, mode),
            # SingleChoiceDataset(args, tokenizer, mode),
            
            # MultiPosDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
        ]
        # self.dataset = ConcatDataset(self.ds_list)
          # 设置每个数据集的采样比例
        # sample_ratios = [0.25, 0.25, 0.25, 0.25, 0.1, 0.25]  # 示例比例，根据需要调整
        # sample_ratios = [0.5, 0.5, 0.5, 0.5, 0.5]  # 示例比例，根据需要调整
        sample_ratios = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
        # sample_ratios = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 
        # sample_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

        

        # 设置随机种子，保证可复现
        np.random.seed(42)
        torch.manual_seed(42)

        # 对每个数据集按比例采
        sampled_datasets = []
        for dataset, ratio in zip(self.ds_list, sample_ratios):
            sample_size = int(len(dataset) * ratio)  # 计算当前数据集的采样数量
            indices = np.random.choice(len(dataset), sample_size, replace=False)  # 随机采样索引
            sampled_dataset = Subset(dataset, indices)  # 创建子集
            sampled_datasets.append(sampled_dataset)

        # 合并所有采样后的数据集
        self.dataset = ConcatDataset( sampled_datasets )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


