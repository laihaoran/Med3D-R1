import torch
import torch.nn as nn
import math

class SmartProjector(nn.Module):
    def __init__(self, image_size, patch_size, pooling_size, image_dim, hidden_dim, topk, use_random=False, random_num=1000, decoder_block_path=None):
        super(SmartProjector, self).__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)  # Linear layer for image tokens
        self.topk = topk  # Number of top image tokens to select
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        if decoder_block_path is None:
            raise ValueError("SmartProjector requires decoder_block_path; release code does not ship decoder block weights.")
        self.decoder_block = torch.load(decoder_block_path, map_location="cpu")

        # cloase the grad of the decoder block
        for param in self.decoder_block.parameters():
            param.requires_grad = False
        self.use_random = use_random
        self.random_num = random_num
        self.count = 0

    def forward(self, image_tokens, text_tokens, labels,  attention_mask=None):
        """
        Args:
            image_tokens: Tensor of shape (B, L, D_image), image tokens
            text_tokens: Tensor of shape (B, T, D_text), text tokens

        Returns:
            selected_image_tokens: Tensor of shape (B, topk, hidden_dim), top-k image tokens
        """
        B, L, D_image = image_tokens.shape
        B, T, D_text = text_tokens.shape
        
        # Step 1: Project image tokens to hidden_dim
        projected_image_tokens = self.image_proj(image_tokens)  # Shape: (B, L, hidden_dim)

        self.count += 1

        if self.use_random:
            if self.count > self.random_num :
        
                combined_tokens = torch.cat(
                        (text_tokens[:, :1, :], projected_image_tokens, text_tokens[:, (self.topk + 1):, :]), dim=1)
                if attention_mask is not None:
                    image_mask = torch.ones(B, L, device=attention_mask.device, dtype=attention_mask.dtype)
                    combined_attention_mask = torch.cat(
                            (attention_mask[:, :1], image_mask, attention_mask[:, (self.topk + 1):]), dim=1)
                else:
                    combined_attention_mask = None

                # Forward pass through the loaded decoder block
                with torch.no_grad():

                    # create position embeddings to be shared across the decoder layers
                    # position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    combined_tokens = self.decoder_block.input_layernorm(combined_tokens)
                    position_ids = torch.arange(combined_tokens.size(1), device=image_tokens.device).unsqueeze(0)
                    # combined_attention_mask = convert_global_attention_mask(combined_attention_mask)

                    bsz, q_len, _ = combined_tokens.size()

                    # Compute Q, K, V
                    query_states = self.decoder_block.self_attn.q_proj(combined_tokens)
                    key_states = self.decoder_block.self_attn.k_proj(combined_tokens)
                    value_states = self.decoder_block.self_attn.v_proj(combined_tokens)

                    query_states = query_states.view(bsz, q_len, self.decoder_block.self_attn.num_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.decoder_block.self_attn.num_key_value_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.decoder_block.self_attn.num_key_value_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)

                    cos, sin = self.decoder_block.self_attn.rotary_emb(value_states, position_ids)
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    key_states = repeat_kv(key_states, self.decoder_block.self_attn.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.decoder_block.self_attn.num_key_value_groups)

                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.decoder_block.self_attn.head_dim)
                    # Step 4: Extract the attention map
                    # attn_map = attn_weights.mean(dim=1)  # Average over all heads, Shape: (B, L+T, L+T)
                
                    topk_indices = select_topk_image_tokens(attn_weights, L, topk=self.topk, attention_mask=combined_attention_mask, labels=labels)
                    # print(topk_indices)
            else:
                # random select topk
                topk_indices = torch.randint(0, L, (B, self.topk), device=image_tokens.device)
        else:
            combined_tokens = torch.cat(
                    (text_tokens[:, :1, :], projected_image_tokens, text_tokens[:, (self.topk + 1):, :]), dim=1)
            if attention_mask is not None:
                image_mask = torch.ones(B, L, device=attention_mask.device, dtype=attention_mask.dtype)
                combined_attention_mask = torch.cat(
                        (attention_mask[:, :1], image_mask, attention_mask[:, (self.topk + 1):]), dim=1)
            else:
                combined_attention_mask = None

            # Forward pass through the loaded decoder block
            with torch.no_grad():

                # create position embeddings to be shared across the decoder layers
                # position_embeddings = self.rotary_emb(hidden_states, position_ids)
                combined_tokens = self.decoder_block.input_layernorm(combined_tokens)
                position_ids = torch.arange(combined_tokens.size(1), device=image_tokens.device).unsqueeze(0)
                # combined_attention_mask = convert_global_attention_mask(combined_attention_mask)

                bsz, q_len, _ = combined_tokens.size()

                # Compute Q, K, V
                query_states = self.decoder_block.self_attn.q_proj(combined_tokens)
                key_states = self.decoder_block.self_attn.k_proj(combined_tokens)
                value_states = self.decoder_block.self_attn.v_proj(combined_tokens)

                query_states = query_states.view(bsz, q_len, self.decoder_block.self_attn.num_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.decoder_block.self_attn.num_key_value_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.decoder_block.self_attn.num_key_value_heads, self.decoder_block.self_attn.head_dim).transpose(1, 2)

                cos, sin = self.decoder_block.self_attn.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                key_states = repeat_kv(key_states, self.decoder_block.self_attn.num_key_value_groups)
                value_states = repeat_kv(value_states, self.decoder_block.self_attn.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.decoder_block.self_attn.head_dim)                

                topk_indices = select_topk_image_tokens(attn_weights, L, topk=self.topk, attention_mask=combined_attention_mask, labels=labels)

    

        # Gather the top-k image tokens
        selected_image_tokens = torch.gather(
            projected_image_tokens, 1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, projected_image_tokens.size(-1))
        )  # Shape: (B, topk, hidden_dim)
        return selected_image_tokens
    
    @property
    def proj_out_num(self):
        # num = 1
        # for n in self.num_patches_post:
        #     num *= n
        return self.topk

def find_question_length(labels):
    """
    找到 labels 中的文本问题部分的长度 (连续的 -100 数量)。
    
    Args:
        labels (torch.Tensor): 形状为 (B, L)，包含问题、答案和 padding。

    Returns:
        question_lengths (torch.Tensor): 每个样本的文本问题长度 (B,)。
    """
    # 找到第一个非 -100 的索引
    is_not_negative = labels != -100  # Mask of valid tokens
    first_non_negative_idx = is_not_negative.float().argmax(dim=1)  # First valid token index for each sample

    return first_non_negative_idx



def select_topk_image_tokens(attention_map, L, topk, attention_mask=None, labels=None):
    """
    根据图像 tokens 对文本 tokens（非 padding 部分）的平均注意力得分，选择 top-k 图像 tokens。

    Args:
        attention_map (torch.Tensor): Shape (B, L+T, L+T), 全局 attention map
        attention_mask (torch.Tensor): Shape (B, L+T)，用于屏蔽 padding 的 mask
        L (int): 图像 tokens 的数量
        topk (int): 需要选择的 top-k 图像 tokens 数量

    Returns:
        topk_indices (torch.Tensor): Shape (B, topk)，每个样本的 top-k 图像 token 索引
    """
    if attention_mask is None:
        B, _, _, _ = attention_map.shape

        # Step 1: 提取图像对文本的 Attention Scores (B, L, T)
        image_to_text_attention = attention_map[:, :, 1:L+1, L+2:]  # 提取图像到文本的部分 (B, L, T)
   
        # Step 2: 计算每个图像 token 对文本的平均注意力得分 (B, L)
        # 直接对文本维度求均值，不考虑 padding 或 mask
        image_to_text_attention = torch.softmax(image_to_text_attention, dim=-1)
        image_importance_scores = image_to_text_attention.mean(dim=1)  # Shape: (B, L)
        image_importance_scores = image_importance_scores.mean(dim=-1)  # Shape: (B,)

        # Step 3: 根据得分选择 Top-K 图像 Token 索引 (B, topk)
        topk_indices = torch.topk(image_importance_scores, k=topk, dim=-1, sorted=False).indices
    else:
        B, _,_,_ = attention_map.shape

        labels = labels[:, 1+ topk + 1:]  # 去掉 [CLS]、图像 tokens 和 [SEP]，只保留文本部分

        question_index = find_question_length(labels)

        # Step 1: 提取图像对文本的 Attention Scores (B, L, T)
        image_to_text_attention = attention_map[:, :, 1:L+1, L+2:]  # 只取图像到文本的部分 (B, L, T)

        # 修正文本的 attention_mask，只保留问题部分
        attention_mask[:, L+2:] = 0  # 清零文本部分
        for i in range(B):
            attention_mask[i, L+2:L+2 + question_index[i]] = 1  # 恢复问题部分


        # Step 2: 利用文本的 attention_mask 屏蔽 padding (B, T)
        text_mask = attention_mask[:, L+2:]  # 提取文本部分的 mask, Shape: (B, T)

        # Step 3: 在 Softmax 前屏蔽无效区域
        text_mask_expanded = text_mask.unsqueeze(1).unsqueeze(1)  # 扩展为 (B, 1,1 T)
        image_to_text_attention = image_to_text_attention.masked_fill(
            text_mask_expanded == 0, float("-inf")
        )  # 将无效区域设置为 -inf

        # 计算 Softmax（仅对有效区域）
        image_to_text_attention = torch.softmax(image_to_text_attention, dim=-2)
        # 将 NaN 替换为 0
        image_to_text_attention = torch.nan_to_num(image_to_text_attention, nan=0.0)
        image_to_text_attention = image_to_text_attention.mean(dim=1)  # 求均值
            
        # text_mask = text_mask.unsqueeze(1)  # 扩展为 (B, 1, T)，方便与 attention 相乘
        # image_to_text_attention = image_to_text_attention * text_mask  # 屏蔽 padding 部分

        # Step 3: 计算每个图像 token 对文本的平均注意力得分 (B, L)
        text_token_counts = text_mask_expanded.sum(dim=-1).squeeze(-1).clamp(min=1)  # 避免除以 0, Shape: (B, 1)
        image_importance_scores = image_to_text_attention.sum(dim=-1) / text_token_counts  # 求均值

        # Step 4: 根据得分选择 Top-K 图像 Token 索引 (B, topk)
        topk_indices = torch.topk(image_importance_scores, k=topk, dim=-1, sorted=False).indices

    return topk_indices



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



def convert_global_attention_mask(attention_mask):
    """
    Convert attention mask from shape (B, L) to global shape (B, 1, L, L).

    Args:
        attention_mask (torch.Tensor): Input attention mask of shape (B, L),
                                        where 1 indicates valid tokens, and 0 indicates padding.

    Returns:
        torch.Tensor: Converted attention mask of shape (B, 1, L, L), where
                      invalid positions are set to -inf and valid ones are 0.0.
    """
    # Ensure the mask is float for numerical operations
    attention_mask = attention_mask.float()

    # Expand dimensions to (B, 1, L, 1) and broadcast to (B, 1, L, L)
    expanded_mask = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]

    # Convert mask: valid positions (1) -> 0.0, invalid positions (0) -> -inf
    global_attention_mask = (1.0 - expanded_mask) * torch.finfo(expanded_mask.dtype).min

    return global_attention_mask

# Utility function to save a specific block
def save_block(transformer_model, n_block, save_path):
    """
    Save the specified block (LlamaDecoderLayer) from a Transformer model.

    Args:
        transformer_model: The pretrained Llama model.
        n_block: Index of the block to save.
        save_path: Path to save the block.
    """
    block = transformer_model.layers[n_block]  # Access the decoder layers
    torch.save(block, save_path)

# Example usage:
# Save the n-th block
# llama_model = ...  # Load your Llama model
# save_block(llama_model, n_block=5, save_path="decoder_block_5.pth")

# Load and use the block
# projector = Projector(image_dim=4096, text_dim=4096, hidden_dim=4096, topk=10, block_path="decoder_block_5.pth")
