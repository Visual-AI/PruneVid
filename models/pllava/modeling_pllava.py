# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import os
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
import einops
from torch import einsum
# from .modeling_clip import CLIPVisionTransformer, CLIPVisionModel
import time
from .llama import LlamaConfig, LlamaForCausalLM, TextPivotMerge_LayerWise, LlamaAttentionTextPrior, LlamaForCausalLMElastic, LlamaForCausalLMVTP
import itertools
from .configuration_pllava import PllavaConfig
import pickle
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

PLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "",
    "",
    "",
    # See all Llava models at https://huggingface.co/models?filter=llava
]

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        x: input token feature, [B, N, C]
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
    return idx_cluster, cluster_num

def smooth_labels(labels, window_size=4):
    # labels: [B, N]
    padding = window_size // 2
    padded_labels = torch.nn.functional.pad(labels, (padding, padding))
    smoothed_labels = []
    for i in range(labels.shape[1]):
        window = padded_labels[:, i:i+window_size]
        # 计算窗口内的众数
        mode_label, _ = torch.mode(window, dim=1)
        smoothed_labels.append(mode_label)
    smoothed_labels = torch.stack(smoothed_labels, dim=1)
    return smoothed_labels

def segment_lengths(tensor):
    # 获取设备信息（CPU 或 GPU）
    device = tensor.device
    B, N = tensor.shape

    # 列表用于存储每个视频的段长度
    segment_lengths_list = []
    max_segments = 0  # 记录最大段数

    for i in range(B):
        seq = tensor[i]
        # 计算值发生变化的位置
        change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
        # 包含起始和结束位置
        boundaries = torch.cat([torch.tensor([0], device=device), change_points, torch.tensor([N], device=device)])
        # 计算每个段的长度
        lengths = boundaries[1:] - boundaries[:-1]
        segment_lengths_list.append(lengths)
        max_segments = max(max_segments, lengths.numel())

    # 初始化结果张量，填充为0
    result = torch.zeros((B, max_segments), dtype=torch.long, device=device)
    # 将每个视频的段长度填入结果张量
    for i in range(B):
        lengths = segment_lengths_list[i]
        result[i, :lengths.numel()] = lengths

    return result

def refine_clusters(cluster_idx):
    """
    根据给定的聚类结果，对每个批次进行精炼处理。

    Args:
        cluster_idx: Tensor of shape (B, N)，每个元素是聚类的索引。

    Returns:
        refined_cluster_idx: Tensor of shape (B, N)，精炼后的聚类结果。
    """
    import torch
    B, N = cluster_idx.shape
    refined_cluster_idx = cluster_idx.clone()
    for b in range(B):
        clusters = torch.unique(cluster_idx[b])
        segment_info = {}
        # 步骤1：对于每个 cluster，找到其所有的连续片段
        for cluster_label in clusters:
            indices = (cluster_idx[b] == cluster_label).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            # 找到连续片段
            segments = []
            start = indices[0].item()
            prev = indices[0].item()
            for idx in indices[1:]:
                idx = idx.item()
                if idx == prev + 1:
                    prev = idx
                else:
                    # 新的片段
                    segments.append((start, prev))
                    start = idx
                    prev = idx
            # 添加最后一个片段
            segments.append((start, prev))
            segment_info[cluster_label.item()] = segments

        # 步骤2：保留每个 cluster 中最长的片段，其余片段需要重新归类
        for cluster_label, segments in segment_info.items():
            # 找到最长的片段长度
            max_length = 0
            for (start, end) in segments:
                length = end - start + 1
                if length > max_length:
                    max_length = length
            # 如果最长的片段长度为1，且只有长度为1的片段，该 cluster 需要移除
            if max_length == 1:
                for (start, end) in segments:
                    refined_cluster_idx[b, start:end+1] = -1  # -1表示需要重新归类
                continue
            # 保留最长的片段，重新归类其他片段
            for (start, end) in segments:
                length = end - start + 1
                if length == max_length:
                    continue  # 保留最长的片段
                else:
                    refined_cluster_idx[b, start:end+1] = -1  # 需要重新归类

        # 步骤3：对于需要重新归类的片段，按照左右邻居最长的片段的 cluster 进行归类
        idx = 0
        while idx < N:
            if refined_cluster_idx[b, idx] == -1:
                # 找到需要重新归类的片段
                start = idx
                while idx < N and refined_cluster_idx[b, idx] == -1:
                    idx += 1
                end = idx - 1
                # 找到左侧和右侧的邻居 cluster 及其片段长度
                left_cluster_label = None
                left_length = 0
                if start > 0:
                    left_label = refined_cluster_idx[b, start - 1].item()
                    # 左侧片段长度
                    l_idx = start - 1
                    while l_idx >= 0 and refined_cluster_idx[b, l_idx] == left_label:
                        l_idx -= 1
                    left_length = start - l_idx - 1
                    left_cluster_label = left_label
                right_cluster_label = None
                right_length = 0
                if end < N - 1:
                    right_label = refined_cluster_idx[b, end + 1].item()
                    # 右侧片段长度
                    r_idx = end + 1
                    while r_idx < N and refined_cluster_idx[b, r_idx] == right_label:
                        r_idx += 1
                    right_length = r_idx - end - 1
                    right_cluster_label = right_label
                # 选择片段长度较长的邻居 cluster 进行归类，若长度相同，选择左侧
                if left_length > right_length:
                    new_label = left_cluster_label
                elif right_length > left_length:
                    new_label = right_cluster_label
                else:
                    new_label = left_cluster_label if left_cluster_label is not None else right_cluster_label
                # 如果左右邻居都不存在，默认归类为 cluster 0
                if new_label is None:
                    new_label = 0
                # 重新归类
                refined_cluster_idx[b, start:end+1] = new_label
            else:
                idx += 1
    return refined_cluster_idx

@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class PllavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class PllavaMultiModalProjector(nn.Module):
    supported_highres = ['pad_crop_four', 'slide', ]
    def __init__(self, config: PllavaConfig):
        super().__init__()
        self.config = config
        self.use_pooling = config.use_pooling
        self.frame_shape = config.frame_shape
        self.num_frames = config.num_frames
        self.pooling_shape = config.pooling_shape
        
        self.pooling = nn.AdaptiveAvgPool3d(config.pooling_shape)
        self.pooling_small = nn.AdaptiveAvgPool3d((config.pooling_shape[0], int(config.pooling_shape[1]//2), int(config.pooling_shape[2]//2)))
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        input = einops.rearrange(input, 
                                '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                num_videos=num_videos, h=frame_shape[0])
        return input
    
    def grid_convert_Fembeddings2video(self, input, num_videos, frame_shape, t_h, t_w):
        input = einops.rearrange(input, 
                                'num_videos (t_h h t_w w) embed_dims -> num_videos embed_dims (t_h t_w) h w', 
                                num_videos=num_videos, h=frame_shape[0], t_h=t_h, t_w=t_w)
        return input
    
    def global_convert_Fembeddings2video(self, input, num_videos, frame_shape, t_h, t_w):
        input = einops.rearrange(input, 
                                '(num_videos num_frames) (t_h h t_w w) embed_dims -> num_videos embed_dims (t_h t_w num_frames) h w', 
                                num_videos=num_videos, h=frame_shape[0] // t_h, t_h=t_h, t_w=t_w)
        return input
    
    def convert_video2Fembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> (num_videos num_frames) (h w) embed_dims ', )
        return input

    def convert_video2MMembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> num_videos (num_frames h w) embed_dims ', )
        return input
    
    
    def forward(self, image_features, media_type, batch_size=None, num_videos=None, num_frames=16, frame_shape=(24, 24)):
        # frame_shape = self.frame_shape
        # num_frames = self.num_frames
        assert media_type in ( 'video', 'image'), f'only image or video, but got media_type {media_type}'
        # hidden_states, global_hidden_states = image_features
        hidden_states = image_features
        if media_type == 'image':
            hidden_states = hidden_states.repeat(num_frames, 1, 1)
        
        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        # #TODO: temporal code, should ensure num_frames == total frames in data loading later
        # if total_frames < num_frames and self.use_pooling: # 
        #     multiplier = int(num_frames/total_frames)+1
        #     hidden_states = hidden_states.repeat_interleave(multiplier, dim=0)[:num_frames]
        #     total_frames, spatial_seqlen, embed_dims = hidden_states.shape

        # assert total_frames % num_frames == 0
        assert frame_shape[0] * frame_shape[1] == spatial_seqlen
                
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, num_videos * batch_size, frame_shape) # b c t h w

        if num_frames == self.num_frames:
            hidden_states_videos = self.pooling(hidden_states_videos)
        else:
            hidden_states_videos = nn.functional.adaptive_avg_pool3d(hidden_states_videos, (num_frames, self.pooling_shape[1], self.pooling_shape[2]))

        hidden_states = einops.rearrange(hidden_states_videos, 'batch_size_num_videos embed_dims num_frames h w -> batch_size_num_videos (num_frames h w) embed_dims')

        return hidden_states


PLLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    PLLAVA_START_DOCSTRING,
)
class PllavaPreTrainedModel(PreTrainedModel):
    config_class = PllavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        # if isinstance(module, (nn.Linear, nn.Conv2d)):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, PllavaMultiModalProjector):
            pass
            # module.register_embed.data.normal_(mean=0.0, std=std)
            # print('-----------------', module)
            # if self.config.register:
            #     module.register_embed.data.zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


PLLAVA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`LlavaProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    PLLAVA_START_DOCSTRING,
)

class PllavaForConditionalGeneration(PllavaPreTrainedModel):
    def __init__(self, config: PllavaConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = PllavaMultiModalProjector(config)

        self.vocab_size = config.vocab_size

        config.text_config._attn_implementation = "sdpa"
        
        config.text_config.kv_mode = "origin"
        config.text_config.num_frames = config.num_frames
        config.text_config.head = config.head
        config.text_config.selected_layer = config.selected_layer
        config.text_config.alpha = config.alpha
        config.text_config.softmax = config.softmax
        config.text_config.pooling_shape = config.pooling_shape
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.text_config.pad_token_id
        assert self.pad_token_id is not None, 'provide the model with pad_token_id, this would be used to arranging new embedings'
        config.text_config.pad_token_id = self.pad_token_id
        self.language_model = LlamaForCausalLMVTP(config.text_config)
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) > nb_image_pad[:, None].to(target_device)

        # # somthing really weird here.
        # temp1 = (image_to_overwrite.cumsum(-1) > nb_image_pad[:, None].to(target_device)) & image_to_overwrite
        # # this is for right padding
        # temp2 = (image_to_overwrite.cumsum(-1) <=  num_special_image_tokens.max() * num_image_patches - nb_image_pad[:, None]) & image_to_overwrite
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        if final_embedding.dtype != image_features.dtype:
            image_features = image_features.to(final_embedding.dtype)
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
    
    def feature_select(self, image_forward_outs, select_layer):
        image_features = image_forward_outs.hidden_states[select_layer] # penultimate layer output
        image_features = image_features[:, 1:]
        return image_features
    
    def compute_cluster_vectors(self, image_key_vectors, cluster_key_idx, num_cluster):
        """
        Args:
            image_key_vectors: Tensor of shape (B, L, D), the feature vectors
            cluster_key_idx: Tensor of shape (B, L), cluster indices for each vector
            num_cluster: int, the total number of clusters

        Returns:
            cluster_vectors: Tensor of shape (B, num_cluster, D), the averaged features for each cluster
        """
        # image_key_vectors: (B, L, D)
        # cluster_key_idx: (B, L)
        # num_cluster: integer, number of clusters

        B, L, D = image_key_vectors.shape

        # Step 1: 将cluster_key_idx进行one-hot编码
        # 得到的cluster_key_idx_onehot形状为 (B, L, num_cluster)
        cluster_key_idx_onehot = F.one_hot(cluster_key_idx, num_classes=num_cluster).to(dtype=image_key_vectors.dtype)

        # Step 2: 计算每个cluster的特征和
        # 首先调整cluster_key_idx_onehot的维度，使其变为 (B, num_cluster, L)
        cluster_key_idx_onehot_t = cluster_key_idx_onehot.permute(0, 2, 1)

        # 然后通过矩阵乘法计算每个cluster的特征和，得到的cluster_sums形状为 (B, num_cluster, D)
        cluster_sums = torch.bmm(cluster_key_idx_onehot_t, image_key_vectors)

        # Step 3: 计算每个cluster的元素数量
        # cluster_counts形状为 (B, num_cluster)
        cluster_counts = cluster_key_idx_onehot.sum(dim=1)

        # Step 4: 计算每个cluster的平均特征
        # 先避免除以0，将cluster_counts中为0的值替换为1
        cluster_counts_nonzero = cluster_counts.clone()
        cluster_counts_nonzero[cluster_counts_nonzero == 0] = 1

        # 计算平均值，结果cluster_features形状为 (B, num_cluster, D)
        cluster_features = cluster_sums / cluster_counts_nonzero.unsqueeze(-1)

        # Step 5: 对于没有元素的cluster，将其特征设置为0
        zero_mask = (cluster_counts == 0).unsqueeze(-1)  # (B, num_cluster, 1)
        cluster_features = cluster_features.masked_fill(zero_mask, 0)

        return cluster_features  # (B, num_cluster, D)
    
    def spatial_merge_tokens(self, feature, num_cluster, k):
        cluster_idx, _ = cluster_dpc_knn(feature, cluster_num=num_cluster, k=k)
        feature = self.compute_cluster_vectors(feature, cluster_idx, num_cluster=num_cluster)
        return feature

    def merge_frames_dynamic(self, frames, threshold=0.8, k=7):
        B, L, C = frames.shape
        assert L == self.config.num_frames * self.config.pooling_shape[1] * self.config.pooling_shape[2]
        frames = frames.view(B, self.config.num_frames, self.config.pooling_shape[1]*self.config.pooling_shape[2], C) # B T L C
        idx_clusters, _ = cluster_dpc_knn(frames.mean(dim=2), cluster_num=int(self.config.num_frames*self.config.temporal_segment_ratio), k=k)
        idx_clusters = refine_clusters(idx_clusters)
        window_list = segment_lengths(idx_clusters)

        L = self.config.pooling_shape[1]*self.config.pooling_shape[2]
        
        static_features = []
        dynamic_features = []
        static_sizes = []
        dynamic_sizes = []
        
        start_idx = 0
        for window_size in window_list[0]:  # 假设window_list的形状为(B, S)
            # 获取当前window的帧
            current_frames = frames[:, start_idx:start_idx+window_size, :, :] # B W L C
            
            # 计算相似度
            frames_normed = F.normalize(current_frames, p=2, dim=-1)
            frames_sim = einsum('b w l c, b t l c -> b w t l', frames_normed, frames_normed)
            frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size*(window_size-1)) # B L
            
            # 创建mask
            mask = frames_sim > threshold
            mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C) # B W L C
            
            # 处理静态特征
            static_mask = mask_expand
            static_feat = torch.masked_select(current_frames, static_mask).view(B, window_size, -1, C).mean(dim=1)
            if static_feat.shape[1] > 14:
                static_feat = self.spatial_merge_tokens(static_feat, num_cluster=static_feat.shape[1]//2, k=7)
            static_features.append(static_feat)
            static_sizes.append(static_feat.shape[1])
            
            # 处理动态特征
            dynamic_mask = ~mask_expand
            dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, window_size, -1, C)
            dynamic_window_list = []
            for i in range(window_size):
                dynamic_feat_window = dynamic_feat[:,i,:,:]
                if dynamic_feat_window.shape[1] > 14:
                    dynamic_feat_window = self.spatial_merge_tokens(dynamic_feat_window, num_cluster=int(dynamic_feat_window.shape[1]*self.config.cluster_ratio), k=7)
                dynamic_window_list.append(dynamic_feat_window)
            dynamic_feat = torch.cat(dynamic_window_list, dim=1)
            # dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, -1, C)
                
            dynamic_features.append(dynamic_feat)
            dynamic_sizes.append(dynamic_feat.shape[1])
            
            start_idx += window_size

        # 合并所有特征
        final_features = []
        for static_feature, dynamic_feature in zip(static_features, dynamic_features):
            final_features.append(static_feature)
            final_features.append(dynamic_feature)
        final_features = torch.cat(final_features, dim=1)

        window_sizes = window_list[0].tolist()  # 转换为列表形式

        return final_features, static_sizes, dynamic_sizes, window_sizes
    
    def merge_frames(self, frames, window_size=4, threshold=0.8):
        B, L, C = frames.shape
        assert L == self.config.num_frames * self.config.pooling_shape[1] * self.config.pooling_shape[2]
        frames = frames.view(B, self.config.num_frames, self.config.pooling_shape[1]*self.config.pooling_shape[2], C) # B T L C

        idx_clusters, _ = cluster_dpc_knn(frames.mean(dim=2), cluster_num=4, k=3)
        idx_clusters = refine_clusters(idx_clusters)
        

        L = self.config.pooling_shape[1]*self.config.pooling_shape[2]
        window_nums = self.config.num_frames // window_size
        frames = frames.view(B, window_nums, window_size, L, C)
        frames_normed = F.normalize(frames, p=2, dim=-1) # B T//W W L C
        frames_sim = einsum('b s w l c, b s t l c ->  b s w t l', frames_normed, frames_normed) # B T//W W W L
        frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size*(window_size-1)) # B T//W L
        mask = frames_sim > threshold
        mask_expand = mask.view(B, window_nums, 1, L, 1).expand(-1, -1, window_size, -1, C) # B T//W W L C
        static_features = []
        for i in range(window_nums):
            mask_expand_window = mask_expand[:, i, :, :, :] # B W L C
            window_feat = torch.masked_select(frames[:,i,:,:,:], mask_expand_window).view(B, window_size, -1, C).mean(dim=1) # B -1 C
            static_features.append(window_feat)
        static_sizes = [feat.shape[1] for feat in static_features]

        dynamic_mask_expand = ~mask_expand
        dynamic_features = []
        for i in range(window_nums):
            dynamic_mask_expand_window = dynamic_mask_expand[:, i, :, :, :] # B W L C
            window_feat = torch.masked_select(frames[:,i,:,:,:], dynamic_mask_expand_window).view(B, -1, C) # B -1 C
            dynamic_features.append(window_feat)
        dynamic_sizes = [feat.shape[1] for feat in dynamic_features]

        final_features = []
        window_sizes = []
        for static_feature, dynamic_feature in zip(static_features, dynamic_features):
            final_features.append(static_feature)
            final_features.append(dynamic_feature)
            window_sizes.append(window_size)
        final_features = torch.cat(final_features, dim=1)

        return final_features, static_sizes, dynamic_sizes, window_sizes

    @add_start_docstrings_to_model_forward(PLLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PllavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        media_type: str = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PllavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        text_indices = None
        static_sizes, dynamic_sizes, window_sizes = [], [], []
        time1 = time.time()
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            # if input_ids.shape[1] != 1 and (input_ids == self.config.image_token_index).sum() == 0:
            #     input_ids = input_ids[:, :1]
                # attention_mask = attention_mask[:, :-1]
            no_img_input_ids = torch.where(input_ids!=self.config.image_token_index, input_ids, self.pad_token_id) # some model used up all the embeddings
            inputs_embeds = self.get_input_embeddings()(no_img_input_ids)
            batch_size = inputs_embeds.shape[0]
            # 2. Merge text and images
            flag = False
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer] #  ( b, img_seqlen, embed_dim)
                # global_selected_image_feature = global_image_outputs.hidden_states[vision_feature_layer] #  ( b, img_seqlen, embed_dim)
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    raise ValueError("not implemented")
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )
                
                image_features = self.multi_modal_projector(selected_image_feature,
                                                            media_type,
                                                            batch_size=batch_size,
                                                            num_videos=pixel_values.shape[0]//self.config.num_frames//batch_size,
                                                            num_frames=self.config.num_frames)

                image_features, static_sizes, dynamic_sizes, window_sizes = self.merge_frames_dynamic(image_features, threshold=self.config.tau, k=7)

                inputs_embeds, attention_mask, labels, position_ids, input_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

                if labels is None:
                    # labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
                    flag = True
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    # print('input_ids:', input_ids, input_ids.shape, attention_mask.shape, target_seqlen)
                    if target_seqlen < attention_mask.shape[1]:
                        extended_attention_mask = torch.ones(
                            (attention_mask.shape[0], 0),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                    else:
                        extended_attention_mask = torch.ones(
                            (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )

                        # Filter out only the tokens that can be un-attended, this can happen
                        # if one uses Llava + Fused modules where the cache on the
                        # first iteration is already big enough, or if one passes custom cache
                        valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                        new_batch_index = batch_index[valid_indices]
                        new_non_attended_tokens = non_attended_tokens[valid_indices]

                        # Zero-out the places where we don't need to attend
                        extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        
        # print("input_ids", input_ids, input_ids.shape, inputs_embeds.shape, self.pad_token_id)
        
        output_attentions = True
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            static_sizes=static_sizes,
            dynamic_sizes=dynamic_sizes,
            window_sizes=window_sizes,
        )

        logits = outputs.logits
        try:
            attention_mask = outputs.attention_mask if outputs.attention_mask is not None else attention_mask
        except:
            pass
        if flag:
            labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return PllavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def clean_cache(self):
        self.TAGET_MODULE = {
            "text_prior_merge": LlamaAttentionTextPrior,
        }
        # for name, m in self.language_model.named_modules():
        try:
            for name, m in self.language_model.named_modules():
                if isinstance(m, self.TAGET_MODULE[self.config.text_config.kv_mode]):
                    m._clean_kv_scores()
        except Exception as e:
            print("Failed to find clean module:", e)
    
    @torch.no_grad()
    def generate(
        self,
        **kwargs,
    ):
        result = super().generate(
            **kwargs
        )
        
        return result
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                # input_ids = input_ids[:, past_length:]
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        media_type = kwargs.get('media_type', None)
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "media_type": media_type,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
