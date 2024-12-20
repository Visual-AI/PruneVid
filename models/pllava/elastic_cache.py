import torch
import random
import numpy as np
import math
def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

def generate_random_name():
    import string
    import random
    letters = string.ascii_lowercase + string.digits + string.ascii_uppercase
    return ''.join(random.sample(letters, 10))

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

import torch.nn.functional as F
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class VTPWindowCache:
    def __init__(self, alpha=0.2, total_num_layers=32, selected_layer=9, pooling_shape=(16, 12, 12), num_frames=16, pad_token_id=None, head=0, softmax=1.0):
        self.alpha = alpha
        self.total_num_layers = total_num_layers
        self.selected_layer = selected_layer
        self.pooling_shape = pooling_shape
        self.num_frames = num_frames
        self.pad_token_id = pad_token_id
        self.head = head
        self.softmax = softmax
        self.img_start, self.img_end = None, None

    def process_attention(self, text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes):
        # [head_num, num_query, num_img]
        b, head_num, num_query, num_img = text_to_image_attentions.shape
        assert b == 1
        assert len(static_sizes) == len(dynamic_sizes) == len(window_sizes)
        text_to_image_attentions = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0] # num_img

        start_idx, end_idx = 0, 0
        topk_indices_list = []
        static_len, dynamic_len = np.sum(static_sizes), np.sum(dynamic_sizes)

        alpha = self.alpha
        for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
            end_idx = start_idx + static_size + dynamic_size
            window_attentions = text_to_image_attentions[start_idx:end_idx]
            static_attentions = window_attentions[:static_size]
            num_retain_static_tokens = int(static_size * alpha)
            _, static_topk_indices = torch.topk(static_attentions, k=num_retain_static_tokens, dim=-1) # num_retain_static_tokens
            static_topk_indices = static_topk_indices + start_idx
            topk_indices_list.append(static_topk_indices)

            dynamic_attentions = window_attentions[static_size:].view(window_size, -1)
            num_retain_dynamic_tokens = int(dynamic_attentions.shape[-1] * alpha)
            _, dynamic_topk_indices = torch.topk(dynamic_attentions, k=num_retain_dynamic_tokens, dim=-1) # window_size num_retain_dynamic_tokens
            dynamic_topk_indices = dynamic_topk_indices + start_idx + static_size
            dynamic_topk_indices_list = []

            for i in range(window_size):
                dynamic_topk_indices_list.append(dynamic_topk_indices[i] + dynamic_attentions.shape[-1] * i)
            dynamic_topk_indices = torch.cat(dynamic_topk_indices_list, dim=0)
            topk_indices_list.append(dynamic_topk_indices)
            start_idx = end_idx
        
        topk_indices = torch.cat(topk_indices_list, dim=0)

        return topk_indices
    
    def obtain_language_attention(self, input_ids, attentions, pad_token, text_indices=None):
        pad_token = self.pad_token_id
        batch_img_indices, seq_img_indices = torch.where(input_ids == pad_token)
        img_start, img_end = seq_img_indices[0].item(), seq_img_indices[-1].item()
        layer_num, head_num, seq_len, seq_len = attentions.shape

        text_to_image_attentions = attentions[:, :, img_end+1:, img_start: img_end+1]

        image_to_image_attentions = attentions[:, :, img_start: img_end+1, img_start: img_end+1]
        text_to_text_attentions = attentions[:, :, img_end+1:, img_end+1:]

        return text_to_image_attentions, text_to_text_attentions, image_to_image_attentions, img_start, img_end, seq_len
    
    def prompt_prefill(self, past_key_values=None, input_ids=None, attentions=None, hidden_states=None, past_hidden_states=None, causal_mask=None, attention_mask=None, pad_token_id=None, position_ids=None, text_indices=None, attn_shallower=None, static_sizes=[], dynamic_sizes=[], window_sizes=[], decoding_flag=False):
        text_to_image_attentions, text_to_text_attentions, image_to_image_attentions, img_start, img_end, seq_len = self.obtain_language_attention(input_ids, attentions, pad_token_id) # batch_size, head_num, num_query, num_img
        
        topk_indices = self.process_attention(text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes) # num

        index_list = topk_indices
        index_list = index_list + img_start # consider that the image is not the first token
        index_list = index_list.sort(descending=False)[0] # The image tokens should be put in order
        image_index_list = index_list
        self.img_start, self.img_end = img_start, img_start + len(index_list) - 1
        index_list_pre_image = torch.arange(0, img_start, device=input_ids.device) # maintain the tokens before the image
        index_list_post_image = torch.arange(img_end+1, seq_len, device=input_ids.device) # maintain the tokens after the image
        index_list = torch.cat([index_list_pre_image, image_index_list, index_list_post_image], dim=0) # concat the tokens before and after the image
        for layer_idx in range(self.selected_layer+1):                                                                                                                               
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:,:,index_list,:].contiguous()
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:,:,index_list,:].contiguous()

        hidden_states = hidden_states[:,index_list,:]
        if causal_mask is not None:
            causal_mask = causal_mask[:,:,index_list,:][:,:,:,index_list]
        if attention_mask is not None:
            attention_mask = attention_mask[:, index_list]

        updated_hidden_states = () if past_hidden_states is not None else None
        if updated_hidden_states is not None:
            for old_hidden_state in past_hidden_states:
                updated_hidden_states += (old_hidden_state[:,index_list,:], )

        num_tokens_left = index_list.shape[0]
        position_ids = position_ids[:,index_list]
        cache_postition = torch.arange(num_tokens_left, device=input_ids.device)

        return past_key_values, hidden_states, updated_hidden_states, causal_mask, attention_mask, position_ids, cache_postition


    def __call__(self, past_key_values=None, input_ids=None, attentions=None, hidden_states=None, past_hidden_states=None, causal_mask=None, attention_mask=None, pad_token_id=None, position_ids=None, text_indices=None, attn_shallower=None, static_sizes=[], dynamic_sizes=[], window_sizes=[], decoding_flag=False):
                
        return self.prompt_prefill(past_key_values, input_ids, attentions, hidden_states, past_hidden_states, causal_mask, attention_mask, pad_token_id, position_ids, text_indices, attn_shallower, static_sizes, dynamic_sizes, window_sizes, decoding_flag)

class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.,
        distance=0,
        layer_num=40,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1)
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num
        self.num_tokens = 0

        self.selected_idx = 0

        self.image_position = None

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention.mean(dim=1) for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        # attn_score = attn_score.mean(dim=1, keepdim=False)
        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[:, idx, :idx+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False

                selected_idx_all = []
                merge_idx_all = []
                throw_idx_all = []
                for idx in range(self.layer_num):
                    selected_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) > forget_num)[0] + self.start_size
                    throw_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) <= forget_num)[0]
                    merge_idx = []
                    for i in range(len(throw_idx)):
                        merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)

                    selected_idx = torch.cat([torch.arange(self.start_size, device=selected_idx.device), selected_idx, torch.tensor([seq_len - self.protect_size], device=selected_idx.device)], dim=0) # the last token is always kept

                    selected_idx_all.append(selected_idx)
                    merge_idx_all.append(merge_idx)
                    throw_idx_all.append(throw_idx)

                if self.distance > 0:
                    self.selected_idx = self.distance
                else:
                    self.selected_idx = seq_len - forget_num + self.distance

                past_key_values_return = []
                for idx, (k, v) in enumerate(past_key_values):
                    selected_idx = selected_idx_all[idx]
                    merge_idx = merge_idx_all[idx]
                    throw_idx = throw_idx_all[idx]

                    k_forget = k.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_forget = v.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

                    k = k.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]), k_forget, 'mean')
                    v = v.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]), v_forget, 'mean')

                    k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))

                    past_key_values_return.append([k_new, v_new])
                    print(len(selected_idx), len(set(list(selected_idx.cpu().numpy()))), len(selected_idx)/seq_len, ((selected_idx >= self.image_position) & (selected_idx < self.image_position+2304)).sum().item()/2304, ((selected_idx < self.image_position).sum() / self.image_position).item(), ((selected_idx >= self.image_position+2304).sum() / (seq_len - self.image_position-2304)).item())
                    # print(idx, forget_num, seq_len, self.ratio)
                return past_key_values_return
            else:
                selected_idx = self.selected_idx
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values]
            

class LocalCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.ratio = ratio

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            return [[torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, forget_num + self.start_size, seq_len),],
                        dim=self.k_seq_dim,),
                    torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, forget_num + self.start_size, seq_len),],
                        dim=self.v_seq_dim,),]
                for k, v in past_key_values]
        

class H2OCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(self.cache_size + 1)
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False).mean(dim=0, keepdim=False)

        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[idx][:idx+1]
                self.score_sum[:len(cur_score)] += cur_score
        else:
            attn_score = attn_score.squeeze(0)
            self.score_sum[:seq_len] += attn_score

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        self.protect_size = 1
        if forget_num <= 0:
            return past_key_values_new
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False
                selected_idx = torch.where(torch.argsort(self.score_sum[:(seq_len - self.protect_size)]) > forget_num)[0]
                selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len, device=selected_idx.device)], dim=0)
                past_key_values_return = []
                for k, v in past_key_values_new:
                    k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))
                    past_key_values_return.append([k_new, v_new])
                
                return past_key_values_return
            else:
                selected_idx = self.score_sum[self.start_size:(seq_len - self.protect_size)].argmin() + self.start_size
                self.score_sum[(selected_idx):-1] = self.score_sum[(selected_idx+1):].clone()
                
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values_new]