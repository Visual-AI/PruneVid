import copy
import itertools
import re
import os
import json
from enum import auto, Enum
import dataclasses
from typing import Any, List

from PIL import Image
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from moviepy.editor import VideoFileClip

from PIL import Image
from decord import VideoReader, cpu # This is Terrible, if you have this line of import in front of torch, will cause model.to(device) to hang
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.easydict import EasyDict

IMAGE_TOKEN = "<image>"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from visualizer import Visualizer
import numpy as np
import matplotlib
import cv2
from matplotlib.colors import XKCD_COLORS

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()

class MultiModalConvStyle(Enum):
    """Different separator style."""
    MM_ALONE = 'mm_alone'
    MM_INTERLEAF = 'mm_inferleaf'

def dump_json(obj_serializable ,save_dir_path, json_file_name):
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, json_file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj_serializable, f, indent=4, ensure_ascii=False, )

def load_json(load_dir_path, json_file_name):
    load_path = os.path.join(load_dir_path, json_file_name)
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        obj_serializable = json.load(f)
    return obj_serializable



@dataclasses.dataclass
class Conversation(EasyDict):
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: List[str]
    mm_token: str
    
    mm_style: MultiModalConvStyle = MultiModalConvStyle.MM_INTERLEAF
    pre_query_prompt: str=None
    post_query_prompt: str=None
    answer_prompt: str=None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.sep, str):
            self.sep = [self.sep for _ in self.roles]

    def get_prompt(self):
        sep = [self.sep for _ in self.roles] if isinstance(self.sep, str) else self.sep  # if only one sep given, then both sep are the sames
        sep = dict(zip(self.roles, sep))
        ret = self.system + sep[self.roles[0]] if self.system != "" else ""
        for i, (role, message) in enumerate(self.messages):
            # if is last msg(the prompt for assistant), if answer prompt exists, no sep added
            if i+1 == len(self.messages):
                if role != self.roles[-1]: # last role is not the model
                    ret += role + message + sep[role] + self.roles[-1]
                else:
                    ret += role + message
            else:
                ret += role + message + sep[role]
        return ret
    # def get_prompt_multichoice(self):
    #     pass
    def user_query(self, query=None, pre_query_prompt=None, post_query_prompt=None, is_mm=False, num_mm_token=1):
        if post_query_prompt is not None:
            query = f"{query} {post_query_prompt}"

        if pre_query_prompt is not None:
            query = f"{pre_query_prompt} {query}"
        role = self.roles[0]
        # TODO: remove the num_mm_token and hack the self.mm_token outside
        if is_mm:
            mm_str = num_mm_token*self.mm_token[:-1] + self.mm_token[-1]
            if self.mm_style == MultiModalConvStyle.MM_ALONE:
                self._append_message(role, mm_str)
            elif self.mm_style == MultiModalConvStyle.MM_INTERLEAF:
                if self.mm_token not in query:
                    query = f'{mm_str} {query}'
        self._append_message(role, query)
    
    def assistant_response(self, response, pre_query_prompt=None, post_query_prompt=None):
        if post_query_prompt is not None:
            response = f"{response} {post_query_prompt}"

        if pre_query_prompt is not None:
            response = f"{post_query_prompt} {response}"

        role = self.roles[1]
        self._append_message(role, response)
    
    def _append_message(self, role, message):
        message = '' if message is None else message
        self.messages.append([role, message])

    def copy(self):
        return copy.deepcopy(self)

conv_video_chatgpt_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)


conv_plain_v1 = Conversation(
    system="",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token='<image>'
)

# Attention to the roles[0] "USER: " has a space!
conv_eval_vcg = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)

conv_eval_vcg_llavanext = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)

SYSTEM_MVBENCH="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
conv_eval_mvbench_llavanextvideo = Conversation(
    system=SYSTEM_MVBENCH,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<video>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)
conv_eval_mvbench = Conversation(
    system=SYSTEM_MVBENCH,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)
SYSTEM_VIDEOMME="Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
conv_eval_videomme = Conversation(
    system=SYSTEM_VIDEOMME,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)
conv_eval_mvbench_llavanext = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_eval_videoqabench = Conversation(
    system="",
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='\nAnswer: In the video,'
)

conv_eval_videoqa_llavanext = Conversation(
    system="<|im_start|>system\nAnswer the question.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='\nAnswer: In the video,'
)


SYSTEM_RECAPTION="""You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video recaptioning task.
The user will provide you with the video and maybe some extra noisy information to help you out. Make use of the information in a proper way to be competent for the recaption job
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""
conv_eval_recaption = Conversation(
    system=SYSTEM_RECAPTION,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_eval_recaption_llavanext = Conversation(
    system=SYSTEM_RECAPTION,
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_ALONE,
)


conv_templates = {
    "plain": conv_plain_v1,
    "eval_vcgbench": conv_eval_vcg,
    "eval_vcg_llavanext": conv_eval_vcg_llavanext,
    "eval_videomme": conv_eval_videomme,
    "eval_mvbench": conv_eval_mvbench,
    "eval_mvbench_llavanext": conv_eval_mvbench_llavanext,
    "eval_videoqabench": conv_eval_videoqabench,
    "eval_videoqa_llavanext": conv_eval_videoqa_llavanext,
    "eval_recaption": conv_eval_recaption,
    "eval_recaption_llavanext": conv_eval_recaption_llavanext,
    "eval_mvbench_llavanextvideo": conv_eval_mvbench_llavanextvideo
}


class EvalDataset(Dataset):

    def __init__(self, num_segments, test_ratio=None):
        super().__init__()
        self.num_segments = num_segments
        self.test_ratio = test_ratio
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_clip_gif,
            'frame': self.read_frame,
        }
        
    def __getitem__(self, index) -> Any:
        raise NotImplementedError('')
        
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        # BGR
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
                if len(images_group) == len(frame_indices):
                    break

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_clip_gif(self, video_path, bound=None, fps=25):
        gif = VideoFileClip(video_path)
        frames = gif.iter_frames()
        max_frame = gif.reader.nframes - 1
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(frames):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            # RGB
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        return images_group
    
    def read_npy(self, video_path, bound=None, fps=3):
        if os.path.isfile(video_path):
            video_path = os.path.splitext(video_path)[0] + '.npy'
        elif os.path.isdir(video_path):
            video_path = video_path + '.npy'
        else:
            raise ValueError(f"No such file or directory: {video_path}")
        
        video_path_npy = video_path.replace('DATAS/MVBench/video', 'DATAS/MVBench/sampled_frames')
        
        video = np.load(video_path_npy, allow_pickle=True)
        images_group = [image.astype(np.uint8) for image in video]

        # seg_path_npy = video_path.replace('DATAS/MVBench/video', 'DATAS/MVBench/sampled_segmentation_multi_frames')

        # if os.path.exists(seg_path_npy):
        #     segs_group = self.read_seg(seg_path_npy)
        #     images_group.extend(segs_group)
        # else:
        #     raise ValueError(f"No such file or directory: {seg_path_npy}")
        
        images_group = [Image.fromarray(image) for image in images_group]

        return images_group
    
    def draw_number_text(self, img_seq, seg_seq):
        # img_seq: t h w c
        # seg_seq: [h w c]*(t*k) (value: 0-255)
        seg_seq = np.stack(seg_seq, axis=0) # t*k h w c
        seg_seq = seg_seq / 255.0
        t, h, w, c = img_seq.shape
        seg_seq = seg_seq.reshape(t, -1, h, w, c)
        color_num = len(seg_seq[0])
        color_all = []
        for k, v in XKCD_COLORS.items():
            color_all.append(v)
        color_list = color_all[:color_num]

        img_list = []
        for t_idx in range(t):
            img = img_seq[t_idx]
            image_object = Visualizer(img)
            k, h, w, c = seg_seq[t_idx].shape
            for i in range(k):
                seg = seg_seq[t_idx][i,:,:,0]
                if seg.sum() / (h*w) < 0.005 or seg.sum() / (h*w) > 0.8:
                    continue
                img=image_object.draw_binary_mask_with_number(binary_mask=seg, text=str(i), anno_mode=['Mask', 'Mark'], \
                                                            alpha=0.1, color=color_list[i], edge_color=color_list[i], font_size=15)
            if not isinstance(img, np.ndarray):
                img = img.get_image()
            img_list.append(img)
            cv2.imwrite('masks/test_mask_{}.jpg'.format(str(t_idx)), img)
        img_list = np.stack(img_list, axis=0) # t h w c
        return img_list
    
    def read_seg(self, seg_path):
        segs = np.load(seg_path, allow_pickle=True).item()
        frame_indices = list(segs.keys())
        if len(frame_indices) == 0:
            return []
        
        frame_indices = sorted(frame_indices)

        # delete over-large segments
        # to_delete = list()
        # for frame_index in frame_indices:
        #     seg = segs[frame_index]
        #     for k, v in seg.items():
        #         if k in to_delete:
        #             continue
        #         h, w = v.shape[1:]
        #         seg_points = v.sum()
        #         ratio = seg_points / (h*w)
        #         if ratio > 0.8:
        #             to_delete.append(k)

        # for frame_index in frame_indices:
        #     seg = segs[frame_index]
        #     for key in to_delete:
        #         del seg[key]
        
        # Add in the group
        
        segs_group = list()
        for frame_index in frame_indices:
            seg = segs[frame_index]
            segs_group.append(seg)
            # segs_group.extend([v.transpose((1, 2, 0)).astype(np.uint8).repeat(3, axis=2)*255.0 for v in seg]) 
            # for k, v in seg.items():
            #     segs_group.append(v.transpose((1, 2, 0)).astype(np.uint8).repeat(3, axis=2)*255.0) # h w c
        
        segs_group = np.concatenate(segs_group, axis=0)*255.0
        segs_group = segs_group.transpose((0, 2, 3, 1)).repeat(3, axis=3).astype(np.uint8) # t h w 3

        return list(segs_group)

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # self.data_list = self.data_list[::200] # debug
        if self.test_ratio is None:
            self.data_list = self.data_list[rank::world_size]
        else:
            np.random.RandomState(42).shuffle(self.data_list)
            if isinstance(self.test_ratio, float):
                num_samples = int(len(self.data_list) * self.test_ratio)
            else:
                num_samples = int(self.test_ratio)
            self.data_list = self.data_list[rank:num_samples:world_size]


class ChatPllava:
    print_res=True
    do_sample=False
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def ask(self, text, conv: Conversation, system):
        conv.system = system
        conv.user_query(text, )
        return conv

    def answer(self, conv: Conversation, img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        torch.cuda.empty_cache()
        prompt = conv.get_prompt()
        if prompt.count(conv.mm_token) < len(img_list):
            diff_mm_num = len(img_list) - prompt.count(conv.mm_token)
            for i in range(diff_mm_num):
                conv.user_query("", is_mm=True)
            prompt = conv.get_prompt()
            
        inputs = self.processor(text=prompt, images=img_list, return_tensors="pt")
        if inputs['pixel_values'] is None:
            inputs.pop('pixel_values')
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_token = self.model.generate(**inputs, media_type='video',
                                        do_sample=self.do_sample,max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                        top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature,
                                        ) # dont need to long for the choice.
            output_text = self.processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if self.print_res:
            print('###PROMPT: ', prompt)
            print('###LM OUTPUT TEXT', output_text)
        # <|im_start|> encode and then decode would extend a space at folloing, this is insane...
        if conv.roles[-1] == "<|im_start|>assistant\n":
            split_tag = "<|im_start|> assistant\n"
        else:
            split_tag = conv.roles[-1]
        output_text = output_text.split(split_tag)[-1].rstrip(conv.sep[1])
        conv.assistant_response(output_text)
        return output_text, output_token.cpu().numpy(), conv
    
        
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, video_path, num_segments=8, return_msg=False):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)
        
        duration = len(vr) // vr.get_avg_fps()
        index = np.linspace(0, len(vr)-1, num=int(duration))
        buffer = vr.get_batch(index).asnumpy()
        # transform
        
        images_group = list()
        for frame in buffer:
            img = Image.fromarray(frame)
            images_group.append(img)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return images_group, msg
        else:
            return images_group

    def upload_video(self, image, conv: Conversation, img_list: list[list], num_segments=None):
        num_segments = self.model.config.num_frames if num_segments is None else num_segments 
        if isinstance(image, str):  # is a image path
            vid, msg = self.load_video(image, num_segments=num_segments, return_msg=True)
        else:
            raise NotImplementedError
        print("Input video shape:", len(vid), *vid[0].size)
        img_list.append(vid)
        conv.user_query("", is_mm=True)
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list, conv
    
    def upload_img(self, image, conv, img_list):
        assert False
        img = image#Image.open(image)#.convert('RGB')
        transform = T.Compose(
            [
                T.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        img = transform(img).unsqueeze(0).unsqueeze(0).cuda()
        image_emb, _ = self.model.encode_img(img, "Observe the image and answer the question.")
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0],
            f"<Image><ImageHere></Image>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg,img_list, conv

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
