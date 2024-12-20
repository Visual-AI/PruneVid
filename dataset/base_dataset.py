import logging
import os
import json
import random
from torch.utils.data import Dataset
import time
from dataset.utils import load_image_from_path
from mmflow.apis import init_model
import torch
import string
import numpy as np
import cv2
try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video", "only_video"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None
        # self.optical_model = self.get_optical_flow_model()

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')
    
    def generate_random_string(self, length=6):
        characters = string.ascii_letters + string.digits
        
        # 随机选择字符并组成字符串
        random_string = ''.join(random.choice(characters) for _ in range(length))
        
        return random_string

    def extract_flow_raft(self, frames, model):
        # t c h w
        t, c, ori_h, ori_w = frames.shape

        assert c == 3

        with torch.no_grad():
            # name = self.generate_random_string()
            # frames_npy = frames.detach().cpu().numpy()
            # np.save(f'frame/frame_{name}.npy', frames_npy)

            frames = (frames - 127.5) / 127.5

            frames = frames.cuda()

            feat = model.encoder(frames) # t c h w

            cxt_feat = model.context(frames)

            h_feat, cxt_feat = torch.split(
                cxt_feat, [model.h_channels, model.cxt_channels], dim=1)
            h_feat = torch.tanh(h_feat)
            cxt_feat = torch.relu(cxt_feat)
            
            t, c, h, w = feat.shape
            feat = feat.view(t, c, h, w)
            pre_feat = feat.clone()
            next_feat = torch.cat([feat[1:], feat[-2].unsqueeze(0)], dim=0) # t c h w
            next_feat = next_feat.contiguous().view(-1, c, h, w)
            
            flow = torch.zeros((next_feat.shape[0], 2, h, w), device=next_feat.device)
            
            upflow_preds = model.decoder(pre_feat, next_feat, flow, h_feat, cxt_feat)

            flow_result = upflow_preds[-1].cpu()

            print(flow_result.shape)

            # flow_result_npy = flow_result.detach().cpu().numpy()
            # np.save(f'flow/flow_{name}.npy', flow_result_npy)

        return flow_result

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno["image"] = os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path, clip_transform=self.clip_transform)
        else:
            return self.load_and_transform_media_data_video(index, data_path, clip_transform=self.clip_transform)

    def load_and_transform_media_data_image(self, index, data_path, clip_transform=False):
        image = load_image_from_path(data_path, client=self.client)
        if not clip_transform:
            image = self.transform(image)
        return image, index
    
    def get_optical_flow_model(device):
        # config = '~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
        # checkpoint = '~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
        config = '~/.cache/mim/raft_8x2_100k_mixed_368x768.py'
        checkpoint = '~/.cache/mim/raft_8x2_100k_mixed_368x768.pth'
        model = init_model(config, checkpoint)
        return model

    def load_and_transform_media_data_video(self, index, data_path, return_fps=False, clip=None, clip_transform=False):
        for _ in range(self.num_tries):
            flow = None
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                if "webvid" in data_path:
                    # hdfs_dir="hdfs://harunava/home/byte_ailab_us_cvg/user/weimin.wang/videogen_data/webvid_data/10M_full_train"
                    # video_name = os.path.basename(data_path)
                    # video_id, extension = os.path.splitext(video_name)
                    # ind_file = os.path.join(hdfs_dir, self.keys_indexfile[video_id])
                    # frames, frame_indices, fps = self.video_reader(ind_file, video_id, self.num_frames, self.sample_type, 
                    #                            max_num_frames=max_num_frames, client=self.client, clip=clip)
                    frames, frame_indices, fps = self.video_reader(
                        data_path, self.num_frames, self.sample_type, 
                        max_num_frames=max_num_frames, client=self.client, clip=clip
                    )
                else:
                    frames, frame_indices, fps = self.video_reader(
                        data_path, self.num_frames, self.sample_type, 
                        max_num_frames=max_num_frames, client=self.client, clip=clip
                    )

                # flow = self.extract_flow_raft(frames, self.optical_model)
                
            except Exception as e:
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                index = random.randint(0, len(self) - 1)
                ann = self.get_anno(index)
                data_path = ann["image"]
                continue
            # shared aug for video frames
            if not clip_transform:
                frames = self.transform(frames)
            # if flow is not None and not clip_transform:
            #     frames = torch.cat([frames, flow], dim=0)
            if return_fps:
                sec = [str(round(f / fps, 1)) for f in frame_indices]
                return frames, index, sec
            else:
                return frames, index
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )
