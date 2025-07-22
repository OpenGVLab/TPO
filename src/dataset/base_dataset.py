import logging
import os
import random
from petrel_client.client import Client
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from .utils import load_audio_from_path, load_image_from_path
from .av_utils import lazy_load_s3video, load_audio_av, load_full_audio_av
from .hd_utils import HD_transform_padding, HD_transform_no_padding
from .dec_utils import dec_transform, box_xyxy_expand2square,  norm_point_xyxy
logger = logging.getLogger(__name__)
import torch

class BaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["audio", "image", "video", "audio_video"]
        self.data_root = None
        self.data_root_prefix = ""
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        ) 
        self.transform = None
        self.audio_reader_type = None
        self.audio_sample_rate = None
        self.max_audio_length = None
        self.video_reader = None
        self.num_tries = None
        self.client = Client('~/petreloss.conf')
        self.trimmed30 = False

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index): # NOTE used for most ret_dataset
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            if self.media_type == "audio":
                anno["audio"] = self.data_root_prefix + os.path.join(self.data_root, anno["audio"])
            else:
                anno["image"] = self.data_root_prefix + os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path, return_raw_size = False):
        try:
            if self.media_type == "image":
                return self.load_and_transform_media_data_image(index, data_path, return_raw_size)
            elif self.media_type == "audio":
                return self.load_and_transform_media_data_audio(index, data_path)
            elif self.media_type in ["video", "interleaved_video"]:
                return self.load_and_transform_media_data_video(index, data_path)
            elif self.media_type == "audio_video":
                return self.load_and_transform_media_data_audio_video(index, data_path)
            else:
                raise NotImplementedError(self.media_type)
        except Exception as e:
            logger.info(f"Basedataset: Something wrong when read {data_path}")
            raise e

    def load_and_transform_media_data_copyimage(self, index, data_path, return_raw_size = False):
        num_frames = self.num_frames if hasattr(self, "num_frames") else -1
        image = load_image_from_path(data_path, client=self.client)
        # print(f'image_size:{image.shape, max_num_frames}')
        image = torch.cat([image] * num_frames, dim=0)
        image_size = image.shape[-2:]  # h, w
        image = self.transform(image)
        if return_raw_size:
            return image, index, image_size
        else:
            return image, index

    def load_and_transform_media_data_image(self, index, data_path, return_raw_size = False, use_dec = False):
        if type(data_path) is dict:
            image = load_image_from_path(data_path["image"], client=self.client)
            if "crop_bbox" in data_path.keys():
                bbox = data_path["crop_bbox"]
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = image[:, :, y0:y1, x0:x1]
            if use_dec:
                image = dec_transform(image)
            image = self.transform(image)
        else:
            image = load_image_from_path(data_path, client=self.client)
            image_size = image.shape[-2:]  # h, w
            if use_dec:
                image = dec_transform(image)
            image = self.transform(image)
        
        if return_raw_size:
            return image, index, image_size
        else:
            return image, index

    def load_and_transform_media_data_audio(self, index, data_path):
        audio = load_audio_from_path(data_path, self.client, self.audio_sample_rate, self.audio_reader_type, self.max_audio_length*self.audio_sample_rate)
        if self.transform is not None:
            audio = self.transform(audio)
        return audio, index
    
    def load_and_transform_media_data_video(self, index, data_path, return_fps=False, clip=None, use_dec=False):
        if type(data_path) is dict:
            if data_path['read_clip_from_video']:
                if self.trimmed30:
                    raise NotImplementedError("lazy_load_s3video 还没实现trimmed30")
                frames = lazy_load_s3video(data_path['video'], self.num_frames, data_path['video_start_frame'], data_path['video_end_frame'], self.client)
            else:
                raise NotImplementedError(data_path)
        else:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            # print(f'reading from {data_path}')
            # frames, frame_indices, fps = self.video_reader(
            #     data_path, self.num_frames, self.sample_type, 
            #     max_num_frames=max_num_frames, client=self.client, clip=clip
            # )
            frames, frame_indices, fps, vlen = self.video_reader(
                data_path, self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client, clip=clip
            )
        # NOTE shared aug for video frames
        frames = self.transform(frames)
        # print(frames.shape)
        
        # if dynamic_res: # Force using hd
        if False: # 拒绝hd!
            local_size = 224
            hd_num = 6
            padding = False
            if padding:
                frames = HD_transform_padding(frames.float(), image_size=local_size, hd_num=hd_num)
            else:
                frames = HD_transform_no_padding(frames.float(), image_size=local_size, hd_num=hd_num)
        if return_fps:
            if fps == None:
                sec = None
            else:
                sec = [str(round(f / fps, 1)) for f in frame_indices]
            # return frames, index, sec
            return frames, sec, vlen, frame_indices, fps
        else:
            return frames, index


    def load_and_transform_media_data_video_sam2(self, data_path,  image = False, frame_num = 1):
        try:
            if image:
                all_frames = load_image_from_path(data_path, self.client)
            else:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, fps, vlen, all_frames = self.video_reader(
                    data_path, self.num_frames, self.sample_type, 
                    max_num_frames=max_num_frames, client=self.client, clip=None, all_frames=True
                )
            from torchvision import transforms
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
            normalize = transforms.Normalize(mean, std)
            image_size = 1024
            sam2_transform = transforms.Compose(
                [
                    type_transform,
                    transforms.Resize((image_size, image_size)),
                    normalize,
                ]
            )
            if not image:
                # all_frames = all_frames[:16]
                if frame_num > 0:
                    all_frames = all_frames[:frame_num]
                else:
                    all_frames = all_frames
            all_frames = sam2_transform(all_frames)
            return all_frames
        except Exception as e:
            logger.info(f"Sam read: Something wrong when read {data_path}")
            raise e




    def load_and_transform_media_data_audio_video(self, index, data_path):

        video_transform = self.transform # audio don't have transform

        # if self.trimmed30:
        #     logger.warn("开了trimmed30, 测试的时候需要考虑一下trimmed30要不要考虑音频")
        
        if data_path['read_clip_from_video']:
            if self.trimmed30:
                raise NotImplementedError("lazy_load_s3video 还没实现trimmed30")
            frames = lazy_load_s3video(data_path['video'], self.num_frames, data_path['video_start_frame'], data_path['video_end_frame'], self.client)
        else:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            frames, _, _ = self.video_reader(
                data_path["video"], self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client,
                trimmed30=self.trimmed30
            )

        if data_path["read_audio_from_video"]:
            if data_path['read_clip_from_video']:
                audio = load_audio_av(data_path['video'], data_path['video_start_frame'], data_path['video_end_frame'], self.audio_sample_rate, self.max_audio_length, self.client)
            else: # read audio from a clip directly
                audio = load_full_audio_av(data_path['video'], self.audio_sample_rate, self.max_audio_length, self.client)
        else:
            audio = load_audio_from_path(data_path['audio'], self.client, self.audio_sample_rate, self.audio_reader_type, self.max_audio_length*self.audio_sample_rate)
        

        # NOTE shared aug for video frames
        frames = video_transform(frames)
        return [audio, frames], index

