"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""
import random
import io
import os
import av
import cv2
import decord
from PIL.Image import Image
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
decord.bridge.set_bridge("torch")
import logging
logger = logging.getLogger(__name__)

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)


def get_frame_indices_by_fps():
    pass


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    # num_frames = int(num_frames)
    # vlen = int(vlen)
    # print(num_frames, vlen, sample)
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        # print(intervals)
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        # print(ranges)
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        # print(frame_indices)
        
        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
        
        # print(frame_indices)
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_av(video_path, num_frames, sample='rand', client=None, fix_start=None, max_num_frames=-1, clip=None):
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_bytes = io.BytesIO(video_bytes)
        video_bytes.seek(0)
        reader = av.open(video_bytes)
    else:
        reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, 25, duration


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None,
    ):
    # print("Reading gif")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    # print('Finish reading gif')
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, 25., len(frames) # for tgif




def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None, all_frames=False
    ):
    # print("Reading decord")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    # print("Finish Reading decord")
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if clip:
        start, end = clip
        end = min(end, duration)
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    # if frame_indices[-1] > len(video_reader):
    #     if clip:
    #         logger.info(f'{start}, {end}, {start_index}')
    #         #2.1 26 62
    #     logger.info(f' {num_frames}, {vlen}, {sample}, {fix_start}, {fps}, {max_num_frames}')
    #     # 16 716 rand None 29.97 -1
    #     logger.info(f'len video:{vlen}, {len(video_reader)},frame_indices:{frame_indices}')
    #     #716 750 74...772
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    if all_frames:
        frame_indices = [i for i in range(0, vlen, 1)]
        all_frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        all_frames = all_frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        return frames, frame_indices, float(fps), vlen, all_frames
    else:
        return frames, frame_indices, float(fps), vlen

# def read_frames_decord(
#         video_path, num_frames, sample='rand', fix_start=None, 
#         max_num_frames=-1, client=None, clip=None
#     ):
#     # print("Reading decord")
#     if 's3://' in video_path:
#         video_bytes = client.get(video_path)
#         video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
#     else:
#         video_reader = VideoReader(video_path, num_threads=1)
#     # print("Finish Reading decord")
#     vlen = len(video_reader)
#     fps = video_reader.get_avg_fps()
#     duration = vlen / float(fps)

#     if clip:
#         start, end = clip
#         duration = end - start
#         vlen = int(duration * fps)
#         start_index = int(start * fps)

#     frame_indices = get_frame_indices(
#         num_frames, vlen, sample=sample, fix_start=fix_start,
#         input_fps=fps, max_num_frames=max_num_frames
#     )
#     if clip:
#         frame_indices = [f + start_index for f in frame_indices]

#     frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
#     frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
#     return frames, frame_indices, float(fps)



def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None, all_frames = False
    ):
    img_list=[]
    if "s3://" in video_path:
        for path in client.list(video_path):
            # if path.startswith('img'):
            img_list.append(path)
    else:
        for path in os.listdir(video_path):
            # if path.startswith('img'):
            img_list.append(path)

    vlen = len(img_list)

    def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=3, max_frame=9999):
        start_idx = max(1, round(start_sec * fps))
        end_idx = min(round(end_sec * fps), max_frame)
        seg_size = float(end_idx - start_idx) / (num_segments - 1)
        offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
        return offsets
    
    if clip is not None:
        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, max_frame=vlen)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            max_num_frames=max_num_frames
        )
    all_imgs = []
    if all_frames:
        all_frame_indices = [i for i in range(len(img_list))]
            
        for idx in all_frame_indices:
            frame_fname = os.path.join(video_path, img_list[idx])
            if "s3://" in video_path:
                img_bytes = client.get(frame_fname)
            else:
                with open(frame_fname, 'rb') as f:
                    img_bytes = f.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            all_imgs.append(img)
            


    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    frames = torch.tensor(np.array(imgs), dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    del imgs, img_list
    if all_frames:
        all_frames = torch.tensor(np.array(all_imgs), dtype=torch.uint8).permute(0, 3, 1, 2)
        return frames, frame_indices, 25, 25, all_frames
    return frames, frame_indices, 25, 25

def read_frames_pt(video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None):
    # print("Reading decord")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video = torch.load(io.BytesIO(video_bytes), map_location='cpu')
    else:
        video = torch.load(io.BytesIO(video_bytes), map_location='cpu')
    
    _, T, C, H, W = video.shape
    
    indices = np.tile(np.random.choice(video.shape[1], num_frames, replace=False), 1).reshape(1, num_frames)
        
    # Create a boolean mask
    mask = np.zeros((1, video.shape[1]), dtype=bool)
        
    # Use advanced indexing to set the selected indices to True
    np.put_along_axis(mask, indices, True, axis=1)
    
    mask = torch.from_numpy(mask)
    
    video = video[mask].view(num_frames, C, H, W)
    
    video_indices = indices[0]
    video_indices.sort()
    
    return video, video_indices, 25, 25


VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'img': read_frames_img,
    'pt': read_frames_pt,
}


if __name__ == '__main__':
    from petrel_client.client import Client
    client = Client('~/petreloss_lxh.conf')
# 'sssd:s3://video_pub/ssv2_video/38262.webm'
    frames = read_frames_decord('sssd:s3://video_pub/ssv2_video/38262.webm', #'pssd:s3://ssv2_video/38262.webm',
                                1, client=client)
    
    print(len(frames))