num_gpu = 8
import os
print(os.environ['SLURM_PROCID'])
print(os.environ['CUDA_VISIBLE_DEVICES'] )
# os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['SLURM_PROCID'])%num_gpu)
gpu_id = int(os.environ['SLURM_PROCID'])%num_gpu
device = f'cuda:{gpu_id}'
procid = int(os.environ['SLURM_PROCID']) + 1



from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "/mnt/petrelfs/share_data/yanziang/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
# for _, param in predictor.named_parameters():
#     # param.requires_grad = False
#     param = param.to(dtype = self.lm.dtype)
sam = predictor
# import pdb;pdb.set_trace()
# predictor = predictor.to(self.language_model.device)
# freeze_module(self.sam)
from PIL import Image
from petrel_client.client import Client
c = Client()
from torchvision.transforms import PILToTensor
def read_frames_decord_all(
        video_path, client=None, ret_all_frames=False
    ):
    # print("Reading decord")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)

    vlen = len(video_reader)
    # print(video_reader[0].shape)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    frame_indices = [i for i in range(0, vlen, 1)]
    all_frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    # import pdb;pdb.set_trace()
    all_frames = torch.tensor(all_frames.asnumpy().tolist())
    all_frames = all_frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    if ret_all_frames:
        return all_frames, frame_indices, float(fps), vlen
    else:
        return all_frames[0], frame_indices, float(fps), vlen



def load_image_from_path(image_path, client):
    if "s3://" in image_path:
        value = client.get(image_path)
        if value is None:
            img_bytes = np.frombuffer(value, dtype=np.uint8)
            buff = io.BytesIO(img_bytes)
            image = Image.open(buff).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image


def load_and_transform_media_data_video_sam2(data_path,  client, image = False, frame_num = 1 ):
    try:
        if image:
            all_frames = load_image_from_path(data_path, client)
        else:
            all_frames, frame_indices, fps, vlen = read_frames_decord_all(
                data_path, client, ret_all_frames=True
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
        # logger.info(f"Sam read: Something wrong when read {data_path}")
        raise e
    
import json
from tqdm import tqdm
import numpy as np
import jsonlines
from decord import VideoReader, cpu
import io
import torch
data = []
def convert_json_to_jsonl(json_file_path, out_path):
    """
    读取 JSON 文件并将其写入 JSON Lines 文件。

    :param json_file_path: 输入的 JSON 文件路径
    :param out_path: 输出的 JSON Lines 文件路径
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        fin_data = []
        for i, dd in tqdm(enumerate(data)):
            if i % procid == 0:
                tt = dd
                fin_data.append(tt)
                # tt['gt_embs'] = 
                video_path = 'pnorm2:s3://' + dd['video']
                if 's3://' in video_path:
                    video_bytes = c.get(video_path)
                    video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
                else:
                    video_reader = VideoReader(video_path, num_threads=1)
            
                vlen = len(video_reader)
                orig_height, orig_width = video_reader[0].shape[0], video_reader[0].shape[1]
                print(orig_height, orig_width)
                sam_gt_bbox = dd['bbox']
                sam_gt_bbox = [ sam_gt_bbox[1] / orig_height, sam_gt_bbox[0] / orig_width, sam_gt_bbox[3] / orig_height, sam_gt_bbox[2] / orig_width ]
                sam_images = load_and_transform_media_data_video_sam2('pnorm2:s3://' + dd['video'], c, False, 1)
                print(sam_gt_bbox)
                sam_images = sam_images.unsqueeze(0)
                print(sam_images.shape)
                inference_state = sam.init_state_images(sam_images, orig_height, orig_width)
                print(orig_height, orig_width)
                # import pdb;pdb.set_trace()
                # sam_gt_bbox = [_.to(torch.bfloat16) for _ in sam_gt_bbox]
                sam_gt_embs = sam.get_prompt_embeding(inference_state, None, None, False, sam_gt_bbox, device = device) 
                
                torch.save(sam_gt_embs.cpu(), dd['name'])
        #         tt['sam_gt_embs'] = f'/mnt/petrelfs/share_data/yanziang/internvl/data/mevis_feature/{i}.pt'
                print(tt)
        #         # break
        # print(len(fin_data))
        # with open(out_path, 'w', encoding='utf-8') as jsonl_file:
        #     if isinstance(fin_data, list):
        #         for item in fin_data:
        #             jsonl_file.write(json.dumps(item) + '\n')
        #     else:
        #         jsonl_file.write(json.dumps(fin_data) + '\n')

# 示例用法

convert_json_to_jsonl('/mnt/petrelfs/share_data/yanziang/internvl/data/mevis_feature/no_feature.json', f'/mnt/petrelfs/share_data/yanziang/internvl/data/mevis_embs_{procid}.jsonl')