import logging
import os
import json
import random
import io
import torch
import numpy as np
import random
import pycocotools.mask as mask_util
from .utils import pre_text
from .base_dataset import BaseDataset
from .video_utils import VIDEO_READER_FUNCS
from ..share_utils.serialize import get_local_rank, TorchShmSerializedList
from .dec_utils import dec_transform, box_xyxy_expand2square,  norm_box_xyxy
from ..share_utils.special_token import need_template_list, load_image_list, load_video_list, EXPR_PLACEHOLDER, IMG_TOKEN, VID_TOKEN, BOX_START, TIME_START, ATBOXES_PLACEHOLDER, ATTEMP_PLACEHOLDER, ATTRACK_PLACEHOLDER, TRACK_START_BOX, TRACK_PLACEHOLDER
# from ..share_utils.grounding_utils import get_cg_item
from os.path import join, exists
from ..model.cgdetr_main.utils.basic_utils import load_jsonl, l2_normalize_np_array
from ..model.cgdetr_main.utils.tensor_utils import pad_sequences_1d
from ..model.cgdetr_main.cg_detr.span_utils import span_xx_to_cxw
'''
example from /mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json:
{'long_video_id': 'aOqaEiyqliY', 'interleaved_list': [{'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_0_70.mp4', 'caption': "it's getting very", 'video_start_frame': 0}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_70_372.mp4', 'caption': 'real now the world cup squad have arrived in Brazil Roy Hodgson side landed in rio de janeiro on sunday after a flight from miami whether it be based for the past week the squad', 'video_start_frame': 70}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_372_625.mp4', 'caption': 'looks relaxed getting off the plane wearing polo shirts reflecting a change of mood from four years ago what', 'video_start_frame': 372}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_625_847.mp4', 'caption': "if the last off was Wayne Rooney there was real pressure on him to perform here if he's ever to be considered one of England's all-time great the", 'video_start_frame': 625}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_847_996.mp4', 'caption': 'only injury concern is Alex oxlade-chamberlain who has a knee ligament problem the', 'video_start_frame': 847}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_996_1160.mp4', 'caption': 'squad have another flight to catch before their opening match in Group D as they face Italy in Manaus which is', 'video_start_frame': 996}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1160_1261.mp4', 'caption': 'four hours on a plane northwest of Rio England then', 'video_start_frame': 1160}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1261_1525.mp4', 'caption': 'face Uruguay in são Paulo before taking on costa rica in Belo Horizonte', 'video_start_frame': 1261}]}
'''
logger = logging.getLogger(__name__)

class ITTrainDataset_mistral_eval(BaseDataset):

    def __init__(
        self, ann_file, transform, tokenizer,
        system="", role=("[INST]", ""),
        random_shuffle=True,
        video_reader_type='decord',
        sample_type='rand',
        num_frames=4,
        num_tries=10,
        seed=None,
        cgdetr_config = None
    ):
        super().__init__()
        self.media_type = ann_file.get('media_type', 'image')
        # media_type: VQ,REC ... (数据集的格式)  
        # image, video, grounding, tracking, DEC   
        # 
        template_root = ann_file.get('template_root', '/mnt/petrelfs/yanziang/trackingllm/videoagent/template/template/')
        
        self.media_type = ann_file.get('media_type', 'image')  
        
        self.label_file, self.data_root = ann_file.anno_path, ann_file.data_root

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        if self.media_type in load_video_list:
            self.num_frames = num_frames
            self.video_reader_type = ann_file.get('video_reader_type', 'decord')  
            self.video_reader = VIDEO_READER_FUNCS[self.video_reader_type]
            self.sample_type = sample_type
            self.num_tries = num_tries
        
        if self.media_type == 'TVG':
            self.dset_name = cgdetr_config['dset_name']
            self.data_path = self.label_file
            # self.data_ratio = data_ratio
            # self.v_feat_dirs1 = cgdetr_config['v_feat_dirs'] if isinstance(cgdetr_config['v_feat_dirs'], list) else [cgdetr_config['v_feat_dirs']]
            # self.q_feat_dir1 = cgdetr_config['q_feat_dir']
            v_feat_dirs = ann_file.get('v_feat_dirs', None)
            self.v_feat_dirs = v_feat_dirs if isinstance(v_feat_dirs, list) else [v_feat_dirs]
            self.q_feat_dir = ann_file.get('q_feat_dir', None)
            self.q_feat_type = cgdetr_config['q_feat_type']
            self.max_q_l = cgdetr_config["max_q_l"]
            self.max_v_l = cgdetr_config["max_v_l"]
            self.ctx_mode = cgdetr_config['ctx_mode']
            self.use_tef = "tef" in self.ctx_mode
            self.use_video = "video" in self.ctx_mode
            self.normalize_t = True
            self.normalize_v = True
            self.load_labels = True
            self.clip_len = cgdetr_config['clip_len']
            self.max_windows = cgdetr_config['max_windows']  # maximum number of windows to use as labels
            self.span_loss_type = cgdetr_config['span_loss_type']
            self.txt_drop_ratio = 0
            self.use_feature = cgdetr_config['use_feature']
            self.duration = 0
            # if "val" in data_path or "test" in data_path:
            #     assert txt_drop_ratio == 0
            # checks
            assert self.q_feat_type in self.Q_FEAT_TYPES
            # data
            self.use_glove = False

        self.tokenizer = tokenizer
        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = ""
        self.end_signal = ""
        self.system = system
        self.role = ("","")
        self.random_shuffle = random_shuffle
        self.max_length = 1024
        # instruction location and number
        if self.media_type in need_template_list:
            template_path = os.path.join(template_root, self.media_type + '.json')
            with open(template_path, 'r') as f:
                self.templates = json.load(f)
        self.rng = np.random.default_rng(seed)
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def __len__(self):
        return self.num_examples

    def get_template(self):
        return self.rng.choice(self.templates)


    def get_anno(self, index):
        if self.media_type in ['image', 'video']:
            filename = self.anno[index][self.media_type]
            qa = self.anno[index]["QA"]
            if "start" in self.anno[index] and "end" in self.anno[index]:
                anno = {
                    "image": os.path.join(self.data_root, filename), "qa": qa,
                    "start": self.anno[index]["start"], "end": self.anno[index]["end"],
                }
            else:
                anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
            return anno
        elif self.media_type == 'REC':
            item = self.anno[index]
            img_path = item['img_path']
            if 'expression' in item:
                # raise NotImplementedError
                expr = item['expression']
            elif 'expressions' in item:
                expr = item['expressions'][0]
            else:
                raise NotImplementedError
            bbox = item['bbox']
            question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

            ret = {
                'image': os.path.join(self.data_root, img_path),
                'target': {
                    'boxes': [bbox],
                    'boxes_seq': [[0]]
                },
                'qa': [
                    {
                        'q': question,
                        'a': f'Answer: {ATBOXES_PLACEHOLDER} .',
                    }
                ]
            }
            return ret
        elif self.media_type == 'flickr':
            item = self.anno[index]
            img_path = item['image_id'] + '.jpg'
            bbox = item['boxes']
            bbox = [[float(x) for x in tt] for tt in bbox] 
            raw_boxes = bbox
            boxes_seq = item['boxes_seq']
            question = self.get_template()
            answer = item['sentence']
            answer = answer.replace('<ph_st>', '')

            answer = answer.replace('<ph_ed>', ATBOXES_PLACEHOLDER)
            answer = self.replace_muti_bbox(answer, boxes_seq)
            ret = {
                'image': os.path.join(self.data_root, img_path),
                'target': {
                    'raw_boxes': raw_boxes, 
                    'boxes': bbox,
                    'boxes_seq': boxes_seq
                },
                'qa': [
                    {
                        'q': question,
                        'a': answer,
                    }
                ]
            }
            return ret
        elif self.media_type == 'TVG':
            filename = self.anno[index]['video']
            qa = self.anno[index]["QA"]
            new_qa = []
            for tt in qa:
                new_qa.append({'q':tt['q'], 
                               'a': "The given query happens in " + ATTEMP_PLACEHOLDER + " secondes."})
            ret = {
                "image": os.path.join(self.data_root, filename), 
                "qa": new_qa,
                # "start_sec" : self.anno[index]["start_sec"],
                # "end_sec": self.anno[index]["end_sec"],
            }
            return ret
        elif self.media_type == 'tracking':
            video_path = self.anno[index]['video_path']
            # logger.info(f'video_path:{video_path}')
            question = self.get_template()
            qa = [{
                'q': question,
                'a': f'Answer: {ATTRACK_PLACEHOLDER} .',
            }]
            gt_masks = eval(self.client.get(self.anno[index]['seg_path']).decode())
            gt_masks = eval(gt_masks["gt_mask"])
            # print(f'gt_masks:{gt_masks["gt_masks"]}')
            ret = {
                'image': os.path.join(self.data_root, video_path),
                'target':  self.anno[index]['bbox'],
                'qa': qa,
                'size': gt_masks[0]['size'], #h, w
                # 'gt_masks': (mask_util.decode(gt_masks)>0)[:,: ,:16] 
                'mask_path' : self.anno[index]['seg_path']
            }
            return ret
        elif self.media_type == 'tracking2':
            video_path = self.anno[index]['video_path']
            question = self.get_template()
            qa = [{
                'q': question.replace('<obj>', self.anno[index]['exp']),
                'a': f'Answer: {ATTRACK_PLACEHOLDER} .',
            }]
            mask_path = None
            if 'mask' in self.anno[index]:
                mask_path = self.anno[index]['mask']
                gt_masks = self.client.get(self.anno[index]['mask'])
                if 'gt_masks' in eval(gt_masks.decode()):
                    gt_masks = eval(gt_masks.decode())["gt_masks"]
                else:
                    gt_masks = eval(gt_masks.decode())["gt_mask"] 
                
                if gt_masks[0] is None:
                    raise IndexError('None masks')
                gt_masks = [gt_masks[0]]
                size = gt_masks[0]['size']
                decode_masks = (mask_util.decode(gt_masks)>0)
                decode_masks = np.array(decode_masks)
                decode_masks = torch.tensor(decode_masks, dtype=torch.bfloat16)
                bbox = self.find_boundaries_torch(decode_masks[:,:,:1].squeeze(2).cpu())
                # print(bbox)
            
            if 'bbox' in self.anno[index]:
                bbox = self.anno[index]['bbox']
                size = self.anno[index]['size_hw']
            ret = {
                'image': video_path,
                'qa': qa,
                'size': size, #h, w
                'target': bbox,
                'mask_path' : mask_path
                # 'gt_masks': (mask_util.decode(gt_masks)>0)[:,: ,:16] 
            }
            return ret
            # raise NotImplementedError('tracking')
        elif self.media_type == 'tracking3':
            item = self.anno[index]
            img_path = os.path.join(self.data_root, item['img_path'])
            if 'expression' in item:
                expr = item['expression']
            elif 'expressions' in item:
                expr = item['expressions'][0]
            else:
                raise NotImplementedError
            bbox = item['bbox']
            question = self.get_template()
            qa = [{
                'q': question.replace('<obj>', expr),
                'a': f'Answer: {ATTRACK_PLACEHOLDER} .',
            }]

            ret = {
                'image': img_path,
                'qa': qa,
                'target': bbox
                # 'gt_masks': (mask_util.decode(gt_masks)>0)[:,: ,:16] 
            }
            return ret
        elif self.media_type == 'tracking4':
            item = self.anno[index]
            img_path = os.path.join(self.data_root, item['video_path'])
            
            bbox = item['bbox']
            question = self.get_template()
            qa = [{
                'q': question,
                'a': f'Answer: {ATTRACK_PLACEHOLDER} .',
            }]

            ret = {
                'image': img_path,
                'size' : self.anno[index]['size_hw'],
                'qa': qa,
                'target': bbox
                # 'gt_masks': (mask_util.decode(gt_masks)>0)[:,: ,:16] 
            }
            return ret
        else:
            raise NotImplementedError('unknown')

    def process_bbox(self, boxes, raw_size):
        ret = []
        for box in boxes:
            # box = [box[0] / raw_size[1] * 224, box[1] / raw_size[0] * 224 ,box[2] / raw_size[1] * 224, box[3] / raw_size[0] * 224 ]
            box = box_xyxy_expand2square(box, raw_size[1], raw_size[0])
            box = norm_box_xyxy(box, max(raw_size[1], raw_size[0]), max(raw_size[0], raw_size[1]))
            ret.append(box)
        return ret

    def replace_muti_bbox(self, answer, box_seq):
        start_index = 0
        replacements = []
        
        while True:
            start_index = answer.find(ATBOXES_PLACEHOLDER, start_index)
            if start_index == -1 or start_index > len(answer) or len(replacements) >= len(box_seq):
                break
            index = len(replacements)
            num_boxes = len(box_seq[index])
            answer = answer[:start_index] + ATBOXES_PLACEHOLDER * ( num_boxes - 1 ) + answer[start_index:]  
            start_index = start_index + len(ATBOXES_PLACEHOLDER * ( num_boxes - 1 ))
            replacements.append(index)

            
        return answer

    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"]

        input_ids, attention_masks, labels = [], [], []
        
        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        conversation += (
            "[INST]" + " "
        )
        
        if self.media_type in load_image_list:
            conversation += "<Image>" + IMG_TOKEN + "</Image>"
        elif self.media_type in load_video_list:
            conversation += "<Video>" + VID_TOKEN + "</Video>"
        else:
            logger.info('No image or video. \n')

        conversation += (
            msg.rstrip() + "[/INST]"
        )
        total_len = 0
        indexs = []
        tokenized = self.tokenizer.build_input_ids(
            text=[conversation],
            max_length=self.max_length,
            add_special_tokens=False,
            truncation=False,
            require_image = (self.media_type in load_image_list),
            require_video = (self.media_type in load_video_list),
            padding=False,
            return_tensors='pt'
        )
        if self.media_type in load_image_list:
            indexs.append(tokenized['image_index'])
        elif self.media_type in load_video_list:
            indexs.append(tokenized['video_index'])
        # logger.info(f'video_index:{indexs}')
        input_ids.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])
        labels.append(torch.ones_like(tokenized['input_ids']) * -100)
        total_len += tokenized['input_ids'].shape[0]
        gtext_input = conversation
        
        for sentence in qa:
            if total_len >= self.max_length:
                break
            q = sentence["q"]
            a = sentence["a"]
            if q != "":
                conversation = (" " + "[INST]" + " " + q + " " + "[/INST]")
            else:
                # no question, often in caption dataset
                conversation = ""
            conversation += (" ")
            tokenized = self.tokenizer.build_input_ids(
                text=[conversation],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )
            indexs.append(torch.zeros_like(tokenized['input_ids']).to(torch.bool))
            input_ids.append(tokenized['input_ids'])
            attention_masks.append(tokenized['attention_mask'])
            labels.append(torch.ones_like(tokenized['input_ids']) * -100)
            total_len += tokenized['input_ids'].shape[0]
            # gtext_input += conversation
            # conversation = a + '</s>'
            # tokenized = self.tokenizer.build_input_ids(
            #     text=[conversation],
            #     max_length=self.max_length - total_len,
            #     add_special_tokens=False,
            #     truncation=True,
            #     padding="max_length",
            #     return_tensors='pt'
            # )
            # indexs.append(torch.zeros_like(tokenized['input_ids']).to(torch.bool))
            # input_ids.append(tokenized['input_ids'])
            # attention_masks.append(tokenized['attention_mask'])
            # labels.append(tokenized['input_ids'])
            # total_len += tokenized['input_ids'].shape[0]
            # gtext_input += conversation
        
        input_ids = torch.cat(input_ids)[:self.max_length]
        attention_masks = torch.cat(attention_masks)[:self.max_length]
        labels = torch.cat(labels)[:self.max_length]
        indexs = torch.cat(indexs)[:self.max_length]
        labels[labels == self.tokenizer.pad_token_id] = -100
        # print(f'labels: {labels[labels == self.tokenizer.box_place_ids]}')
        
        attention_masks[labels == self.tokenizer.track_token] = 0
        attention_masks[labels == self.tokenizer.box_token] = 0
        attention_masks[labels == self.tokenizer.temp_token] = 0 
        labels[labels == self.tokenizer.temp_token] = -100
        labels[labels == self.tokenizer.box_token] = -100
        labels[labels == self.tokenizer.track_token] = -100
        # torch.save(attention_masks, 'tt.pth')
        # logger.info(f'labels:{labels}')
        # logger.info(f'attention_masks:{attention_masks}')
        # print(f'ids: {self.tokenizer.box_place_ids}')

        return input_ids, attention_masks, labels, indexs, cur_instruction, gtext_input

    def find_boundaries_torch(self, mask):
        # 将torch.bfloat16转换为numpy数组
        from skimage.segmentation import find_boundaries
        mask_np = mask.to(torch.bool).numpy()
        boundaries = find_boundaries(mask_np, mode='outer')
        boundary_points = np.argwhere(boundaries)
        if boundary_points.size == 0:
            return torch.tensor([-1, -1, -1, -1], dtype = torch.bfloat16)
        h0, w0 = boundary_points.min(axis=0)
        h1, w1 = boundary_points.max(axis=0)
        return torch.tensor([w0, h0, w1, h1], dtype = torch.bfloat16)
        return torch.tensor([w0 / mask.shape[1], h0 / mask.shape[0],  w1 / mask.shape[1], h1 / mask.shape[0]], dtype = torch.bfloat16)
    # h0, w0, h1, w1

    def map_obj(self, boxes_value, boxes_seq) :
        """
        >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
        >>> boxes_seq_ = [[3, 1], [2]]
        >>> var = map_obj(normalized_boxes, boxes_seq_)
        >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]] /改掉
        >>> 改为[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1], [0.2,0.2,0.2,0.2]]
        """
        try:
            ret = []
            for boxes in boxes_seq:
                for box_index in boxes:
                    if isinstance(box_index, (list, tuple)):
                        for tt in box_index:
                            ret.append(boxes_value[tt])
                    else:
                        ret.append(boxes_value[box_index])
            return ret
        except:
            raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            # logger.info(ann["image"])
            if self.media_type in load_image_list:
                clip = None
                image, index, raw_size = self.load_and_transform_media_data_image(index, ann["image"], return_raw_size = True, use_dec=True) # raw_size : h, w
                # logger.info(f'image:{image.shape}')
                
            elif self.media_type in load_video_list:
                clip = None
                if "start" in ann and "end" in ann:
                    clip = [ann["start"], ann["end"]]
                # image, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
                image, sec, vlen, frame_indices, fps = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip, use_dec=False)

            # add q-former message
            if self.media_type == 'TVG':
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            else:
                msg = ""

            if self.media_type in ['REC', 'flickr']:
                ann['target']['boxes'] = self.process_bbox(ann['target']['boxes'], raw_size)
                # print(ann['target']['boxes_seq'])
                ann['target']['boxes_final'] = self.map_obj(ann['target']['boxes'], ann['target']['boxes_seq'])
            # logger.info(ann['target'])
            if self.media_type in ['tracking']:
                ann['target'] = [ann['target'][0] / ann['size'][1], ann['target'][1] / ann['size'][0], ann['target'][2] / ann['size'][1], ann['target'][3] / ann['size'][0]]
                sam_images = self.load_and_transform_media_data_video_sam2(ann["image"], False, 0)
            if self.media_type in ['tracking2']:
                ann['target'] = [ann['target'][0] / ann['size'][1], ann['target'][1] / ann['size'][0], ann['target'][2] / ann['size'][1], ann['target'][3] / ann['size'][0]]
                sam_images = self.load_and_transform_media_data_video_sam2(ann["image"], False, -1)
            if self.media_type == 'tracking3':
                # sam_images = 
                ann['target'] = [ann['target'][0] / raw_size[1], ann['target'][1] / raw_size[0], ann['target'][2] / raw_size[1], ann['target'][3] / raw_size[0]]
                sam_images = self.load_and_transform_media_data_video_sam2(ann["image"], image=True)
            if self.media_type == 'tracking4':
                # sam_images = 
                ann['target'] = [ann['target'][0] / ann['size'][1], ann['target'][1] / ann['size'][0], ann['target'][2] / ann['size'][1], ann['target'][3] / ann['size'][0]]
                sam_images = self.load_and_transform_media_data_video_sam2(ann["image"], False, 0)


            input_ids, attention_masks, labels, indexs, inst, gtext_input = self.process_qa(ann["qa"], msg)

            ret = {
                'input_ids': input_ids, 
                'attention_mask': attention_masks, 
                'labels': labels,
                'instruction': inst,
            }
            # 
            if 'mask_path' in ann:
                ret['mask_path'] = ann['mask_path']
            else:
                ret['mask_path'] = None
            ret['path'] = ann["image"]
            if self.media_type in load_image_list:
                ret['image'] = image
                ret['image_idx'] = indexs
            elif self.media_type in load_video_list:
                ret['video'] = image     
                ret['video_idx'] = indexs
                ret['sec'] = sec
                ret['vlen'] = [vlen]
                ret['frame_indices'] = frame_indices
                ret['fps'] = [fps]
            if self.media_type == 'REC' :
                ret['output_boxes'] = list(ann['target']['boxes_final'])
            elif self.media_type == 'flickr':
                ret['output_boxes'] = list(ann['target']['boxes_final'])
                if ret['output_boxes'] == []:
                    raise Exception('Null boxes')
            elif self.media_type == 'TVG':
                self.duration = round(vlen/fps, 2)
                # cg_data = self.get_cg_item(self.anno[index])
                cg_data = self.get_cg_item(index)
                cg_data['meta']['duration'] = round(vlen/fps, 2)
                ret['meta'] = cg_data['meta']
                ret['model_inputs'] = cg_data['model_inputs']
                
                
            elif self.media_type == 'grounding':
                raise NotImplementedError
            elif self.media_type == 'tracking':
                ret['input_boxes'] = ann['target']
                # ret['gt_masks'] = ann['gt_masks']
                ret['sam_images'] = sam_images
                ret['size_hw'] = ann['size'] #hw
            elif self.media_type == 'tracking2':
                # ret['gt_masks'] = ann['gt_masks']
                ret['input_boxes'] = ann['target']
                ret['sam_images'] = sam_images
                ret['size_hw'] = ann['size'] #hw
            elif self.media_type == 'tracking3':
                ret['input_boxes'] = ann['target']
                ret['sam_images'] = sam_images
                ret['size_hw'] = raw_size #hw
            elif self.media_type == 'tracking4':
                ret['input_boxes'] = ann['target']
                ret['sam_images'] = sam_images
                ret['size_hw'] = ann['size'] #hw

            return ret 


        except Exception as e:
            logger.info(e)
            logger.warning(f"Caught exception {e} when loading clips_anno {ann}")
            
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
        
        
    def get_cg_item(self, index):
        meta = self.anno[index]
        # print(meta.keys())
        # exit()
        # meta = self.data[index]

        if self.use_feature == False:
            return dict(meta=meta, model_inputs=dict())

        model_inputs = dict()

        if self.use_glove:  # False
            model_inputs["query_feat"] = self.get_query(meta["query"])
        else:
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)   # [16, 4096]
            
        if self.use_video : # True
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video :
                model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if "relevant_windows" in meta: ## For Qvhighlights test set
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if self.dset_name in ['charadesSTA', 'tacos', 'activitynet']: ## charades, tacos, nlq
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], self.duration, ctx_l)  # only one gt
            elif "subs_train" not in self.data_path:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt

        if 'qvhighlight' in self.data_path:
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]
        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return dict(meta=meta, model_inputs=model_inputs)
    
    def _get_query_feat_by_qid(self, qid):
        # QVhighlight dataset
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.pt")
        # q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        q_feat = torch.load(q_feat_path).numpy().astype(np.float32)
        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)  # 在GT frame idx中随机选两个
        else:
            if self.dset_name == 'nlq':
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l)) # 非GT的frame idx
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)  # 在非GT frame idx中随机选两个
        except:
            neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        
    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        max_len=ctx_l
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                max_len=max(max_len,rel_clip_ids[idx])
                # score_array_new = np.zeros(ctx_l + 1)
                score_array_new = np.zeros(max_len+1)
                # score_array_new[:ctx_l] = score_array
                score_array_new[:len(score_array)] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array   
    
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        # QVhighlight dataset
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.pt")
        # q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        q_feat = torch.load(q_feat_path).numpy().astype(np.float32)
        if self.q_feat_type == "last_hidden_state":
            # print(self.max_q_l)
            # exit()
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            try:
                _feat_path = join(_feat_dir, f"{vid}.pt")
                _feat = torch.load(_feat_path)["features"][:self.max_v_l].numpy().astype(np.float32)
            except:
                _feat_path = join(_feat_dir, f"{vid}.pt")
                _feat = torch.load(_feat_path)[:self.max_v_l].numpy().astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)               # (vlen=34, 768)
        return torch.from_numpy(v_feat)  # (Lv, D)

def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == 'qid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        if k == 'vid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data

def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        vid=batched_model_inputs["vid"],
        qid=batched_model_inputs["qid"],
    )
    targets = {}

    # import pdb; pdb.set_trace()

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets




if __name__ == '__main__':
    from ..tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
    tokenizer = MultimodalLlamaTokenizer.from_pretrained("BAAI/Emu2", local_files_only=True) 
    from easydict import EasyDict as edict
    from .create import get_train_transform
    Myd = ITTrainDataset(
        ann_file=edict(
            anno_path=f"/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new/image/caption/coco/train_100k.json", 
            data_root="p2:s3://coco_caption",
        ),
        transform=get_train_transform(None, None),
        tokenizer=tokenizer
    )
    data1 = Myd[0]
    data2 = Myd[1]
    breakpoint()
    assert data1['input_ids'].shape[0] == data2['input_ids'].shape[0]