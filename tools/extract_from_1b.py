import torch

pth = '/mnt/petrelfs/share_data/lixinhao/avp_1b_f4_coco_smit_e4.pt'

s = torch.load(pth, map_location='cpu')

new_state_dict = s['module']
sv_state_dict = {}

for k in new_state_dict.keys():
    if 'vision_encoder' in k:
        sv_state_dict[k.replace('vision_encoder.','')] = new_state_dict[k]

torch.save(sv_state_dict, 'pretrained_vit.pth')