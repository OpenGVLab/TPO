import torch

model = torch.load("/mnt/petrelfs/share_data/likunchang/model/videochat2/umt_l16_qformer.pth", map_location='cpu')

for k in model.keys():
    print(k, model[k].shape)
# print(model['pos_embed'].shape)