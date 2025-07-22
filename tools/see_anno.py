import json

anno_path = "/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json"

with open(anno_path, "r") as f:
    infos = json.load(f)


for info in infos[:10]:
    print(len(info['interleaved_list']))
