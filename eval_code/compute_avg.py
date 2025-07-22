import json
from statistics import mean

data_root = "/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/results/merge_all_1:1/grounding/"
R1_03 = []
R1_05 = []
R1_07 = []
mIoU = []
HL_mAP = []
HL_Hit1 = []

for i in range(8):
    # with open(data_root+f'{i}-HL_metrics.json', 'r') as f:
    with open(data_root+f'{i}-hl_metrics.json', 'r') as f:
        data = json.load(f)
        R1_03.append(data["brief"]["MR-full-R1@0.3"])
        R1_05.append(data["brief"]["MR-full-R1@0.5"])
        R1_07.append(data["brief"]["MR-full-R1@0.7"])
        mIoU.append(data["brief"]["MR-full-mIoU"])

        if "HL-min-VeryGood-mAP" in data["brief"]:
            HL_mAP.append(data["brief"]["HL-min-VeryGood-mAP"])
            HL_Hit1.append(data["brief"]["HL-min-VeryGood-Hit1"])

print("R1@0.3: ", mean(R1_03))
print("R1@0.5: ", mean(R1_05))
print("R1@0.7: ", mean(R1_07))
print("mIoU: ", mean(mIoU))
if "HL-min-VeryGood-mAP" in data["brief"]:
    print("HL_mAP: ", mean(HL_mAP))
    print("HL_Hit1: ", mean(HL_Hit1))