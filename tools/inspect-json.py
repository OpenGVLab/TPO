# import json

# def filter_json(json_file, output_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     total_len = 0
#     total_c = 0
    
#     for item in data:
#         total_len += len(item['caption'])
#         total_c += 1
    
#     print(total_len / total_c)

#     # with open(output_file, 'w') as f:
#         # json.dump(filtered_json, f)

# # Example usage
# json_file = '/mnt/petrelfs/wangchenting/MultiModalLLM/dataset/cc3m/cc3m_train.json'
# output_file = 'coco_5k_test_final_concat.json'              # Path to the output JSON file

# filter_json(json_file, output_file)

import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/webvid_10m_train_clean_230413.json', 'r') as file:
    data = json.load(file)

# 提取caption长度
caption_lengths = [len(item['caption']) for item in data]
word_count = [len(item['caption'].split(' ')) for item in data]

# 计算平均长度
# average_length = sum(caption_lengths) / len(caption_lengths)
# print('Average Caption word count', sum(word_count) / len(word_count))

# 绘制直方图
plt.hist(word_count, bins=20)
plt.title('Caption Length Histogram')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig('hist.png')

# 输出平均长度
# print('Average Caption Length:', average_length)