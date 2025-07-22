qformer_fix_vicuna_msrvtt_cider = [31.4, 41.0, 42.3, 46.1, 44.7, 44.3, 48.9, 46.8]
coco_cider = [37.0, 51.8, 50.2, 50.0, 52.6, 52.0, 52.1, 53.7]

qformer_unfreeze_vicuna_msrvtt_cider = [3.5, 18.0, 22.0, 24.1, 29.7, 31.0, 32.8, 30.2]
unfreeze_coco_cider = [8.4, 14.4, 23.0, 23.4, 25.8, 34.3, 33.8, 34.8]

qformer_fix_vicuna_with_image_regression_msrvtt_cider = [46.8, 46.8, 42.9, 48.4, 45.6, 47.0]
qformer_fix_vicuna_with_image_regression_coco_cider = [53.7, 52.6, 52.4, 52.5, 54.7, 55.3]


further_msrvtt = [46.8, 45.5, 44.5, 46.4]
further_coco = [53.7, 53.9, 54.6, 53.3]


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

data = coco_cider
ax.plot(range(len(data)), data, color='blue', label='base')
data = qformer_fix_vicuna_with_image_regression_coco_cider
ax.plot(range(len(qformer_fix_vicuna_msrvtt_cider)-1, len(data)+len(qformer_fix_vicuna_msrvtt_cider)-1), data, color='red', label='continual training with regression & freeze vicuna')
data = further_coco
ax.plot(range(len(qformer_fix_vicuna_msrvtt_cider)-1, len(data)+len(qformer_fix_vicuna_msrvtt_cider)-1), data, color='green', label='continual training with regression & unfreeze vicuna')

plt.xlabel('Iters/1000')
plt.ylabel('cider value')
plt.title('coco-cider')
ax.legend()
plt.savefig(f'compare.png')


# 绘制折线图
# plot_line_chart(unfreeze_coco_cider)