DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

BOX_START = '<box_begin>'
# BOX_END = '<box_end>'
ATBOXES_PLACEHOLDER = '<box_begin><boxes>'
# ATBOXES_PLACEHOLDER = '<box_begin>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
QUESTION_PLACEHOLDER = '<question>'
TIME_START = '<time_begin>'
# TIME_END = '<time_end>'
TIME_PLACEHOLDER = '<temp>'
ATTEMP_PLACEHOLDER = TIME_START + TIME_PLACEHOLDER
# ATTEMP_PLACEHOLDER = TIME_START
TRACK_START='<track_begin>'
TRACK_PLACEHOLDER = '<tracking>'
TRACK_START_BOX = '<track_box>'
ATTRACK_PLACEHOLDER = TRACK_START + TRACK_PLACEHOLDER
need_template_list = ['REC', 'REC_qa', 'flickr', 'tracking', 'tracking_qa', 'tracking2', 'tracking2_qa', 'tracking3', 'tracking4'] 

load_image_list = ['image', 'REC', 'REC_qa', 'flickr']
load_video_list = ['video', 'TVG', 'TVG_qa', 'tracking','tracking_qa', 'tracking2', 'tracking2_qa','tracking3', 'tracking4', 'TVG+HL', 'TVG_reason']
special_tokens = [BOX_START, TIME_START, TIME_PLACEHOLDER, BOXES_PLACEHOLDER, TRACK_START, TRACK_PLACEHOLDER, TRACK_START_BOX]

# 任务种类：image:imageQA, video: videoQA, TVG:Grounding, REC:detection, flickr:reasoning detection