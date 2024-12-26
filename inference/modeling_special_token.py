import transformers
DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

BOX_START = '<box_begin>'
ATBOXES_PLACEHOLDER = '<box_begin><boxes>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
QUESTION_PLACEHOLDER = '<question>'
TIME_START = '<time_begin>'
TIME_PLACEHOLDER = '<temp>'
ATTEMP_PLACEHOLDER = TIME_START + TIME_PLACEHOLDER
TRACK_START='<track_begin>'
TRACK_PLACEHOLDER = '<tracking>'
TRACK_START_BOX = '<track_box>'
ATTRACK_PLACEHOLDER = TRACK_START + TRACK_PLACEHOLDER
need_template_list = ['REC', 'flickr', 'tracking', 'tracking2', 'tracking3', 'tracking4'] 

load_image_list = ['image', 'REC', 'flickr']
load_video_list = ['video', 'TVG', 'tracking', 'tracking2','tracking3', 'tracking4', 'TVG+HL']
special_tokens = [BOX_START, TIME_START, TIME_PLACEHOLDER, BOXES_PLACEHOLDER, TRACK_START, TRACK_PLACEHOLDER, TRACK_START_BOX]
