import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy
from easydict import EasyDict as edict
anno_root_it = "/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new"
data_dir = '/mnt/petrelfs/share/videointern/annotations'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")
anno_root_downstream = __os.path.join(data_dir, "anno_downstream")

# ============== pretraining datasets=================
available_corpus = dict(
     # pretraining image datasets
     caption_coco = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    vqa_tvqa = dict (
        anno_path=f'/mnt/petrelfs/share_data/wangchenting/tvqa.json',
        data_root="p2:s3://tvqa/frames_fps3_hq/",
        media_type='video',
        video_reader_type='img'
    ),
    caption_coco_100k = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train_100k.json", 
        data_root="p2:s3://coco_caption",
    ),
    caption_llava = dict(
        anno_path=f"{anno_root_it}/image/caption/llava/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    caption_svit = dict(
        anno_path=f"{anno_root_it}/image/caption/svit/train_20k.json", 
        data_root="p2:s3://svit",
    ),
    caption_minigpt4 = dict(
        anno_path=f"{anno_root_it}/image/caption/minigpt4/train.json", 
        data_root="p2:s3://minigpt4/image",
    ),
    caption_paragraph_captioning = dict(
        anno_path=f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
        data_root="p2:s3://m3it/image-paragraph-captioning",
    ),
    caption_textcaps = dict(
        anno_path=f"{anno_root_it}/image/caption/textcaps/train.json", 
        data_root="p2:s3://m3it/textcap",
    ),
    caption_som = dict(
        anno_path=f"{anno_root_it}/image/caption/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    classification_imagenet = dict(
        anno_path=f"{anno_root_it}/image/classification/imagenet/train.json", 
        data_root="p2:s3://m3it/imagenet",
    ),
    classification_coco_itm = dict(
        anno_path=f"{anno_root_it}/image/classification/coco_itm/train.json", 
        data_root="p2:s3://m3it/coco-itm",
    ),
    conversation_svit = dict(
        anno_path=f"{anno_root_it}/image/conversation/svit/train_30k.json", 
        data_root="p2:s3://svit",
    ),
    conversation_lvis_instruct4v = dict(
        anno_path=f"{anno_root_it}/image/conversation/lvis_instruct4v/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data",
    ),
    conversation_som = dict(
        anno_path=f"{anno_root_it}/image/conversation/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    reasoning_clevr = dict(
        anno_path=f"{anno_root_it}/image/reasoning/clevr/train.json", 
        data_root="p2:s3://m3it/clevr",
    ),
    reasoning_visual_mrc = dict(
        anno_path=f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
        data_root="p2:s3://m3it/visual-mrc",
    ),
    reasoning_llava = dict(
        anno_path=f"{anno_root_it}/image/reasoning/llava/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    reasoning_svit = dict(
        anno_path=f"{anno_root_it}/image/reasoning/svit/train_50k.json", 
        data_root="p2:s3://svit",
    ),
    reasoning_science_qa = dict(
        anno_path=f"{anno_root_it}/image/reasoning/science_qa/train.json", 
        data_root="p2:s3://m3it/science-qa",
    ),
    vqa_vqav2 = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_vqav2_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2_chinese/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_gqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/gqa/train.json", 
        data_root="p2:s3://m3it/gqa",
    ),
    vqa_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_okvqa_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa_chinese/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_a_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
        data_root="p2:s3://m3it/a-okvqa",
    ),
    vqa_viquae = dict(
        anno_path=f"{anno_root_it}/image/vqa/viquae/train.json", 
        data_root="p2:s3://m3it/viquae",
    ),
    vqa_ocr_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
        data_root="p2:s3://m3it/ocr-vqa",
    ),
    vqa_ocr_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/OCRVQA/images",
    ),
    vqa_text_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train.json", 
        data_root="p2:s3://m3it/text-vqa",
    ),
    vqa_text_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/TextVQA/train_images",
    ),
    vqa_st_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train.json", 
        data_root="p2:s3://m3it/st-vqa",
    ),
    vqa_st_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ST-VQA",
    ),
    vqa_docvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train.json", 
        data_root="p2:s3://m3it/docvqa",
    ),
    vqa_docvqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/DocVQA/train/train/documents",
    ),
    vqa_infovqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/infovqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/InfoVQA/infographicVQA_train_v1.0_images",
    ),
    vqa_ai2d = dict(
        anno_path=f"{anno_root_it}/image/vqa/ai2d/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ai2diagram/ai2d/images",
    ),
    vqa_chart_qa = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/chartqa/ChartQA Dataset/train/png",
    ),
    vqa_chart_qa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ChartQA/train",
    ),
    vqa_dvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    vqa_dvqa_80k = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train_80k.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    grounding_coco = dict(
        anno_path=f"{anno_root_it}/image/grounding/coco/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/coco",
    ),
    grounding_vg = dict(
        anno_path=f"{anno_root_it}/image/grounding/vg/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/vg",
    ),
    # video
    caption_textvr = dict(
        anno_path=f"{anno_root_it}/video/caption/textvr/train.json", 
        data_root="p2:s3://TextVR/Video",
        media_type="video"
    ),
    caption_videochat = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    caption_videochat_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat_chinese/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    caption_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/caption/videochatgpt/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30",
        media_type="video"
    ),
    caption_webvid = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video"
    ),
    caption_webvid_80k = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train_80k.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video"
    ),
    caption_youcook2 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_debug.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_youcook2_L20 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_L20.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_smit = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_smit_40k = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train_40k.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_vatex_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/vatex_chinese/train.json", 
        data_root="p2:s3://k600",
        media_type="video"
    ),
    classification_k710 = dict(
        anno_path=f"{anno_root_it}/video/classification/k710/train.json", 
        data_root="",
        media_type="video"
    ),
    classification_ssv2 = dict(
        anno_path=f"{anno_root_it}/video/classification/ssv2/train.json", 
        # anno_path=f"{anno_root_it}/video/classification/ssv2/train_60k.json", 
        data_root="pnorm2:s3://ssv2-video",
        media_type="video"
    ),
    classification_mitv1 = dict(
        anno_path=f"{anno_root_it}/video/classification/mitv1/train.json", 
        data_root="p2hdd:s3://Moments_in_Time_Raw/videos",
        media_type="video"
    ),
    conversation_videochat1 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat1/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    conversation_videochat2 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    conversation_videochat2_chinese = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2_chinese/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    vqa_rgbd = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/nturgbd.json',
        data_root='pssd:s3://nturgbd',
        media_type='video',
        video_reader_type='av'
    ),
    conversation_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30",
        media_type="video"
    ),
    reasoning_next_qa = dict(
        anno_path=f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        data_root="p2:s3://nextqa",
        media_type="video"
    ),
    reasoning_clevrer_qa = dict(
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc.json", 
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc_plus.json", 
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_clevrer_mc = dict(
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k.json",  
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k_debug.json",  
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_star = dict(
        anno_path=f"{anno_root_it}/video/reasoning/star/train.json", 
        data_root="p2:s3://star/Charades_v1_480",
        media_type="video"
    ),
    vqa_ego_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        data_root="pnorm2:s3://egoqa/split_videos",
        media_type="video"
    ),
    vqa_ego_qa_L15 = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train_L15.json", 
        data_root="pnorm2:s3://egoqa/split_videos",
        media_type="video"
    ),
    vqa_tgif_frame_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_tgif_transition_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_webvid_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video",
    ),
    vqa_webvid_qa_30k = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train_30k.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video",
    ),
    vqa_perception_train = dict (
        anno_path='/mnt/petrelfs/share_data/wangchenting/mc_question_train_forchoice.json',
        data_root='pssd:s3://perception/videos',
        media_type='video'
    ),
    egotask = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/direct_train_qas_convertmc.json',
        data_root='',
        media_type="video",
        video_reader_type='pt'
    ),
    caption_sharegpt4v = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train.json", 
        data_root="p2:s3://sharegpt4v/data",
    ),
    caption_sharegpt4o = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/sharegpt4o.json',
        data_root="pssd:s3://perception/videos",
        media_type="video",
    ),
    lsmdc = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/lsmdc.json',
        data_root="pvideo:s3://LSMDC",
        media_type="video",
        video_reader_type="av",
    ),
    caption_sharegpt4v_420k = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train_420k.json", 
        data_root="p2:s3://sharegpt4v/data",
    ),
    caption_sharegptvideo_300k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_300k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_sharegptvideo_600k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_600k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_600k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_240k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_240k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_900k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_900k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_kinetics = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/kinetics_train.json", 
        data_root="",
        media_type="video",
    ),
    caption_vidln_oops = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_oops_L15 = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train_L15.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_ovis = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/ovis_train.json", 
        data_root="p2:s3://ovis/train",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_uvo_sparse = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_sparse_train.json", 
        data_root="s3://UVO/uvo_videos_sparse",
        media_type="video",
    ),
    caption_vidln_uvo_dense = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_dense_train.json", 
        data_root="sssd:s3://UVO/uvo_videos_dense",
        media_type="video",
    ),
    caption_favd = dict(
        anno_path=f"{anno_root_it}/video/caption/favd/train.json", 
        data_root="p2:s3://favd",
        media_type="video",
    ),
    grounding_didemo = dict(
        anno_path=f"{anno_root_it}/video/grounding/didemo/train.json", 
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="video",
    ),
    # text
    conversation_sharegpt = dict(
        anno_path=f"{anno_root_it}/text/sharegpt/train.json", 
        data_root="",
        media_type="text",
    ),
    conversation_openhermes = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train.json", 
        data_root="",
        # "text",
    ),
    conversation_openhermes_200k = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train_200k.json", 
        data_root="",
        # "text",
    ),
    # add docci
    caption_docci = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/train.json", 
        data_root="pnorm2:s3://docci/images",
    ),
    caption_docci_test = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/test.json", 
        data_root="pnorm2:s3://docci/images",
    ),
    # TimeIT
    timeit_ANet = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ANet/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30/",
        media_type="video"
    ),
    timeit_COIN = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/COIN/train.json", 
        data_root="p2hdd:s3://COIN_320p",
        media_type="video"
    ),
    timeit_DiDeMo = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/DiDeMo/train.json", 
        data_root="s3://yjsBucket/DiDeMo/",
        media_type="video"
    ),
    timeit_HiREST = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/HiREST/train.json", 
        data_root="zxy_pssd:s3://HiREST/",
        media_type="video"
    ),
    timeit_QuerYD = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/QuerYD/train.json", 
        data_root="zxy_pssd:s3://QuerYD/",
        media_type="video"
    ),
    timeit_SumMe = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/SumMe/train.json", 
        data_root="zxy_pssd:s3://SumMe",
        media_type="video"
    ),

    timeit_TVSum = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/TVSum/train.json", 
        data_root="zxy_pssd:s3://TVSum",
        media_type="video"
    ),
    timeit_ViTT = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ViTT/train.json", 
        data_root="zxy_pssd:s3://ViTT",
        media_type="video"
    ),
    timeit_yttemporal180m = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/yttemporal180m/train.json", 
        data_root="p2hdd:s3://YT-Temporal-180M",
        media_type="video"
    ),

    cc_new_3m=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/cc3m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    cc_new_12m=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/cc12m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    sbu_new=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/sbu-1m-fuse.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    # pretraining video datasets
    webvid_fuse_10m=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/webvid-fuse-10m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="video",
        jump_filter=True
    ),
    internvid_v1_40m=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/internvid1-40m.json",
        data_root="",
        data_root_prefix="phdd:",
        media_type="video",
        jump_filter=True
    ),
    internvid_v2_avs_50m=dict( 
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/internvid2.s3path.success_avs_50M_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    internvid_v2_avs_43m_av=dict( 
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data-h2/internvid2.s3path.success_av.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    # pretraining datasets
    internvid_recap=dict(
        anno_path="/mnt/petrelfs/share_data/videointern/annotations/anno_pretrain_lkc/internvid_0.6M_recap.json", 
        data_root="",
        media_type="video"
    ),
    smit=dict(
        anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
        data_root="pvideo:s3://S-MiT",
        media_type="video"
    ),
    smit_av=dict(
        anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
        data_root="pvideo:s3://S-MiT",
        media_type="audio_video",
        read_audio_from_video=False
    ),
    allseeing_short_2M=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/as-short-2m_lxh.json", 
        data_root="changeme:s3://SA-1B",
        media_type="image",
        caption_augmentation=dict(caption_sample_type='uniform')
    ),
    cc3m=dict(
        anno_path=f"{anno_root_pt}/cc3m_train.json", 
        data_root="pssd:s3://GCC",
        media_type="image"
    ),
    cc12m=dict(
        anno_path=f"{anno_root_pt}/cc12m_train.json", 
        data_root="pssd:s3://GCC/GCC12m",
        media_type="image"
    ),
    sbu=dict(
        anno_path=f"{anno_root_pt}/sbu.json", 
        data_root="pssd:s3://SBU/images",
        media_type="image"
    ),
    vg=dict(
        anno_path=f"{anno_root_pt}/vg.json", 
        data_root="pnorm2:s3://vg-dataset/images",
        media_type="image"
    ),
    coco_test=dict(
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_test_vast_only_first.json',
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_test_vast.json',
        anno_path=f"/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json", 
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_5k_test_final.json',
        data_root="pnorm2:s3://coco-caption",
        media_type="image"
    ),
    coco_test_4shot=dict(
        anno_path=f"/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json", 
        data_root="pnorm2:s3://coco-caption",
        media_type="image",
        few_shot_config=dict(
            num_shot=4,
            few_shot_template="{visual}{caption}",
            use_rice=False
        )
    ),

    coco=dict(
        anno_path=f"{anno_root_pt}/coco.json", 
        data_root="pnorm2:s3://coco-caption",
        media_type="image"
    ),
    imagenet1k=dict(
        anno_path=f"{anno_root_pt}/imagenet1k_train.json", 
        data_root="/mnt/petrelfs/share/images/train",
        media_type="image",
        prompt="imagenet",
    ),
    imagenet1kval=dict(
        anno_path='/mnt/petrelfs/wangchenting/working/imagenet/imagenet_1k_val.json',
        data_root='/mnt/petrelfs/share/images/val',
        media_type="image",
        prompt="imagenet",
    ),
    webvid=dict(
        anno_path=f"{anno_root_pt}/webvid_train.json", 
        data_root="pssd:s3://WebVid2M",
        media_type="video"
    ),
    webvid2m_jiaer=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/webvid2M_jiaer.json", 
        data_root="pssd:s3://WebVid2M",
        media_type="video"
    ),
    webvid_10m=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/webvid_10m_train_clean_230413.json",
        data_root="pssd:s3://WebVid10M",
        media_type="video",
    ),
    summarized_230415=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/video_caption_summarized_list/video_caption_summarized_list_230415_processed_resized.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_3m=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part3M.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_3m_overlap=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part3M_overlap.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_10m=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part10M.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_10m_new=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/video_caption_summarized_list/resized_0602_200m_10m.json",
        data_root="",
        media_type="video"
    ),
    internv2_10m_avs=dict(
        anno_path="/mnt/petrelfs/share_data/lixinhao/annos/internvid2.s3path.success_avs_random10M.json",
        data_root="",
        media_type="video"
    ),
    internv2_10m_v=dict(
        anno_path="/mnt/petrelfs/share_data/lixinhao/annos/internvid2.s3path.success_v_random10M.json",
        data_root="",
        media_type="video"
    ),
    kinetics400=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics400_train.json",
        data_root="/mnt/petrelfs/videointern/k400",
        media_type="video",
        prompt="prompt",
    ),
    kinetics710=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics710_train.json",
        data_root="",
        media_type="video",
        prompt="prompt",
    ),
    kinetics710_raw=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics710_raw_train.json",
        data_root="",
        media_type="only_video", # NOTE NotImplemented
    ),
    # downstream datasets.
    videocc3m_full=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/videocc3m_full.json", 
        data_root="pssd:s3://videocc3m",
        media_type="video",
    ),
    videocc3m_part=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/videocc3m_part2.5M.json", 
        data_root="pssd:s3://videocc3m",
        media_type="video",
    ),
    youtube_full=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/youtube_0407/youtube_full.json", 
        data_root="",
        media_type="video",
    ),
    youtube_part=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/youtube_0407/youtube_part2.5M.json", 
        data_root="",
        media_type="video",
    ),
    intern10mfiltered=dict(
        anno_path="phdd:s3://video_caption_summarized_list/resized_0613_filtered_30p_10m.json",
        data_root="",
        media_type="video"
    ),
    summarized_230613_filtered_30p_random2m=dict( # internvid2m-filtered
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/resized_0613_filtered_30p_10m_random2m.json",
        data_root="",
        media_type="video" # 2129654
    ),
    summarized_230930_filtered_30p_random2m=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/resized_0930_filtered_30p_10m_random2m.json",
        data_root="",
        media_type="video" # 2089651
    ),
    youtube_5M_avscap_av=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_av.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_avscap_av_audio=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_av.json",
        data_root="",
        media_type="audio_video"
    ),
    youtube_5M_avscap_av_audio_full_caption=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_clean_full_caption.json",
        data_root="",
        media_type="audio_video"
    ),
    youtube_5M_avscap_a=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_selected_audiocap_av.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_avscap_v=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_selected_vidcap_v.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_vscap_v=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_vs_captions_v.json",
        data_root="",
        media_type="video"
    ),

    ytt_8M_avs=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_avscap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_av=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_7.6m_avcap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_vs=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_7.6m_vscap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_avs_all=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/ytt_pack1_for_exp_7.6m_avcaps_all.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_2M_avs_all=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/ytt_pack1_for_exp_2m_avcaps_all.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    # audio-text
    wavcaps_400k_1=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/AudioSet_SL/as_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/AudioSet_SL_flac/",
        media_type="audio"
    ),
    wavcaps_400k_2=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/BBC_Sound_Effects/bbc_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/BBC_Sound_Effects_flac/",
        media_type="audio"
    ),
    wavcaps_400k_3=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/FreeSound/fsd_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/FreeSound_flac/",
        media_type="audio"
    ),
    wavcaps_400k_4=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/SoundBible/sb_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/SoundBible_flac/",
        media_type="audio"
    ),
    # interleaved-video-text
    internvid2_interleaved_1M_avscap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    internvid2_interleaved_1M_avscap_2clip=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=2
    ),
    internvid2_interleaved_1M_avscap_l2048=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=2048
    ),
    internvid2_interleaved_1M_asrcap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    internvid2_interleaved_1M_asrcap_l2048=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=2048
    ),
    internvid2_interleaved_1M_vcap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_vcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    howtointern7m=dict(
        anno_path="/mnt/petrelfs/wangchenting/multimodalllm/HowtoInterlink7M.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    REC_refcocog_google_val=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_google_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_ref3_train=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_ref3_train.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_ref3_train_qa=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_ref3_train.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC_qa',
    ),
    REC_ref3_train_32k=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_ref3_train_32k.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_allseeing_train=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/all_seeing_400k.json',
        data_root="p2hddimg:s3://SA-1B/",
        media_type='REC',
    ),
    REC_allseeing_train_qa=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/all_seeing_400k.json',
        data_root="p2hddimg:s3://SA-1B/",
        media_type='REC_qa',
    ),
    REC_refcocop_unc_val=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcoco_unc_val=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcocog_umd_test=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_test.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcoco_unc_testB=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testB.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcoco_unc_testA=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testA.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcocop_unc_testB=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testB.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcocop_unc_testA=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testA.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    REC_refcocog_umd_val=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    conversation_llava = dict(
        anno_path=f"{anno_root_it}/image/conversation/llava/train.json", 
        data_root="pnorm2:s3://coco-caption",
        media_type='image',
    ),
    samv2 = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/third-party/segment-anything-2/data/sam5.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking',
    ),
    samv2_100k = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/third-party/segment-anything-2/data/sam5_100k.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking',
    ),
    samv2_100k_qa = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/third-party/segment-anything-2/data/sam5_100k.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking_qa',
    ),
    davis = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/davis17.json", 
        data_root="shdd:s3://",
        media_type='tracking2',
    ),
    ref_youtube = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/refros_youtube.json", 
        data_root="shdd:s3://",
        media_type='tracking2',
    ), 
    mevis = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/mevis2.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking2',
    ),
    mevis_qa = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/mevis2.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking2_qa',
    ),
    mevis_val = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/mevis_val_u2.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking2',
    ),
    refros_youtube = dict(
        anno_path="/mnt/petrelfs/yanziang/trackingllm/videoagent/refros_youtube.json",
        data_root = '',
        media_type='tracking2',
    ),
    lasot1 = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/lasot.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="",
        media_type='tracking',
    ),
    lasot2 = dict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/lasot.json", 
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/sam_debug2.json',
        data_root="pnorm2:s3://",
        media_type='tracking2',
    ),
    REC_allseeing_track=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/all_seeing_400k.json',
        data_root="p2hddimg:s3://SA-1B/",
        media_type='tracking3',
    ),
    REC_allseeing_track_150k=dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/all_seeing_150k.json',
        data_root="p2hddimg:s3://SA-1B/",
        media_type='tracking3',
    ),
    sam1ktest = dict(
        # anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/third-party/segment-anything-2/data/sam1ktest.json", 
        anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/third-party/segment-anything-2/data/samexample.json',
        data_root="pnorm2:s3://",
        media_type='tracking',
    ),
    recvg = dict(
        # type='RECDataset',
        # filename=r'/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/GC_genome196_train.jsonl',
        # image_folder=r'pnorm2:s3://vg-dataset/',
        # template_file=r'{{fileDirname}}/template/REC.json',

        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/GC_genome196_train.json',
        data_root="pnorm2:s3://vg-dataset/",
        media_type='REC',
    ),
    
    flickr_bbox = dict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/CWB_flickr30k_train.json',
        # anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/flicktest.json', # 1k data for debug  
        data_root='/mnt/petrelfs/share_data/chenzhe1/data/flickr30k/flickr30k-images/',
        media_type = 'flickr',
    ),
    flickr_bbox_debug = dict(
        # anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/CWB_flickr30k_train.json',
        anno_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/flicktest.json', # 1k data for debug  
        data_root='/mnt/petrelfs/share_data/chenzhe1/data/flickr30k/flickr30k-images/',
        media_type = 'flickr',
    ),
    
    didemo_tvg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/didemo_tvg.json",
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="TVG"
    ),
    queryd_tvg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/queryd_tvg.json",
        data_root="pssd_zxy:s3://QuerYD/videos",
        media_type="TVG"
    ),
    hirest_tvg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/hirest_tvg.json",
        data_root="pssd_zxy:s3://HiREST",
        media_type="TVG"
    ),
    ActivityNet_tvg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/ActivityNet_tvg.json",
        data_root="pssd:s3://anet/ANet_320p/train",
        media_type="TVG"
    ),

    ActivityNet_tvg_reason = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/ActivityNet_tvg_cg/ActivityNet_tvg_cg-clean.json",
        data_root="pssd:s3://anet/ANet_320p/train",
        media_type="TVG_reason",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/text_feature_4096",
    ),

    charades_tvg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
    ),
    charades_tvg_1 = dict(
        anno_path = '/mnt/petrelfs/lizhilin/AnnoData/Temporal-Grounding/charades_split/charades_tvg_1.json',
        data_root='pssd:s3://star/Charades_v1_480',
        media_type = 'TVG',
    ),

    didemo_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/didemo_tvg_cg/didemo_tvg_cg.json",
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/text_feature_4096",
    ),
    didemo_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/didemo_tvg_cg/didemo_tvg_cg.json",
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/text_feature_4096",
    ),
    didemo_tvg_cg_val = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/didemo_tvg_cg/didemo_g_val.json",
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/text_feature_4096",
    ),
    didemo_tvg_cg_test = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/didemo_tvg_cg/didemo_g_test.json",
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/didemo/text_feature_4096",
    ),
    queryd_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/queryd_tvg_cg/queryd_tvg_cg.json",
        data_root="pssd:s3://QuerYD/videos",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/queryd/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/queryd/text_feature_4096",
    ),
    queryd_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/queryd_tvg_cg/queryd_tvg_cg.json",
        data_root="pssd:s3://QuerYD/videos",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/queryd/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/queryd/text_feature_4096",
    ),
    hirest_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/hirest_tvg_cg/hirest_tvg_cg.json",
        data_root="pssd:s3://HiREST",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/hirest/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/hirest/text_feature_4096",
    ),
    hirest_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/hirest_tvg_cg/hirest_tvg_cg.json",
        data_root="pssd:s3://HiREST",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/hirest/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/hirest/text_feature_4096",
    ),
    ActivityNet_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/ActivityNet_tvg_cg/ActivityNet_tvg_cg-clean.json",
        data_root="pssd:s3://anet/ANet_320p/train",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/text_feature_4096",
    ),
    ActivityNet_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/ActivityNet_tvg_cg/ActivityNet_tvg_cg-clean.json",
        data_root="pssd:s3://anet/ANet_320p/train",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/ActivityNet/text_feature_4096",
    ),
    charades_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/charades_tvg_cg_train1.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
    ),
    tacos_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/tacos_tvg_cg/TACoS_tvg_cg.json",
        data_root="pssd:s3://mpii/videos",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/tacos/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/tacos/text_feature_4096",
    ),
    tacos_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/tacos_tvg_cg/TACoS_tvg_cg.json",
        data_root="pssd:s3://mpii/videos",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/tacos/video_feature_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/tacos/text_feature_4096",
    ),
    nlq_tvg_cg_qa = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/nlq_tvg_cg/nlq_tvg_cg.json",
        data_root="pssd:s3://ego4d_v2/320p",
        media_type="TVG_qa",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/nlq/video_feature_1b_v2",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/nlq/text_feature_4096",
    ),
    nlq_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/nlq_tvg_cg/nlq_tvg_cg.json",
        data_root="pssd:s3://ego4d_v2/320p",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/nlq/video_feature_1b_v2",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/nlq/text_feature_4096",
    ),
    qvhl_tvg_cg = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/qvhl_tvg_cg.json",
        data_root="pssd:s3://qvhighlight/videos",
        media_type="TVG+HL",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
    ),
    qvhl_tvg_cg_new = dict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/qvhl_tvg_cg_tg.json",
        data_root="pssd:s3://qvhighlight/videos",
        media_type="TVG+HL",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
    ),
    FT_QVHighlights= dict(
        anno_path = f"/mnt/petrelfs/share_data/zengxiangyu/TimeIT/QVHigh-FT/FT_QVHighlights.json", 
        data_root = "pssd:s3://qvhighlight/videos/",
        media_type= "video"
    ),
    FT_QVHighlights_v1= dict(
        anno_path = f"/mnt/petrelfs/share_data/zengxiangyu/TimeIT/QVHigh-FT/FT_QVHighlights_V1.json", 
        data_root = "pssd:s3://qvhighlight/videos/",
        media_type= "video"
    ),
)

available_corpus['stage2_TVG'] = [
    available_corpus['didemo_tvg_cg'],
    available_corpus['tacos_tvg_cg'],
    available_corpus['nlq_tvg_cg'],
    available_corpus['queryd_tvg_cg'],
    available_corpus['hirest_tvg_cg'],
    available_corpus['ActivityNet_tvg_cg'],
]

available_corpus['stage2_TVG_ablate'] = [
    available_corpus['didemo_tvg_cg'],
    available_corpus['tacos_tvg_cg'],
    available_corpus['nlq_tvg_cg'],
    available_corpus['queryd_tvg_cg'],
    available_corpus['hirest_tvg_cg'],
    available_corpus['ActivityNet_tvg_cg'],
    available_corpus['qvhl_tvg_cg']
]

available_corpus["stage2_Track"] = [
    available_corpus['samv2_100k'],
] 


available_corpus_qa = dict(
    # image
    caption_coco = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train.json", 
        data_root="pnorm2:s3://coco-caption",
    ),
    vqa_tvqa = dict (
        anno_path=f'/mnt/petrelfs/share_data/wangchenting/datasets/tvqa.json',
        data_root="p2:s3://tvqa/frames_fps3_hq/",
        media_type='video',
        video_reader_type='img'
    ),
    caption_coco_100k = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train_100k.json", 
        data_root="pnorm2:s3://coco-caption",
    ),
    caption_llava = dict(
        anno_path=f"{anno_root_it}/image/caption/llava/train.json", 
        data_root="pnorm2:s3://coco-caption",
    ),
    caption_svit = dict(
        anno_path=f"{anno_root_it}/image/caption/svit/train_20k.json", 
        data_root="p2:s3://svit",
    ),
    caption_minigpt4 = dict(
        anno_path=f"{anno_root_it}/image/caption/minigpt4/train.json", 
        data_root="p2:s3://minigpt4/image",
    ),
    caption_paragraph_captioning = dict(
        anno_path=f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
        data_root="p2:s3://m3it/image-paragraph-captioning",
    ),
    caption_textcaps = dict(
        anno_path=f"{anno_root_it}/image/caption/textcaps/train.json", 
        data_root="p2:s3://m3it/textcap",
    ),
    caption_som = dict(
        anno_path=f"{anno_root_it}/image/caption/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    classification_imagenet = dict(
        anno_path=f"{anno_root_it}/image/classification/imagenet/train.json", 
        data_root="p2:s3://m3it/imagenet",
    ),
    classification_coco_itm = dict(
        anno_path=f"{anno_root_it}/image/classification/coco_itm/train.json", 
        data_root="p2:s3://m3it/coco-itm",
    ),
    conversation_llava = dict(
        anno_path=f"{anno_root_it}/image/conversation/llava/train.json", 
        data_root="pnorm2:s3://coco-caption",
    ),
    conversation_svit = dict(
        anno_path=f"{anno_root_it}/image/conversation/svit/train_30k.json", 
        data_root="p2:s3://svit",
    ),
    conversation_lvis_instruct4v = dict(
        anno_path=f"{anno_root_it}/image/conversation/lvis_instruct4v/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data",
    ),
    conversation_som = dict(
        anno_path=f"{anno_root_it}/image/conversation/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    reasoning_clevr = dict(
        anno_path=f"{anno_root_it}/image/reasoning/clevr/train.json", 
        data_root="p2:s3://m3it/clevr",
    ),
    reasoning_visual_mrc = dict(
        anno_path=f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
        data_root="p2:s3://m3it/visual-mrc",
    ),
    reasoning_llava = dict(
        anno_path=f"{anno_root_it}/image/reasoning/llava/train.json", 
        data_root="pnorm2:s3://coco-caption",
    ),
    reasoning_svit = dict(
        anno_path=f"{anno_root_it}/image/reasoning/svit/train_50k.json", 
        data_root="p2:s3://svit",
    ),
    reasoning_science_qa = dict(
        anno_path=f"{anno_root_it}/image/reasoning/science_qa/train.json", 
        data_root="p2:s3://m3it/science-qa",
    ),
    vqa_vqav2 = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_vqav2_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2_chinese/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_gqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/gqa/train.json", 
        data_root="p2:s3://m3it/gqa",
    ),
    vqa_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_okvqa_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa_chinese/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_a_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
        data_root="p2:s3://m3it/a-okvqa",
    ),
    vqa_viquae = dict(
        anno_path=f"{anno_root_it}/image/vqa/viquae/train.json", 
        data_root="p2:s3://m3it/viquae",
    ),
    vqa_ocr_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
        data_root="p2:s3://m3it/ocr-vqa",
    ),
    vqa_ocr_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/OCRVQA/images",
    ),
    vqa_text_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train.json", 
        data_root="p2:s3://m3it/text-vqa",
    ),
    vqa_text_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/TextVQA/train_images",
    ),
    vqa_st_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train.json", 
        data_root="p2:s3://m3it/st-vqa",
    ),
    vqa_st_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ST-VQA",
    ),
    vqa_docvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train.json", 
        data_root="p2:s3://m3it/docvqa",
    ),
    vqa_docvqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/DocVQA/train/train/documents",
    ),
    vqa_infovqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/infovqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/InfoVQA/infographicVQA_train_v1.0_images",
    ),
    vqa_ai2d = dict(
        anno_path=f"{anno_root_it}/image/vqa/ai2d/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ai2diagram/ai2d/images",
    ),
    vqa_chart_qa = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/chartqa/ChartQA Dataset/train/png",
    ),
    vqa_chart_qa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ChartQA/train",
    ),
    vqa_dvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    vqa_dvqa_80k = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train_80k.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    grounding_coco = dict(
        anno_path=f"{anno_root_it}/image/grounding/coco/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/coco",
    ),
    grounding_vg = dict(
        anno_path=f"{anno_root_it}/image/grounding/vg/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/vg",
    ),
    # video
    caption_textvr = dict(
        anno_path=f"{anno_root_it}/video/caption/textvr/train.json", 
        data_root="pnorm2:s3://textvr/Video",
        media_type="video"
    ),
    caption_videochat = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    caption_videochat_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat_chinese/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    caption_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/caption/videochatgpt/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30",
        media_type="video"
    ),
    caption_webvid = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video"
    ),
    caption_webvid_80k = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train_80k.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video"
    ),
    caption_youcook2 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_debug.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_youcook2_L20 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_L20.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_smit = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_smit_40k = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train_40k.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_vatex_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/vatex_chinese/train.json", 
        data_root="p2:s3://k600",
        media_type="video"
    ),
    classification_k710 = dict(
        anno_path=f"{anno_root_it}/video/classification/k710/train.json", 
        # anno_path=f"{anno_root_it}/video/classification/k710/train_60k.json", 
        data_root="",
        media_type="video"
    ),
    classification_ssv2 = dict(
        anno_path=f"{anno_root_it}/video/classification/ssv2/train.json", 
        # anno_path=f"{anno_root_it}/video/classification/ssv2/train_60k.json", 
        data_root="pnorm2:s3://ssv2-video",
        media_type="video"
    ),
    classification_mitv1 = dict(
        anno_path=f"{anno_root_it}/video/classification/mitv1/train.json", 
        data_root="p2hdd:s3://Moments_in_Time_Raw/videos",
        media_type="video"
    ),
    conversation_videochat1 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat1/train.json", 
        data_root="pnorm2:s3://webvid10m",
        media_type="video"
    ),
    conversation_videochat2 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    conversation_videochat2_chinese = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2_chinese/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    vqa_rgbd = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/nturgbd.json',
        data_root='pssd:s3://nturgbd',
        media_type='video',
        video_reader_type='av'
    ),
    conversation_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30",
        media_type="video"
    ),
    reasoning_next_qa = dict(
        anno_path=f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        data_root="p2:s3://nextqa",
        media_type="video"
    ),
    reasoning_clevrer_qa = dict(
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc.json", 
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc_plus.json", 
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_clevrer_mc = dict(
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k.json",  
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k_debug.json",  
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_star = dict(
        anno_path=f"{anno_root_it}/video/reasoning/star/train.json", 
        data_root="p2:s3://star/Charades_v1_480",
        # data_root = 'shddy:s3://yanziang/Charades',
        media_type="video"
    ),
    vqa_ego_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        data_root="pnorm2:s3://egoqa/split_videos",
        media_type="video"
    ),
    vqa_ego_qa_L15 = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train_L15.json", 
        data_root="pnorm2:s3://egoqa/split_videos",
        media_type="video"
    ),
    vqa_tgif_frame_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_tgif_transition_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_webvid_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video",
    ),
    vqa_webvid_qa_30k = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train_30k.json", 
        data_root="pnorm2:s3://webvid2m",
        media_type="video",
    ),
    vqa_perception_train = dict (
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/mc_question_train_forchoice.json',
        data_root='pssd:s3://perception/videos',
        media_type='video'
    ),
    egotask = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/direct_train_qas_convertmc.json',
        data_root='',
        media_type="video",
        video_reader_type='pt'
    ),
    # 2024/04: Add detailed caption data for dynamic resolution finetuning
    caption_sharegpt4v = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train.json", 
        data_root="p2lkc:s3://sharegpt4v/data",
    ),
    caption_sharegpt4o = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/sharegpt4o.json',
        data_root="pssd:s3://perception/videos",
        media_type="video",
    ),
    lsmdc = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/lsmdc.json',
        data_root="pvideo:s3://LSMDC",
        media_type="video",
        video_reader_type="av",
    ),
    caption_sharegpt4v_420k = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train_420k.json", 
        data_root="p2lkc:s3://sharegpt4v/data",
    ),
    caption_sharegptvideo_300k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_300k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_sharegptvideo_600k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_600k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_600k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_240k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_240k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_900k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_900k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_kinetics = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/kinetics_train.json", 
        data_root="",
        media_type="video",
    ),
    caption_vidln_oops = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_oops_L15 = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train_L15.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_ovis = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/ovis_train.json", 
        data_root="p2:s3://ovis/train",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_uvo_sparse = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_sparse_train.json", 
        data_root="sssdct:s3://UVO/uvo_videos_sparse",
        media_type="video",
    ),
    caption_vidln_uvo_dense = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_dense_train.json", 
        data_root="sssdct:s3://UVO/uvo_videos_dense",
        media_type="video",
    ),
    caption_favd = dict(
        anno_path=f"{anno_root_it}/video/caption/favd/train.json", 
        data_root="p2:s3://favd",
        media_type="video",
    ),
    grounding_didemo = dict(
        anno_path=f"{anno_root_it}/video/grounding/didemo/train.json", 
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="video",
    ),
    # text
    conversation_sharegpt = dict(
        anno_path=f"{anno_root_it}/text/sharegpt/train.json", 
        data_root="",
        media_type="text",
    ),
    conversation_openhermes = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train.json", 
        data_root="",
        # "text",
    ),
    conversation_openhermes_200k = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train_200k.json", 
        data_root="",
        # "text",
    ),
    # add docci
    caption_docci = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/train.json", 
        data_root="pnorm2:s3://docci/images",
    ),
    caption_docci_test = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/test.json", 
        data_root="pnorm2:s3://docci/images",
    ),
    # TimeIT
    timeit_ANet = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ANet/train.json", 
        data_root="pnorm2:s3://anet/ANet_320p_fps30/",
        media_type="video"
    ),
    timeit_COIN = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/COIN/train.json", 
        data_root="p2hdd:s3://COIN_320p",
        media_type="video"
    ),
    timeit_DiDeMo = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/DiDeMo/train.json", 
        data_root="sssd:s3://yjsBucket/DiDeMo/",
        media_type="video"
    ),
    timeit_HiREST = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/HiREST/train.json", 
        data_root="zxy_pssd:s3://HiREST/",
        media_type="video"
    ),
    timeit_QuerYD = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/QuerYD/train.json", 
        data_root="zxy_pssd:s3://QuerYD/",
        media_type="video"
    ),
    timeit_SumMe = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/SumMe/train.json", 
        data_root="zxy_pssd:s3://SumMe",
        media_type="video"
    ),

    timeit_TVSum = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/TVSum/train.json", 
        data_root="zxy_pssd:s3://TVSum",
        media_type="video"
    ),
    timeit_ViTT = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ViTT/train.json", 
        data_root="zxy_pssd:s3://ViTT",
        media_type="video"
    ),
    timeit_yttemporal180m = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/yttemporal180m/train.json", 
        data_root="p2hdd:s3://YT-Temporal-180M",
        media_type="video"
    ),
    llava_video1 = dict(
        anno_path=f"/mnt/petrelfs/share_data/yanziang/llava_video/0_30_s_academic_mc_v0_1_qa_processed.json", 
        data_root="pssd:s3://",
        media_type="video"
    ),
)



available_corpus["videochat2_instruction_2024_0204_bbox"] = [
    # available_corpus['tacos_tvg_cg'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['REC_allseeing_train'],
    available_corpus['REC_allseeing_track'],
    # available_corpus['lasot2'],
    available_corpus['REC_ref3_train'],
    # available_corpus['tacos_tvg_cg'],
    # available_corpus['didemo_tvg_cg'],
    # available_corpus['tacos_tvg_cg'],
    # available_corpus['nlq_tvg_cg'],
    # available_corpus['REC_refcocog_google_val'],
    # available_corpus['REC_ref3_train_32k'],
    # available_corpus['samv2_100k'],
    # available_corpus['mevis'],
    # available_corpus['refros_youtube'],
    # available_corpus['mevis_val']
    # available_corpus['lasot2'],
    # available_corpus['REC_allseeing_track'],
    # available_corpus['REC_refcocop_unc_val'],
    # available_corpus['REC_refcoco_unc_val'],
    # available_corpus['REC_refcocog_umd_test'],
    # available_corpus['REC_refcoco_unc_testB'],
    # available_corpus['REC_refcoco_unc_testA'],
    # available_corpus['REC_refcocop_unc_testB'],
    # available_corpus['REC_refcocop_unc_testA'],
    # available_corpus['REC_refcocog_umd_val'],
    # available_corpus['conversation_llava'],
    available_corpus['recvg'],
    # available_corpus['flickr_bbox'],
    # available_corpus['didemo_tvg']

    available_corpus_qa["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    available_corpus_qa["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],

    available_corpus_qa["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    available_corpus_qa["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],

    available_corpus_qa["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    available_corpus_qa["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],

    available_corpus_qa["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    available_corpus_qa["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],
]



# use coco_100k, webvid_80k, add smit
available_corpus["videochat2_instruction_2024_0705"] = [
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['mevis'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    # available_corpus['qvhl_tvg_cg_new'],
    available_corpus['REC_allseeing_train'],
    # available_corpus['REC_allseeing_train'],
    # available_corpus['REC_allseeing_track'],
    available_corpus['REC_allseeing_track'],
    # available_corpus['lasot2'],   
    # available_corpus['recvg'],
    available_corpus['flickr_bbox'],
    available_corpus['REC_ref3_train'],
    # available_corpus['samv2_100k'],
    # available_corpus['samv2_100k'],
    # available_corpus['samv2_100k'],
    # available_corpus['samv2_100k'],
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    available_corpus['ActivityNet_tvg_reason'],
    # available_corpus['mevis_qa'],
    # available_corpus['mevis_qa'],
    # available_corpus['mevis_qa'],
    
    # available_corpus['mevis_qa'],
    # available_corpus['mevis_qa'],
    # available_corpus['mevis_qa'],
    # available_corpus['didemo_tvg_cg_qa'],
    # available_corpus['tacos_tvg_cg_qa'],
    # available_corpus['nlq_tvg_cg_qa'],
    # available_corpus['queryd_tvg_cg_qa'],
    # available_corpus['hirest_tvg_cg_qa'],
    # available_corpus['ActivityNet_tvg_cg_qa'],
    # available_corpus['REC_allseeing_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['samv2_100k_qa'],
    # available_corpus['FT_QVHighlights_v1'],
    # available_corpus['FT_QVHighlights_v1'],
    # available_corpus['FT_QVHighlights_v1'],
    # available_corpus['FT_QVHighlights_v1'],
    # # available_corpus['FT_QVHighlights'],
    # available_corpus['FT_QVHighlights'],
    available_corpus_qa["caption_sharegpt4o"],  # 2
    # available_corpus_qa["lsmdc"], # 1
    available_corpus_qa["vqa_rgbd"],
    available_corpus_qa["vqa_perception_train"],  # 2
    available_corpus_qa["egotask"],
    available_corpus_qa["vqa_tvqa"],  # 2
    # available_corpus[""]
    # available_corpus["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["reasoning_star"],  # 2
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    # available_corpus["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["caption_sharegpt4v"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],   
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],
    available_corpus_qa["caption_sharegpt4v_420k"],
    available_corpus_qa["caption_sharegptvideo_300k"],
    available_corpus_qa["vqa_sharegptvideo_240k"],
    available_corpus_qa["caption_vidln_kinetics"],
    available_corpus_qa["caption_vidln_oops"],
    available_corpus_qa["caption_vidln_ovis"],
    available_corpus_qa["caption_vidln_uvo_sparse"],
    available_corpus_qa["caption_vidln_uvo_dense"],
    available_corpus_qa["caption_favd"],
    available_corpus_qa["conversation_lvis_instruct4v"],
    available_corpus_qa["vqa_ai2d"],
    available_corpus_qa["vqa_chart_qa"],
    available_corpus_qa["vqa_dvqa_80k"],
    available_corpus_qa['llava_video1'],
    # new 0429
    available_corpus_qa["grounding_coco"],
    available_corpus_qa["grounding_vg"],
    available_corpus_qa["grounding_didemo"],
    # available_corpus["conversation_sharegpt"], # pure text
]

available_corpus["videochat2_instruction_2024_0204"] = [
    available_corpus_qa["caption_coco_100k"],
    available_corpus_qa["caption_llava"],
    available_corpus_qa["caption_minigpt4"],
    available_corpus_qa["caption_paragraph_captioning"],
    available_corpus_qa["caption_textcaps"],
    available_corpus_qa["classification_imagenet"],
    available_corpus_qa["classification_coco_itm"],
    available_corpus_qa["conversation_llava"],
    available_corpus_qa["reasoning_clevr"],
    available_corpus_qa["reasoning_visual_mrc"],
    available_corpus_qa["reasoning_llava"],
    available_corpus_qa["vqa_vqav2"],
    available_corpus_qa["vqa_gqa"],
    available_corpus_qa["vqa_okvqa"],
    available_corpus_qa["vqa_a_okvqa"],
    available_corpus_qa["vqa_viquae"],
    available_corpus_qa["vqa_ocr_vqa"],
    available_corpus_qa["vqa_text_vqa"],
    available_corpus_qa["vqa_st_vqa"],
    available_corpus_qa["vqa_docvqa"],
    available_corpus_qa["caption_textvr"],
    available_corpus_qa["caption_videochat"],
    available_corpus_qa["caption_webvid_80k"],
    available_corpus_qa["caption_youcook2"],
    available_corpus_qa["caption_smit"],
    available_corpus_qa["classification_k710"],
    available_corpus_qa["classification_ssv2"],
    available_corpus_qa["conversation_videochat1"],
    available_corpus_qa["conversation_videochat2"],
    available_corpus_qa["conversation_videochatgpt"],
    available_corpus_qa["reasoning_next_qa"],
    available_corpus_qa["reasoning_clevrer_qa"],
    available_corpus_qa["reasoning_clevrer_mc"],
    available_corpus_qa["vqa_ego_qa"],
    available_corpus_qa["vqa_tgif_frame_qa"],
    available_corpus_qa["vqa_tgif_transition_qa"],
    available_corpus_qa["vqa_webvid_qa"],
]



available_corpus['bbox'] = [
    available_corpus['mevis'],
    available_corpus['mevis'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['qvhl_tvg_cg'],
    available_corpus['REC_allseeing_train'],
    available_corpus['REC_allseeing_track'],
    # available_corpus['lasot2'],   
    available_corpus['recvg'],
    available_corpus['flickr_bbox'],
    available_corpus['REC_ref3_train'],
    # available_corpus['samv2_100k'],
    # available_corpus['conversation_llava'],
    # available_corpus["conversation_videochat2"],
    # available_corpus["conversation_videochatgpt"],
    # available_corpus['tacos_tvg_cg'],
    # available_corpus['didemo_tvg_cg'],
    # available_corpus['tacos_tvg_cg'],
    # available_corpus['nlq_tvg_cg'],
    # available_corpus['REC_refcocog_google_val'],
    # available_corpus['REC_ref3_train_32k'],
    # available_corpus['mevis'],
    # available_corpus['refros_youtube'],
    # available_corpus['mevis_val']
    # available_corpus['lasot2'],
    # available_corpus['REC_allseeing_track'],
    # available_corpus['REC_refcocop_unc_val'],
    # available_corpus['REC_refcoco_unc_val'],
    # available_corpus['REC_refcocog_umd_test'],
    # available_corpus['REC_refcoco_unc_testB'],
    # available_corpus['REC_refcoco_unc_testA'],
    # available_corpus['REC_refcocop_unc_testB'],
    # available_corpus['REC_refcocop_unc_testA'],
    # available_corpus['REC_refcocog_umd_val'],
    # available_corpus['conversation_llava'],
    # available_corpus['didemo_tvg']
]

available_corpus['debug'] = [
    # available_corpus['ActivityNet_tvg_reason'],
    # available_corpus['mevis_qa'],
    # available_corpus_qa['llava_video1']
    # available_corpus['ActivityNet_tvg_reason'],
    # available_corpus['didemo_tvg_cg_qa'],
    # available_corpus['tacos_tvg_cg_qa'],
    # available_corpus['nlq_tvg_cg_qa'],
    # available_corpus['queryd_tvg_cg_qa'],
    # available_corpus['hirest_tvg_cg_qa'],
    # available_corpus['ActivityNet_tvg_cg_qa'],
    # available_corpus['REC_allseeing_train_qa'],
    # available_corpus['REC_ref3_train_qa'],
    # available_corpus['samv2_100k_qa']
    # available_corpus['REC_allseeing_track'],
    # available_corpus['REC_allseeing_track'], 
    # available_corpus['recvg'],
    # available_corpus['flickr_bbox'],
    


    # available_corpus_qa["caption_sharegpt4v_420k"],
    # available_corpus_qa["caption_sharegptvideo_300k"],
    # available_corpus_qa["vqa_sharegptvideo_240k"],
    # available_corpus_qa["caption_vidln_kinetics"],
    # available_corpus_qa["caption_vidln_oops"],
    # available_corpus_qa["caption_vidln_ovis"],
    # available_corpus_qa["caption_vidln_uvo_sparse"],
    # available_corpus_qa["caption_vidln_uvo_dense"],
    # available_corpus_qa["caption_favd"],
    # available_corpus_qa["conversation_lvis_instruct4v"],
    # available_corpus_qa["vqa_ai2d"],
    # available_corpus_qa["vqa_chart_qa"],
    # available_corpus_qa["vqa_dvqa_80k"],
    # # new 0429
    # available_corpus_qa["grounding_coco"],
    # available_corpus_qa["grounding_vg"],
    # available_corpus_qa["grounding_didemo"],
    # available_corpus_qa["egotask"],
    # available_corpus_qa["caption_sharegpt4o"],  # 2
    # # available_corpus_qa["lsmdc"], # 有问题
    # # # available_corpus_qa["vqa_rgbd"],
    # available_corpus_qa["vqa_perception_train"],  # 2
    # available_corpus_qa["vqa_tvqa"],
    # available_corpus_qa["caption_sharegpt4v"],
    # available_corpus_qa["reasoning_star"],
    # available_corpus['mevis'],
    # available_corpus['mevis'],
    # available_corpus['REC_allseeing_train'],
    # available_corpus['REC_allseeing_track'],
    # # available_corpus['lasot2'],   
    # available_corpus['recvg'],
    # available_corpus['flickr_bbox'],
    # available_corpus['REC_ref3_train'],
    # available_corpus_qa["caption_sharegpt4o"],
    # available_corpus_qa["lsmdc"],
    # # available_corpus_qa["vqa_rgbd"],
    # available_corpus_qa["vqa_perception_train"],
    # available_corpus_qa["egotask"],
    # available_corpus_qa["vqa_tvqa"],
    # # available_corpus[""]
    # # available_corpus["caption_coco_100k"],
    # available_corpus_qa["caption_llava"],
    # available_corpus_qa["caption_minigpt4"],
    # available_corpus_qa["caption_paragraph_captioning"],
    # available_corpus_qa["caption_textcaps"],
    # available_corpus_qa["classification_imagenet"],
    # available_corpus_qa["classification_coco_itm"],
    # available_corpus_qa["conversation_llava"],
    # available_corpus_qa["reasoning_clevr"],
    # available_corpus_qa["reasoning_visual_mrc"],
    # available_corpus_qa["reasoning_llava"],
    # available_corpus_qa["reasoning_star"],
    # available_corpus_qa["vqa_vqav2"],
    # available_corpus_qa["vqa_gqa"],
    # available_corpus_qa["vqa_okvqa"],
    # available_corpus_qa["vqa_a_okvqa"],
    # available_corpus_qa["vqa_viquae"],
    # available_corpus_qa["vqa_ocr_vqa"],
    # available_corpus_qa["vqa_text_vqa"],
    # available_corpus_qa["vqa_st_vqa"],
    # available_corpus_qa["vqa_docvqa"],
    # available_corpus_qa["caption_textvr"],
    # available_corpus_qa["caption_videochat"],
    # # available_corpus["caption_webvid_80k"],
    # available_corpus_qa["caption_youcook2"],
    # available_corpus_qa["caption_smit"],
    # available_corpus_qa["caption_sharegpt4v"],
    # available_corpus_qa["classification_k710"],
    # available_corpus_qa["classification_ssv2"],
    # available_corpus_qa["conversation_videochat1"],
    # available_corpus_qa["conversation_videochat2"],
    # available_corpus_qa["conversation_videochatgpt"],
    # available_corpus_qa["reasoning_next_qa"],
    # available_corpus_qa["reasoning_clevrer_qa"],
    # available_corpus_qa["reasoning_clevrer_mc"],
    # available_corpus_qa["vqa_ego_qa"],
    # available_corpus_qa["vqa_tgif_frame_qa"],
    # available_corpus_qa["vqa_tgif_transition_qa"],
    # available_corpus_qa["vqa_webvid_qa"],
    # available_corpus_qa["caption_sharegpt4v_420k"],
    # available_corpus_qa["caption_sharegptvideo_300k"],
    # available_corpus_qa["vqa_sharegptvideo_240k"],
    # available_corpus_qa["caption_vidln_kinetics"],
    # available_corpus_qa["caption_vidln_oops"],
    # available_corpus_qa["caption_vidln_ovis"],
    # available_corpus_qa["caption_vidln_uvo_sparse"],
    # available_corpus_qa["caption_vidln_uvo_dense"],
    # available_corpus_qa["caption_favd"],
    # available_corpus_qa["conversation_lvis_instruct4v"],
    # available_corpus_qa["vqa_ai2d"],
    # available_corpus_qa["vqa_chart_qa"],
    # available_corpus_qa["vqa_dvqa_80k"],
    # new 0429
    # available_corpus_qa["grounding_coco"],
    # available_corpus_qa["grounding_vg"],
    # available_corpus_qa["grounding_didemo"],
    # available_corpus["conversation_sharegpt"], # pure text
    # available_corpus_qa["caption_coco_100k"],
    # available_corpus_qa["caption_llava"],
    # available_corpus_qa["caption_minigpt4"],
    # available_corpus_qa["caption_paragraph_captioning"],
    # available_corpus_qa["caption_textcaps"],
    # available_corpus_qa["classification_imagenet"],
    # available_corpus_qa["classification_coco_itm"],
    # available_corpus_qa["conversation_llava"],
    # available_corpus_qa["reasoning_clevr"],
    # available_corpus_qa["reasoning_visual_mrc"],
    # available_corpus_qa["reasoning_llava"],
    # available_corpus_qa["vqa_vqav2"],
    # available_corpus_qa["vqa_gqa"],
    # available_corpus_qa["vqa_okvqa"],
    # available_corpus_qa["vqa_a_okvqa"],
    # available_corpus_qa["vqa_viquae"],
    # available_corpus_qa["vqa_ocr_vqa"],
    # available_corpus_qa["vqa_text_vqa"],
    # available_corpus_qa["vqa_st_vqa"],
    # available_corpus_qa["vqa_docvqa"],
    # available_corpus_qa["caption_textvr"],
    # available_corpus_qa["caption_videochat"],
    # available_corpus_qa["caption_webvid_80k"],
    # available_corpus_qa["caption_youcook2"],
    # available_corpus_qa["caption_smit"],
    # available_corpus_qa["classification_k710"],
    # available_corpus_qa["classification_ssv2"],
    # available_corpus_qa["conversation_videochat1"],
    # available_corpus_qa["conversation_videochat2"],
    # available_corpus_qa["conversation_videochatgpt"],
    # available_corpus_qa["reasoning_next_qa"],
    # available_corpus_qa["reasoning_clevrer_qa"],
    # available_corpus_qa["reasoning_clevrer_mc"],
    # available_corpus_qa["vqa_ego_qa"],
    # available_corpus_qa["vqa_tgif_frame_qa"],
    # available_corpus_qa["vqa_tgif_transition_qa"],
    # available_corpus_qa["vqa_webvid_qa"],
    # available_corpus['REC_refcocog_google_val'],
    # available_corpus['REC_ref3_train'],
    # available_corpus['REC_refcocop_unc_val'],
    # available_corpus['REC_refcoco_unc_val'],
    # available_corpus['REC_refcocog_umd_test'],
    # available_corpus['REC_refcoco_unc_testB'],
    # available_corpus['REC_refcoco_unc_testA'],
    # available_corpus['REC_refcocop_unc_testB'],
    # available_corpus['REC_refcocop_unc_testA'],
    # available_corpus['REC_refcocog_umd_val'],
    # available_corpus['conversation_llava'],
    available_corpus['mevis'],
    
    # available_corpus['didemo_tvg']
]

available_corpus['grounding'] = [
    available_corpus['REC_refcocog_google_val'],
    available_corpus['REC_ref3_train'],
    # available_corpus['REC_refcocop_unc_val'],
    # available_corpus['REC_refcoco_unc_val'],
    # available_corpus['REC_refcocog_umd_test'],
    # available_corpus['REC_refcoco_unc_testB'],
    # available_corpus['REC_refcoco_unc_testA'],
    # available_corpus['REC_refcocop_unc_testB'],
    # available_corpus['REC_refcocop_unc_testA'],
    # available_corpus['REC_refcocog_umd_val'],
    available_corpus['conversation_llava'],
    available_corpus['flickr_bbox'],
    # available_corpus['didemo_tvg']
]
# composed datasets.

available_corpus["webvid10m_image15m_smit"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_avscap_smit"] = [
    available_corpus["internvid2_interleaved_1M_avscap"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_avscap_smit_clip2"] = [
    available_corpus["internvid2_interleaved_1M_avscap_2clip"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_asrcap_smit"] = [
    available_corpus["internvid2_interleaved_1M_asrcap"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_vcap_smit"] = [
    available_corpus["internvid2_interleaved_1M_vcap"],
    available_corpus["smit"],
]

available_corpus["howtointern7M_avscap_smit"] = [
    available_corpus["howtointern7m"],
    available_corpus["smit"]
]

available_corpus["wavcaps_400k"] = [available_corpus["wavcaps_400k_1"], available_corpus["wavcaps_400k_2"], available_corpus["wavcaps_400k_3"], available_corpus["wavcaps_400k_4"]]

available_corpus["avp_debug"] = [available_corpus["ytt_2M_avs_all"], available_corpus["cc3m"], available_corpus["summarized_230613_filtered_30p_random2m"]]

available_corpus["avp_1b"] = [available_corpus["ytt_8M_avs_all"],
                              available_corpus["cc3m"],
                              available_corpus["coco"],
                              available_corpus["vg"],
                              available_corpus["sbu"],
                              available_corpus["cc12m"],
                              available_corpus["intern10mfiltered"]]

available_corpus["ytt_8M_avs_cc3m"] = [available_corpus["ytt_8M_avs"], available_corpus["cc3m"]]
available_corpus["webvid2m_internvid_filter2m_cc3m"] = [available_corpus["webvid"], available_corpus["summarized_230613_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["internvid_filter2m_cc3m"] = [available_corpus["summarized_230613_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["internvid_chat2m_cc3m"] = [available_corpus["summarized_230930_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_cc3m"] = [available_corpus["youtube_5M_avscap_av"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_full_caption_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio_full_caption"], available_corpus["cc3m"]]
available_corpus["internvid2m_filtered_avscap5m_av_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio"], available_corpus["cc3m"], available_corpus["summarized_230613_filtered_30p_random2m"]]

available_corpus["avscap5m_v_cc3m"] = [available_corpus["youtube_5M_avscap_v"], available_corpus["cc3m"]]
available_corpus["vscap5m_v_cc3m"] = [available_corpus["youtube_5M_vscap_v"], available_corpus["cc3m"]]

available_corpus["coco_vg"] = [available_corpus["coco"], available_corpus["vg"]]
available_corpus["in1k_k710"] = [
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m"] = [available_corpus["webvid"], available_corpus["cc3m"]]
available_corpus["webvid10m_cc3m_smit"] = [available_corpus["webvid_10m"], available_corpus["cc3m"], available_corpus["smit"]]
available_corpus["webvid_cc3m_smit"] = [available_corpus["webvid"], available_corpus["cc3m"], available_corpus["smit"]]
available_corpus["webvid2m_jiaer_cc3m"] = [available_corpus["webvid2m_jiaer"], available_corpus["cc3m"]]
available_corpus["cc3m_videocc3m_part"] = [
    available_corpus["cc3m"],
    available_corpus["videocc3m_part"]
]
available_corpus["cc3m_youtube_part"] = [
    available_corpus["cc3m"],
    available_corpus["youtube_part"]
]
available_corpus["cc3m_intern_part"] = [
    available_corpus["cc3m"],
    available_corpus["intern_3m"]
]
available_corpus["cc3m_intern_part_overlap"] = [
    available_corpus["cc3m"],
    available_corpus["intern_3m_overlap"]
]
available_corpus["webvid_cc3m_in1k_k710"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m_k710raw"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["kinetics710_raw"],
]
available_corpus["webvid_14m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid12m_14m"] = [
    available_corpus["webvid"],
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["internv2_10m_avs_14m"] = [
    available_corpus["internv2_10m_avs"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

available_corpus["internv2_10m_avs_smit"] = [
    available_corpus["internv2_10m_avs"],
    available_corpus["smit"]
]

available_corpus["internv2_10m_v_14m"] = [
    available_corpus["internv2_10m_v"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m_smit"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"]
]

available_corpus["webvid10m_3m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
]
available_corpus["simple_17m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["simple_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m_our230415"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["summarized_230415"],
]
available_corpus["our230415_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["summarized_230415"],
]
available_corpus["intern10m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern_10m"],
]
available_corpus["intern10m_new_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern_10m_new"],
]

available_corpus["intern10mfiltered_webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern10mfiltered"]
]

available_corpus["intern10mfiltered_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern10mfiltered"]
]

available_corpus["intern10mfiltered_14m_smit"] = [
    available_corpus["intern10mfiltered"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"]
]

available_corpus["internvid_v1_40m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["internvid_v1_40m"]
]

available_corpus["webvid_fuse_10m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["webvid_fuse_10m"]
]


# ============== for validation =================
available_corpus["msrvtt_1k_test"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video"
)

available_corpus["msrvtt_1k_test_2shot"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    few_shot_config = dict(
        num_shot=2,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)

available_corpus["msrvtt_1k_test_4shot"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    few_shot_config = dict(
        num_shot=4,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)


available_corpus["msrvtt_1k_vs_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_new.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    use_subtitle=True
)

available_corpus["msrvtt_1k_vs_translate_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_translate.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    use_subtitle=True
)

available_corpus["msrvtt_1k_av_test"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["msrvtt_1k_avs_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_new.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    use_subtitle=True
)

available_corpus["msrvtt_1k_avs_translate_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_translate.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    use_subtitle=True
)



available_corpus["ssv2_ret_label_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_train.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_label_val_small"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_val_small.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_template_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_train.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_template_val_small"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_val_small.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["anet_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_train.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/train",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["anet_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_val.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/val",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["anet_ret_val_4shot"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_val.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/val",
    media_type="video",
    few_shot_config = dict(
        num_shot=4,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)

available_corpus["didemo_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_train.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64 # TODO RetTrain其实应该还没支持设置这个参数
)

available_corpus["didemo_ret_av_train"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_train.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64, # TODO RetTrain其实应该还没支持设置这个参数
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["didemo_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_val.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["didemo_ret_test"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_test.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["didemo_ret_vs_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_new.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_vs_translate_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_translate.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_av_test"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_test.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64
)

available_corpus["didemo_ret_avs_translate_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_translate.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_avs_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_new.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["lsmdc_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_train.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_val.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_test_1000"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_test_1000.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["msrvtt_ret_train9k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_train9k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
)

available_corpus["msrvtt_ret_av_train9k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_train9k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["msrvtt_ret_test1k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_test1k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
)

available_corpus["msvd_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_train.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64,
    has_multi_txt_gt=True
)

available_corpus["msvd_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_val.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64
)

available_corpus["msvd_ret_test"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_test.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64
)

available_corpus["ssv2_ret_label_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_train.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)


available_corpus["ssv2_ret_label_val"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_val_small.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt = True
)

available_corpus["ssv2_ret_template_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_train.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)


available_corpus["ssv2_ret_template_val"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_val_small.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)

available_corpus["vatex_en_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share/videointern/annotations/anno_downstream_lkc/vatex_en_ret_train.json",
    data_root="",
    data_root_prefix="sssd:",
    media_type="video",
    has_multi_txt_gt=True
)


available_corpus["vatex_en_ret_val"] = dict(
    anno_path="/mnt/petrelfs/share/videointern/annotations/anno_downstream_lkc/vatex_en_ret_val.json",
    data_root="",
    data_root_prefix="sssd:",
    media_type="video"
)

available_corpus["youcook2_ret_train"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_train.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

available_corpus["youcook2_ret_test"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_test.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

available_corpus["youcook2_ret_merge_subtitle_test"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_test_merge_subtitle.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

# audio-text

available_corpus["audiocaps_ret_train"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/audiocaps/audiocaps_ret_trainval.json",
    data_root="phdd:s3://yjsbucket2/audiocaps/full/",
    media_type="audio",
)

available_corpus["audiocaps_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/audiocaps/audiocaps_ret_test.json",
    data_root="phdd:s3://yjsbucket2/audiocaps/full/",
    media_type="audio",
)


available_corpus["clothov1_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v1/clotho_ret_development.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_development/development/",
    media_type="audio",
)

available_corpus["clothov1_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v1/clotho_ret_evaluation.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_evaluation/evaluation/",
    media_type="audio",
)

available_corpus["clothov2_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v2/clotho_ret_development.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_development/development/",
    media_type="audio",
)

available_corpus["clothov2_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v2/clotho_ret_evaluation.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_evaluation/evaluation/",
    media_type="audio",
)

# available_corpus["debug"] = dict(
#     anno_path=f"/mnt/petrelfs/share_data/lixinhao/smit_debug.json",
#     data_root="pvideo:s3://S-MiT",
#     media_type="video"
# )