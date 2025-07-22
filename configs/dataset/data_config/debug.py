from configs.dataset.data_config.available_corpus import available_corpus


# ========================= data ==========================
train_corpus = "debug" # 
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation

# test_file = dict()

# ========================= input ==========================
num_frames = 16  # 总帧数
num_clips = 1
num_frames_test = 8
num_clips_test = 8
max_txt_l = 64

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        num_clips="${num_clips}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        num_clips_test="${num_clips_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    cgdetr_config=dict(
        q_feat_type = "last_hidden_state",
        ctx_mode = "video_tef",
        clip_len = 1,
        max_windows = 5,
        max_q_l = 100,
        max_v_l = 100000000,
        span_loss_type = "l1",
        use_feature = True,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}")
)

