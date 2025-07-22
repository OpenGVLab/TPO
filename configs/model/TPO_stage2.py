num_frames = 16

custom_lr_wd_dict=dict(
    lm={'lr':2e-5, 'wd':None},
    vision_encoder={'lr':2e-5, 'wd':None},
    loc_encoder={'lr':2e-5, 'wd':None},
    loc_decoder={'lr':2e-5, 'wd':None},
    box_token={'lr':2e-5, 'wd':None},
    track_embed={'lr':2e-5, 'wd':None},
    cg_model={'lr':2e-5, 'wd':None},
    qformer={'lr':2e-5, 'wd':None},
    track_token={'lr':2e-5, 'wd':None},
    track_embed_decode2={'lr':2e-5, 'wd':None},
    temporal_token={'lr':2e-5, 'wd':None},
    temporal_embed={'lr':2e-5, 'wd':None},
)

model = dict(
    model_cls="MultiModalLLM_PT",
    vision_encoder=dict(
        name="vit_l14",
        img_size=224, 
        patch_size=16, 
        d_model=1024,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames="${num_frames}",
        tubelet_size=1,
        use_checkpoint=False,
        checkpoint_num=0,
        pretrained=None,
        return_index=-2,
        vit_add_ln=True,
    ),
    bridge=dict(
        name='qformer',
        num_query_token=32,
        extra_num_query_token=64,
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_drop_path_rate=0.2,

    ),
    llm=dict(
        name='mistral_7b',
        pretrained_llm_path='/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2',
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        hidden_size = 4096
    ),
    loss=dict(
        use_vision_regression_loss=False,
        add_bbox_loss=True,
        use_temporal_loss=True,
        use_mask_loss = True
    ),
    pretrained_paths=dict(
        # pretrained_vit_qformer_path='/mnt/petrelfs/wangchenting/multimodalllm/model.pth',
        # videochat2_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/umt-l.pth'
    ),
    use_flash_attention=True,
    freeze_vision_encoder=False,
    freeze_bridge=False,
    freeze_llm=False,
    freeze_lora=False
)