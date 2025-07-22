if __name__ == '__main__':
    from ..share_utils.my_easydict import MyEasyDict as edict
    import os
    
    data_dir = '/mnt/petrelfs/share/videointern/annotations'
    anno_root_pt = os.path.join(data_dir, "anno_pretrain")
    max_txt_l = 64
    batch_size = 32
    batch_size_test = 16
    
    config = edict(
        epochs=10,
        num_frames = 4,
        train_file = [
            dict(
                anno_path=f"{anno_root_pt}/cc3m_train.json", 
                data_root="pssd:s3://GCC",
                media_type="image"
            ),
            dict(
                anno_path=f"{anno_root_pt}/webvid_train.json", 
                data_root="pssd:s3://WebVid2M",
                media_type="video"
            ),
        ],
        inputs = edict(
            image_res=224,
            video_input=edict(
                num_frames=4,
                sample_type="rand",
                num_frames_test=4,
                sample_type_test="middle",
                random_aug=False,
            ),
            max_txt_l=edict(image=f"{max_txt_l}", audio=f"{max_txt_l}", video=f"{max_txt_l}", audio_video=f"{max_txt_l}"),
            batch_size=edict(image=f"{batch_size}", audio=f"{batch_size}", video=f"{batch_size}", audio_video=f"{batch_size}"),
            batch_size_test=edict(image=f"{batch_size_test}", audio=f"{batch_size_test}", video=f"{batch_size_test}", audio_video=f"{batch_size_test}"),
        )
    )
    breakpoint()