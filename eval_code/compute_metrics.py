from src.dataset.it_dataset_mistral_eval import ITTrainDataset_mistral_eval

Myd = ITTrainDataset_mistral_eval(
    ann_file = val_list1[gpu_id],   #gpu_id
    transform=get_train_transform(config='', train_file='',mode='resize'),
    tokenizer=tokenizer,
    num_frames = 16,
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
)

def eval_epoch_post_processing(submission, gt_data, submission_path):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    # submission_path = os.path.join(save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=False, 
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug,
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths
    
metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch_post_processing(submission, Myd.anno, submission_path)

if opt.eval_split_name == 'val':
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
if metrics_nms is not None:
    logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))