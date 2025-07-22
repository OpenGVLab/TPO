# import accelerator
from typing import (
    Dict,
    List,
    Optional,
)
from tqdm import tqdm
import json
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import transformers
from transformers.trainer_utils import has_length
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    # ShardedDDPOption,
    has_length,
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


from packaging import version
# from fairscale.optim import OSS

from ..dataset.dataloader import MetaLoader
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)

from src.model.cgdetr_main.utils.tensor_utils import pad_sequences_1d
from src.model.cgdetr_main.cg_detr.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs


import pdb

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


import pyrootutils
import logging
import numpy as np
logger = logging.getLogger(__name__)

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.train.optimization import get_scheduler
from src.dataset.create import get_media_types, create_sampler, create_loader
from src.share_utils.distributed import get_world_size, get_rank, gather_together
from torch.utils.data._utils.collate import default_collate

def collate_hd(batch):
    # Get the keys from the first sample in the batch
    keys = batch[0].keys()
    
    # Initialize a dictionary to store the collated tensors
    collated_batch = {}

    # Iterate over the keys and collate the corresponding tensors
    # print(batch)
    for key in keys:
        # Collect the values for the current key across the batch
        if 'instruction' in key or 'sec' == key:
            # old_values = [sample[key] for sample in batch]
            old_values = []
            for sample in batch:
                if sample is not None:
                    old_values.append(sample[key])
            values = []
            for s in old_values:
                if s is not None:
                    for i in s:
                        values.append(i)
        else:
            values = []
            for sample in batch:
                if sample is not None:
                    values.append(sample[key])
            # values = [sample[key] for sample in batch]
        
        if 'image' == key or 'video' == key:
            collated_values = torch.stack(values)
        elif 'sec' == key or 'vlen' == key or 'fps' == key or 'start_sec'==key or 'end_sec'==key:
            collated_values = default_collate(values)
        elif 'instruction' in key :
            collated_values = ''
        elif 'text_input' == key:
            # logger.info(f'insr:{values}')
            collated_values = default_collate(values)
            # logger.info(f'insr:{values}')
        elif 'frame_indices' == key:
            collated_values = values
        elif 'boxes' in key:
            collated_values = []
            for value in values:
                collated_values.append(torch.tensor(value, dtype=torch.bfloat16))
                # logger.info(f'item box:{collated_values}')
            # collated_values = torch.stack(collated_values)
        elif 'output_boxes' in key:
            collated_values = []
            for value in values:
                collated_values.append(torch.tensor(value, dtype=torch.bfloat16))
        elif 'input_boxes' in key:
            collated_values = torch.tensor(values, dtype=torch.bfloat16)
        elif 'answer' in key:
            collated_values = values
        elif 'gt_masks' in key:
            values = np.array(values)
            collated_values = torch.tensor(values, dtype=torch.bfloat16)
        elif 'size_hw' in key:
            collated_values = np.array(values)
        elif 'path' in key:
            collated_values = values
        elif "meta" == key:
            collated_values = None
        elif "model_inputs" == key:
            collated_values = None
        else:
            collated_values = torch.stack(values)
        
        # Add the collated tensor to the output dictionary
        collated_batch[key] = collated_values
    
    
    if "meta" in keys:
        batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?
        model_inputs_keys = batch[0]["model_inputs"].keys()
        # print(model_inputs_keys)
        batched_data = dict()
        for k in model_inputs_keys:
            if k == "span_labels":
                batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
                continue
            if k in ["saliency_pos_labels", "saliency_neg_labels"]:
                batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
                continue
            if k == "saliency_all_labels":
                pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
                batched_data[k] = torch.tensor(pad_data, dtype=torch.bfloat16)
                continue
            if k == 'qid':
                batched_data[k] = [e["model_inputs"][k] for e in batch]
                continue
            if k == 'vid':
                batched_data[k] = [e["model_inputs"][k] for e in batch]
                continue
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.bfloat16, fixed_length=None)
    else:
        return collated_batch                

    return collated_batch, batch_meta, batched_data


class CustomTrainer(transformers.Trainer):
    """Override the Trainer to avoid IterableDatasetShard"""
    def __init__(self, self_defined_logging_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_defined_logging_dir = self_defined_logging_dir
    
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        
        media_types = get_media_types(train_dataset)
        
        samplers = create_sampler(
            train_dataset, [True] * len(media_types), get_world_size(), get_rank()
        )
        
        if self.args.custom_per_device_train_bs_dict is None:
            batch_size_list = [self.args.per_device_train_batch_size] * len(media_types)
        else:
            batch_size_list = []
            for media_type in media_types:
                batch_size_list.append(self.args.custom_per_device_train_bs_dict.get(media_type, self.args.per_device_train_batch_size))

        dataloaders = create_loader(
            train_dataset,
            samplers,
            batch_size=batch_size_list,
            # num_workers=[6] * len(media_types), # NOTE modify it when memory overflow
            num_workers=[self.args.dataloader_num_workers] * len(media_types), # NOTE modify it when memory overflow
            is_trains=[True] * len(media_types),
            collate_fns=[collate_hd] * len(media_types),
        )
        
        return self.accelerator.prepare(MetaLoader(name2loader=dict(list(zip(media_types, dataloaders)))))


    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = create_sampler(
            eval_dataset, True, get_world_size(), get_rank()
        )
        
        return DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return None


    def compute_loss(self, model, data_all, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if len(data_all) == 3:
            inputs, batch_meta, tvg_data = data_all[0], data_all[1], data_all[2]
        else:
            inputs = data_all
            tvg_data = None
            batch_meta = None


        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        input_labels = inputs.get('labels', None)
        image = inputs.get('image', None)
        video = inputs.get('video', None)
        image_idx = inputs.get("image_idx", None)
        video_idx = inputs.get("video_idx", None)
        instruction = inputs.get('instruction', None)
        output_boxes = inputs.get('output_boxes', None)
        input_boxes = inputs.get('input_boxes', None)
        text_input = inputs.get('text_input', None)
        sec = inputs.get('sec', None)
        vlen = inputs.get('vlen', None)
        frame_indices = inputs.get('frame_indices', None)
        fps = inputs.get('fps', None)
        video_info = {'sec':sec, 'vlen':vlen, 'frame_indices':frame_indices, 'fps':fps}
        start_sec = inputs.get('start_sec', None)
        end_sec = inputs.get('end_sec', None)
        temporal_labels = {"start_sec":start_sec, "end_sec":end_sec}
        gt_masks = inputs.get('gt_masks', None)
        sam_images = inputs.get('sam_images', None)
        size_hw = inputs.get('size_hw', None)
        path = inputs.get('path', None)
        mask_path = inputs.get('mask_path', None)
        if tvg_data == None:
            tvg_inputs, tvg_targets = None, None
        else:
            # import pdb; pdb.set_trace()
            tvg_inputs, tvg_targets = prepare_batch_inputs(tvg_data)

        outputs = model(image=image, video=video, input_ids=input_ids, attention_mask=attention_mask, labels=input_labels, image_idx=image_idx, video_idx=video_idx, \
                        instruction=instruction, output_boxes=output_boxes, input_boxes=input_boxes, text_input=text_input, video_info=video_info, temporal_labels=temporal_labels, \
                            gt_masks=gt_masks, sam_images=sam_images, size_hw=size_hw, path=path, mask_path=mask_path, tvg_inputs=tvg_inputs, tvg_targets=tvg_targets)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            raise NotImplementedError
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # logger.info(f'loss: {outputs}')
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        logger.info(f'comput_metrics: {self.compute_metrics}')
        
        dataset_cls_name = eval_dataset.__class__.__name__
        logger.info(f'evaldataset cls name: {dataset_cls_name}')
        
        if dataset_cls_name in ['VideoCapEvalDataset', 'ImageCapEvalDataset']:
            eval_results = self.evaluation_loop_caption(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        elif dataset_cls_name in ['ITEvalDataset']:
            eval_results = self.evaluation_loop_vqa(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        else:
            eval_results = None

        eval_results_sync = gather_together(eval_results)

        eval_results = []
        for item in eval_results_sync:
            eval_results.extend(item)

        metrics = self.compute_metrics(eval_results=eval_results, eval_dataset=eval_dataset if eval_dataset is not None else self.eval_dataset, output_dir=self.args.output_dir, global_step=self.state.global_step)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    #     # want to save except FullyShardedDDP.
    #     # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    #     # Save model checkpoint
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()

    #     run_dir = self._get_output_dir(trial=trial)
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     # self.save_model(output_dir, _internal_call=True)
    #     assert self.is_deepspeed_enabled
    #     if self.is_deepspeed_enabled:
    #         # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
    #         # config `stage3_gather_16bit_weights_on_model_save` is True
    #         accelerator.get_state_dict(model)
    #         self.model_wrapped.save_checkpoint(output_dir)

    
    def _self_before_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> List:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        
        # if eval is called w/o train init deepspeed here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
            
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        assert batch_size == 1, batch_size
        logger.info(f"  Batch size = {batch_size}")
        # assert len(dataloader.dataset) == len(dataloader), f"{len(dataloader.dataset)} != {len(dataloader)}"

        model.eval()

        return model

    def evaluation_loop_caption(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> List:
        model = self._self_before_evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        model.cuda()
        eval_results = []
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs['input_ids'].to(device=self.args.device)
            attention_mask = inputs['attention_mask'].to(device=self.args.device)
            image, image_idx, video, video_idx = None, None, None, None
            if 'image' in inputs.keys():
                if isinstance(inputs['image'], list):
                    inputs['image'] = inputs['image'][0]
                image_idx = inputs['image_idx'].to(device=self.args.device)
                image = inputs['image'].to(device=self.args.device)
            if 'video' in inputs.keys():
                if isinstance(inputs['video'], list):
                    inputs['video'] = inputs['video'][0]
                video_idx = inputs['video_idx'].to(device=self.args.device)
                video = inputs['video'].to(device=self.args.device)   
            generate_ids = model.generate_caption(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                video=video,
                image_idx=image_idx,
                video_idx=video_idx
            )
            generate_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            # if step % 100 == 0: # We don't need too much logger
            logger.info(f"Input caption: {inputs['caption']} Generated text: {generate_text} Generated Ids: {generate_ids}", )
            eval_results.append({'image_id': inputs['image_id'][0], 'caption': generate_text, 'original_caption': inputs['caption']})
            
        return eval_results

    def evaluation_loop_vqa(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> List:
        model = self._self_before_evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        eval_results = []
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):            
            input_ids = inputs['input_ids'].to(device=self.args.device)
            attention_mask = inputs['attention_mask'].to(device=self.args.device)
            image, image_idx, video, video_idx = None, None, None, None
            if 'image' in inputs.keys():
                if isinstance(inputs['image'], list):
                    inputs['image'] = inputs['image'][0]
                image_idx = inputs['image_idx'].to(device=self.args.device)
                image = inputs['image'].to(device=self.args.device)
            if 'video' in inputs.keys():
                if isinstance(inputs['video'], list):
                    inputs['video'] = inputs['video'][0]
                video_idx = inputs['video_idx'].to(device=self.args.device)
                video = inputs['video'].to(device=self.args.device)   
            
            generate_ids = model.generate_caption(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                video=video,
                image_idx=image_idx,
                video_idx=video_idx
            )
            generate_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            # if step % 10 == 0: # We don't need too much logger
            logger.info(f"'question_id': {inputs['question_id'][0]}, 'question': {inputs['question'][0][0]}, 'answer': {generate_text}, 'original_answer': {inputs['answer'][0][0]}")
            eval_results.append({'question_id': inputs['question_id'][0], 'question': inputs['question'][0][0], 'answer': generate_text, 'original_answer': inputs['answer'][0][0]})
            
        return eval_results
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        if self.args.custom_lr_wd_dict is None:
            return super().create_optimizer()

        logger.info(
            f"use custom lr and wd: {self.args.custom_lr_wd_dict}"
        )

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = [
                dict(params=[], weight_decay=0.0),
                dict(params=[], weight_decay=self.args.weight_decay),
            ]
            for para_name in self.args.custom_lr_wd_dict.keys():
                if self.args.custom_lr_wd_dict[para_name]["wd"] != None:
                    optimizer_grouped_parameters.extend(
                        [
                            dict(
                                params=[],
                                lr=self.args.custom_lr_wd_dict[para_name]["lr"],
                                weight_decay=self.args.weight_decay
                                if self.args.custom_lr_wd_dict[para_name]["wd"] is None
                                else self.args.custom_lr_wd_dict[para_name]["wd"],
                            ),
                        ]
                    )
                else:
                    optimizer_grouped_parameters.extend(
                        [
                            dict(params=[], lr=self.args.custom_lr_wd_dict[para_name]["lr"], weight_decay=0.0)
                        ]
                    )

                # optimizer_grouped_parameters.extend(
                #     [
                #         dict(params=[], lr=self.args.custom_lr_wd_dict[para_name]["lr"], weight_decay=0.0),
                #         dict(
                #             params=[],
                #             lr=self.args.custom_lr_wd_dict[para_name]["lr"],
                #             weight_decay=self.args.weight_decay
                #             if self.args.custom_lr_wd_dict[para_name]["wd"] is None
                #             else self.args.custom_lr_wd_dict[para_name]["wd"],
                #         ),
                #     ]
                # )

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            custom_param_names = list(self.args.custom_lr_wd_dict.keys())
            custom_params = {
                custom_param_name: [] for custom_param_name in custom_param_names
            }

            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue

                is_custom = False
                for idx, custom_param_name in enumerate(custom_param_names):
                    if custom_param_name in n:
                        custom_params[custom_param_name].append(n)
                        # group_idx = (idx + 1) * 2 + int(n in decay_parameters)
                        group_idx = (idx + 2) #+ int(n in decay_parameters)
                        is_custom = True
                        break
                if not is_custom:
                    group_idx = int(n in decay_parameters)

                optimizer_grouped_parameters[group_idx]["params"].append(p)
                logger.info(f"Params: {n} lr: {optimizer_grouped_parameters[group_idx].get('lr', '-')}: wd: {optimizer_grouped_parameters[group_idx]['weight_decay']}")

            logger.info(f"Detected custom_params: {custom_param_names}")

            for idx, group in enumerate(optimizer_grouped_parameters):
                lr = group.get("lr", "-")
                wd = group["weight_decay"]
                # total = [p.numel() for p in group["params"]]
                # logger.info(f"Group {idx}: lr {lr}, wd {wd}, params {total}")
                logger.info(f"Group {idx}: lr {lr}, wd {wd}")

            optimizer_cls, optimizer_kwargs = CustomTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32"
                        )
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        # if self.args.include_num_input_tokens_seen:
        #     logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        if self.state.global_step % 1000 == 0:
            import gc
            gc.collect()
            logger.info('gc collect')
        logger.info(output)

        # import pdb; pdb.set_trace()
        # logger.info(f"CG learning rate:{self.optimizer.optimizer.param_groups[-1]['lr']}")
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        logger.info(f"optimizer: {type(self.optimizer)} {self.optimizer}")
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(self.args.lr_scheduler_type,
                                              optimizer=self.optimizer if optimizer is None else optimizer,
                                              num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                              num_training_steps=num_training_steps,
                                              min_lr_ratio=self.args.min_lr_ratio)
        return self.lr_scheduler

    def evaluation_temporal(self):
            dataloader = self.get_eval_dataloader(self.eval_dataset)
            model = self._self_before_evaluation_loop(dataloader, "Inference")
            model.cuda()

            for step, inputs in tqdm(enumerate(dataloader)):
                input_ids = inputs['input_ids'].to(device=self.args.device)
                attention_mask = inputs.get('attention_mask', None).to(device=self.args.device)
                input_labels = inputs.get('labels', None).to(device=self.args.device)
                image = inputs.get('image', None)
                video = inputs.get('video', None).to(device=self.args.device)
                image_idx = inputs.get("image_idx", None)
                video_idx = inputs.get("video_idx", None).to(device=self.args.device)
                instruction = inputs.get('instruction', None)
                text_input = inputs.get('text_input', None)
                sec = inputs.get('sec', None)
                vlen = inputs.get('vlen', None)
                frame_indices = inputs.get('frame_indices', None)
                fps = inputs.get('fps', None)
                video_info = {'sec':sec, 'vlen':vlen, 'frame_indices':frame_indices, 'fps':fps}
                start_sec = inputs.get('start_sec', None)
                end_sec = inputs.get('end_sec', None)
                temporal_labels = {"start_sec":start_sec, "end_sec":end_sec}

                outputs = model.generate_temporal(image=image, video=video, input_ids=input_ids, attention_mask=attention_mask, \
                                labels=input_labels, image_idx=image_idx, video_idx=video_idx, instruction=instruction, \
                                text_input=text_input, video_info=video_info, temporal_labels=temporal_labels)
                
            return None
