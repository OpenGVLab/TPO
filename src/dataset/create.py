import os
import logging
import pyrootutils
import torch
from torch.utils.data import ConcatDataset, DataLoader
from .resample_concat_dataset import ResampleConcatDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..share_utils.my_easydict import MyEasyDict as edict

from ..share_utils.py_config_utils import Config, eval_dict_leaf
from .sampler import StatefulDistributedSampler
from .dataloader import MetaLoader, MetaLoader_rs, MetaLoader_rs2 # NOTE keep it
from src.dataset.it_dataset_mistral_new import ITTrainDataset_mistral_new
from src.share_utils.special_token import load_image_list, load_video_list
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
logger = logging.getLogger(__name__)


def get_media_types(datasources):
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasources
    ]
    return media_types

def get_media_type(dataset_config):
    # import pdb;pdb.set_trace()
    # if isinstance(dataset_config, list):
    #     dataset_config = dataset_config[0]
    return dataset_config.get('media_type', 'image')

def get_dataset_cls(dataset_type, media_type, data_cfg):
    if dataset_type in ["pt_train"]:
        
        raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    elif dataset_type in ["it_train_bbox"]:
        dataset_cls = ITTrainDataset_mistral_new
    else:
        raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    
    print(f"\033[31m dataset_type: {dataset_type} media_type: {media_type} dataset_cls: {dataset_cls}\033[0m")
    logger.info(f"dataset_type: {dataset_type} media_type: {media_type} dataset_cls: {dataset_cls}")

    return dataset_cls

def get_train_transform(config, train_file, mode = 'resize'):
    # resize mode (RandomResizedCrop). padding mode (black padding). hd mode (no resize)
    # 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    padding_resize_transform = transforms.Resize(224)
    # pad_transform = transforms.Pad((0, 0, 0, 0), fill=0, padding_mode='constant')  

    aug_transform = transforms.Lambda(lambda x: x)
    if mode == 'resize' :
        train_transform = transforms.Compose(
            [
                aug_transform,
                transforms.RandomResizedCrop(
                    224,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                type_transform,
                normalize,
            ]
        )
    elif mode == 'hd' :
        train_transform = transforms.Compose(
            [
                aug_transform,
                type_transform,
                normalize,
            ]
        )
    elif mode == 'padding' :
        train_transform = transforms.Compose(
            [
                aug_transform,
                type_transform,
                normalize,
                # transforms.Resize((224, 224))
                # padding_resize_transform,
                # pad_transform
            ]
        )
    elif mode == 'resize_noflip' :
        train_transform = transforms.Compose(
            [
                aug_transform,
                type_transform,
                normalize,
                transforms.Resize((224, 224))
                # padding_resize_transform,
                # pad_transform
            ]
        )
    else:
        raise NotImplementedError
    
    return train_transform

def get_test_transform(config, test_file):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform

def create_default_config():
    data_dir = '/mnt/petrelfs/share/videointern/annotations'
    anno_root_pt = os.path.join(data_dir, "anno_pretrain")
    max_txt_l = 64

    config = edict(
        num_frames = 8,
        inputs = edict(
            image_res=224,
            video_input=edict(
                num_frames=8,
                sample_type="rand",
                num_frames_test=8,
                sample_type_test="middle",
                random_aug=False,
            ),
            max_txt_l=edict(image=f"{max_txt_l}", audio=f"{max_txt_l}", video=f"{max_txt_l}", audio_video=f"{max_txt_l}")
        )
    )
    config['train_file'] = [
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
        dict(
            anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
            data_root="pvideo:s3://S-MiT",
            media_type="video"
        )
    ]

    config['test_file'] = {
            'coco_test':dict(
                anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_test_vast_only_first.json',
                # anno_path=f"/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json", 
                data_root="pssd:s3://coco_caption",
                media_type="image"),
            # 'msrvtt-test': dict(
            #     anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
            #     data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
            #     media_type="video"
            # )
    }
    return config

def create_dataset(dataset_type, config_path=None, tokenizer=None):
    ##########################################################
    # Shared setting for all datasets
    # import hydra
    if config_path is None:
        config = create_default_config()
    else:
        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)
    
    if config.inputs.get('video_input', None) is not None:
        video_reader_type = config.inputs.video_input.get("video_reader_type", "decord")
        video_only_dataset_kwargs_train = dict(
            video_reader_type=video_reader_type,
            sample_type='rand',
            num_frames=config.num_frames,
            num_tries=10,  # false tolerance
        )
        video_only_dataset_kwargs_eval = dict(
            video_reader_type=video_reader_type,
            sample_type=config.inputs.video_input.sample_type_test,
            num_frames=config.inputs.video_input.num_frames_test,
            num_tries=1,  # we want to have predictions for all videos
        )
    else:
        logger.warn("Make sure that you don't need video input!!!")
    if config.inputs.get('audio_input', None) is not None:
        audio_reader_type = config.inputs.audio_input.get("audio_reader_type", "torchaudio")
        audio_only_dataset_kwargs_train = dict(
            audio_reader_type=audio_reader_type,
            audio_sample_rate=config.inputs.audio_input.get('audio_sample_rate', 16000),
            max_audio_length=config.inputs.audio_input.get('max_audio_length', 10),
            num_tries=10,
        )
        audio_only_dataset_kwargs_eval = dict(
            audio_reader_type=audio_reader_type,
            audio_sample_rate=config.inputs.audio_input.get('audio_sample_rate_test', 16000),
            max_audio_length=config.inputs.audio_input.get('max_audio_length', 10),
            num_tries=1,
        )
    else:
        logger.warn("Make sure that you don't need audio input!!!")
    print(f'dataset_type:{dataset_type}')
    if dataset_type == 'it_train_bbox':
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file, dict) else config.train_file
        )
        # import pdb;pdb.set_trace()

        train_media_types = sorted(list({get_media_type(e) for e in train_files}))
        # 把同media_type的dataset拼成一个ConcatDataset，没懂为什么要排序，但是不敢删
        train_datasets = []
        for m in train_media_types:
            
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            sample_weights = []
            for train_file in _train_files:
                dataset_cls = get_dataset_cls(dataset_type=dataset_type, media_type=m, data_cfg=train_file)
                if m == "audio":
                    train_transform = None
                elif 'tracking' in m:
                    train_transform = get_train_transform(config, train_file, mode='resize_noflip')
                elif m == "TVG" or m == "image" or m == "video" or m=="TVG+HL" or m =='TVG_qa' or m == 'TVG_reason':
                    train_transform = get_train_transform(config, train_file, mode='resize')
                else:
                    train_transform = get_train_transform(config, train_file, mode='padding')
                dataset_kwargs = dict(
                    ann_file=train_file, # NOTE 靠这个dict传参
                    transform=train_transform)

                # if m == "audio_video":
                #     dataset_kwargs.update(video_only_dataset_kwargs_train)
                #     dataset_kwargs.update(audio_only_dataset_kwargs_train)
                # elif m in ["video", "interleaved_video"]:
                #     dataset_kwargs.update(video_only_dataset_kwargs_train)
                # elif m == 'audio':
                #     dataset_kwargs.update(audio_only_dataset_kwargs_train)
                if m in load_video_list:
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                elif m not in load_image_list and m != 'text':
                    raise NotImplementedError(m)
                logger.info(f"dataset_type={dataset_type}, train_file={train_file}")
                logger.info(dataset_kwargs)
                logger.info('train_transform:')
                logger.info(str(train_transform))

                datasets.append(dataset_cls(tokenizer=tokenizer, cgdetr_config=config.inputs.cgdetr_config,  **dataset_kwargs))
                # if isinstance(train_file, list):
                #     train_file = train_file[0]
                sample_weights.append(train_file.get("sample_weight", 1))
                # assert train_file.get("sample_weight", 1) == 1, train_file

            if sum(sample_weights) > len(sample_weights):
                logger.info(f'Use ResampleConcatDataset for {m}, sample_weights={sample_weights}')
                dataset = ResampleConcatDataset(datasets, sample_weights=sample_weights)
            else:
                logger.info(f'Use ConcatDataset for {m}')
                dataset = ConcatDataset(datasets)

            train_datasets.append(dataset)

        return train_datasets
    elif dataset_type == "pt_train" or dataset_type == 'it_train':
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file, dict) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))
        # 把同media_type的dataset拼成一个ConcatDataset，没懂为什么要排序，但是不敢删
        train_datasets = []
        for m in train_media_types:
            
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            sample_weights = []
            for train_file in _train_files:
                dataset_cls = get_dataset_cls(dataset_type=dataset_type, media_type=m, data_cfg=train_file)
                if m == "audio":
                    train_transform = None
                else:
                    train_transform = get_train_transform(config, train_file)
                dataset_kwargs = dict(
                    ann_file=train_file, # NOTE 靠这个dict传参
                    transform=train_transform)

                if m == "audio_video":
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    dataset_kwargs.update(audio_only_dataset_kwargs_train)
                elif m in ["video", "interleaved_video"]:
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                elif m == 'audio':
                    dataset_kwargs.update(audio_only_dataset_kwargs_train)
                elif m != 'image':
                    raise NotImplementedError(m)
                logger.info(f"dataset_type={dataset_type}, train_file={train_file}")
                logger.info(dataset_kwargs)
                logger.info('train_transform:')
                logger.info(str(train_transform))

                datasets.append(dataset_cls(tokenizer=tokenizer, **dataset_kwargs))
                sample_weights.append(train_file.get("sample_weight", 1))
                # assert train_file.get("sample_weight", 1) == 1, train_file

            if sum(sample_weights) > len(sample_weights):
                logger.info(f'Use ResampleConcatDataset for {m}, sample_weights={sample_weights}')
                dataset = ResampleConcatDataset(datasets, sample_weights=sample_weights)
            else:
                logger.info(f'Use ConcatDataset for {m}')
                dataset = ConcatDataset(datasets)

            train_datasets.append(dataset)

        return train_datasets

    elif dataset_type == "ret_train":
        assert isinstance(config.train_file, dict), config.train_file
        train_transform = get_train_transform(config, config.train_file)
        dataset_cls = get_dataset_cls(dataset_type=dataset_type,
                                      media_type=config.train_file.media_type,
                                      data_cfg=config.train_file)
        if config.train_file.media_type in ["video", "interleaved_video"]:
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=train_transform)
            dataset_kwargs.update(video_only_dataset_kwargs_train)
        elif config.train_file.media_type == 'audio':
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=None) # NOTE audio目前没有用数据增强
            dataset_kwargs.update(audio_only_dataset_kwargs_train)
        elif config.train_file.media_type == 'audio_video':
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=train_transform)
            dataset_kwargs.update(video_only_dataset_kwargs_train)
            dataset_kwargs.update(audio_only_dataset_kwargs_train)
        elif config.train_file.media_type == 'image':
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=train_transform
            )
        
        logger.info(f"dataset_type={dataset_type}, train_file={config.train_file}")
        logger.info(dataset_kwargs)
        logger.info('train_transform:')
        logger.info(str(train_transform))

        return [dataset_cls(**dataset_kwargs)]
    
    elif dataset_type == "qa_train":
        assert type(config.train_file) is dict, f"assuming single train media type but get {config.train_file}"
        
        media_type = get_media_type(config.train_file[0])  # assuming single train media type
        if media_type == "audio":
            train_transform = None
        else:
            train_transform = get_train_transform(config, config.train_file)
            
        dataset_cls = get_dataset_cls(dataset_type=dataset_type,
                                      media_type=media_type,
                                      data_cfg=config.train_file)
        dataset_kwargs = dict(
            ann_file=config.train_file, transform=train_transform, eos=tokenizer.eos_token, mode="train"
        )
        if media_type in ["video", "interleaved_video"]:
            dataset_kwargs.update(video_only_dataset_kwargs_train)
        train_dataset = dataset_cls(**dataset_kwargs)

        logger.info(f"dataset_type={dataset_type}, train_file={config.train_file}")
        logger.info(dataset_kwargs)
        logger.info('train_transform:')
        logger.info(str(train_transform))

        return train_dataset
    
    elif dataset_type in ["pt_eval", "ret_eval", "qa_eval", 'cap_eval', 'vqa_eval']:
        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        for name, data_cfg in config.test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_names.append(name)
            test_transform = get_test_transform(config, data_cfg)

            if dataset_type == "qa_eval" or (dataset_type == "pt_eval" and "_qa_" in name): # TODO 也许有更优雅的方式
                test_dataset_cls = get_dataset_cls(dataset_type='qa_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                dataset_kwargs = dict(
                    ann_file=data_cfg,
                    transform=test_transform,
                    eos=tokenizer.eos_token,
                    mode="eval",
                    answer_list=config.answer_list,
                )
                if media_type in ["video", "interleaved_video"]:
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                else:
                    raise NotImplementedError(f"media_type={media_type}")
            
            elif dataset_type == 'vqa_eval':
                test_dataset_cls = get_dataset_cls(dataset_type='vqa_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                dataset_kwargs = dict(
                    ann_file=data_cfg,
                    transform=test_transform,
                    tokenizer=tokenizer
                )
                if media_type in ["video"]:
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                
            elif dataset_type == 'cap_eval':
                test_dataset_cls = get_dataset_cls(dataset_type='cap_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                dataset_kwargs = dict(
                    ann_file=data_cfg,
                    transform=test_transform,
                    tokenizer=tokenizer
                )
                
                if media_type in ["video", "interleaved_video"]:
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                
            else: # ret
                test_dataset_cls = get_dataset_cls(dataset_type='ret_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                if media_type in ["video", "interleaved_video"]:
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=test_transform
                    )
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                elif media_type == "audio":
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=None)
                    dataset_kwargs.update(audio_only_dataset_kwargs_eval)
                elif media_type == 'audio_video':
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=test_transform)
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                    dataset_kwargs.update(audio_only_dataset_kwargs_eval)
                elif media_type == 'image':
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=test_transform
                    )
                else:
                    raise NotImplementedError(f"media_type={media_type}")
                
            logger.info(f"dataset_type={dataset_type}, test_file={data_cfg}")
            logger.info(dataset_kwargs)
            logger.info('test_transform:')
            logger.info(str(test_transform))

            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        return test_datasets
      
    else:
        raise NotImplementedError(f"dataset_type={dataset_type}")

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    if type(datasets) is list:
        assert len(datasets) == len(shuffles)
        samplers = []
        for dataset, shuffle in zip(datasets, shuffles):
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
            )
            samplers.append(sampler)
        return samplers
    else:
        return torch.utils.data.DistributedSampler(
                datasets, num_replicas=num_tasks, rank=global_rank, shuffle=shuffles
            )

def create_stateful_sampler(datasets, batch_size):
    samplers = []
    for dataset, bs in zip(datasets, batch_size):
        sampler = StatefulDistributedSampler(dataset, batch_size=bs)
        samplers.append(sampler)
    return samplers

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
            pin_memory = True
            persistent_workers = True if n_worker > 0 else False
        else:
            shuffle = False
            drop_last = False
            pin_memory = False
            persistent_workers = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
        loaders.append(loader)
    return loaders

def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data

if __name__ == '__main__':
    print('success')