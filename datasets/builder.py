# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import os.path as osp
import random
import warnings
from functools import partial

import numpy as np
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from timm.data import create_transform
from torchvision import transforms as T
import us
import json
import io
import shared

from sclip.clip import tokenize

from torch.utils.data._utils.collate import default_collate as torch_default_collate
from .noun_parser import WordAugTokenizeWrapper
import random
import torch
from sclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from sclip import tokenize
from datasets.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize


def collate(data):
    data_new = []
    for sample in data:
        if len(sample['category']) >= 2:
            masks = []
            category_copy = sample['category'].copy()
            random.shuffle(sample['category'])
            category = sample['category'][:2]
            for element in category:
                index = category_copy.index(element)
                chosen_mask = sample['mask'][index, :, :]
                masks.append(chosen_mask)
            masks = np.stack(masks)
            data_new.append((sample['image'], category, masks, sample['caption'], sample['npz_path']))
        elif len(sample['category']) == 1:
            category = sample['category'][0]
            mask = sample['mask'][0]
            mask_background = 1 - mask
            masks = np.stack([mask, mask_background])
            data_new.append((sample['image'], [category, "background"], masks, sample['caption'], sample['npz_path']))

    # data = [(sample['image'], sample['category'], sample['mask']) for sample in data]
    output = torch_default_collate(data_new)
    image, category, mask, caption, npz_path = output
    return {
        "image": image,
        "category": category,
        "mask_gt": mask,
        "caption": caption,
        "npz_path": npz_path
    }


class NounNotEnoughError(Exception):
    pass


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config):
    dataset_train = build_dataset(config=config)
    us.dprint("successfully build train dataset")

    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn,
    )

    train_len = len(dataset_train)
    train_nbatches = max(
        1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = data_loader_train.with_epoch(
        train_nbatches).with_length(train_nbatches)

    return dataset_train, data_loader_train


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    if isinstance(exn, NounNotEnoughError):
        return True
    warnings.warn(repr(exn))
    return True

def img_json_npz_decoder(key, value):
    # print(f"image_decoder: {key}") 
    # print(f"image_decoder: {value}")
    if key.endswith(".jpg") or key.endswith(".png"):
        return wds.imagehandler("pil")(key, value)
    elif key.endswith(".json"):
        return json.loads(value)
    elif key.endswith(".npz"):
        return np.load(io.BytesIO(value))['masks']
    elif key.endswith(".txt"):
        return value.decode('utf-8')
    elif key.endswith(".md"):
        return value.decode('utf-8')
    return value

def build_dataset(config):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
    """
    img_transform = build_img_transform(config.img_aug)
    mask_transform = build_mask_transform(config.img_aug)
    # text_transform = TextPreprocess(
    #     num_words=config.num_words, word_type=config.word_type)
    image_mask_transform = build_img_mask_transform(config.img_aug)
    split = "train"
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, "All datasets must be of the same type"

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f"Found {len(cur_tar_file_list)} files for dataset {ds}")
        tar_file_list.extend(cur_tar_file_list)
        total_length += length

    print(f"Found {len(tar_file_list)} files in total for split {split}")
    # dataset = (
    #     wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
    #     .shuffle(40000)  # datapoint-level shuffle
    #     .decode(img_json_npz_decoder, handler=warn_and_continue)
    #     .rename(
    #         image="jpg",
    #         category="json",
    #         mask="npz",
    #         caption='txt',
    #         keep=False,
    #         handler=warn_and_continue,
    #     )
    #     .map_dict(image=img_transform, mask=mask_transform, handler=warn_and_continue)
    #     .with_length(total_length)
    # )
    dataset = (
        wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
        .shuffle(40000)  # datapoint-level shuffle
        .decode(img_json_npz_decoder, handler=warn_and_continue)
        .rename(
            image="jpg",
            category="json",
            mask="npz",
            caption='txt',
            npz_path='md',
            keep=False,
            handler=warn_and_continue,
        )
        .map(image_mask_transform, handler=warn_and_continue)
        .with_length(total_length)
    )

    return dataset


def build_img_transform(config):
    if not config.deit_aug:
        transform = T.Compose(
            [
                # T.RandomResizedCrop(config.img_size, scale=config.img_scale),
                # T.RandomHorizontalFlip(),
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD),
            ]
        )
    else:
        # deit_aug
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != "none" else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
        )

    return transform

def build_mask_transform(config):
    if not config.deit_aug:
        transform = T.Compose(
            [
                # T.RandomResizedCrop(config.img_size, scale=config.img_scale),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Resize(224),
                # T.CenterCrop(224),
            ]
        )
    else:
        # deit_aug
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != "none" else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
        )

    return transform

def build_img_mask_transform(config):
    transform = Compose(
        [
            ToTensor(),
            RandomResizedCrop(config.img_size, config.img_scale),
            RandomHorizontalFlip(0.5),
            Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD)
        ]
    )
    return transform