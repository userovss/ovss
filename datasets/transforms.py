import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomResizedCrop:
    def __init__(self, img_size, img_scale):
        self.img_size = img_size
        self.img_scale = img_scale
    
    def __call__(self, sample):
        crop_params = T.RandomResizedCrop(self.img_size).get_params(sample['image'], self.img_scale, [3/4, 4/3])
        sample['image'] = F.resized_crop(sample['image'], *crop_params, [self.img_size, self.img_size])
        sample['mask'] = F.resized_crop(sample['mask'], *crop_params, [self.img_size, self.img_size])
        # sample['image'] = F.crop(sample['image'], *crop_params)
        # sample['mask'] = F.crop(sample['mask'], *crop_params)
        return sample


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if random.random() < self.flip_prob:
            sample['image'] = F.hflip(sample['image'])
            sample['mask'] = F.hflip(sample['mask'])
        return sample


class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample['image'] = F.to_tensor(sample['image'])
        # sample['mask'] = F.to_tensor(sample['mask'])
        sample['mask'] = torch.from_numpy(sample['mask']).contiguous()
        # print(sample['image'].shape)
        # print(sample['mask'].shape)
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample
