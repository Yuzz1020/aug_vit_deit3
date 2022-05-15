# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
from re import S
import torch
import torch.nn.functional as F
import torch.nn as nn

import time
import matplotlib.pyplot as plt

from torchvision import transforms

from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF

def construct_transform(args):
    trans_list = [] 
    if 'jit' in args.patch_aug:
        trans_list.append(transforms.ColorJitter(args.patch_color_jitter))
    # if 'gray' in args.patch_aug or 'blur' in args.patch_aug or 'solar' in args.patch_aug:
    T_aug_list = []
    if 'blur' in args.patch_aug:    
        T_aug_list.append(transforms.GaussianBlur(7, sigma=(0.1, 2))) 
    if 'gray' in args.patch_aug:
        T_aug_list.append(transforms.Grayscale(3))
    if 'solar' in args.patch_aug:
        T_aug_list.append(transforms.RandomSolarize(0.5, p=1.0))
    trans_list.append(transforms.RandomChoice(T_aug_list))
    
    # if 'cut' in args.patch_aug:
        # trans_list.append(T.)
    trans_list = transforms.Compose(trans_list)
    return trans_list
    
        

def apply_attn_augment(samples, attn, args, patch_num=14, patch_size=16):
    # TODO: check mapping function and which patch is removed in each image 
    # convert samples to patches 
    B, C, H, W = samples.shape
    B, P = attn.shape 
    assert P == patch_num*patch_num

    patches = samples.reshape(B, C, patch_num, patch_size, patch_num, patch_size).permute(0,2,4,1,3,5)
    patches = patches.reshape(B, patch_num*patch_num, 3, patch_size, patch_size)
    
    # calculate score
    if 'gs' in args.mapping:
        if 'inv' in args.mapping:
            score = F.gumbel_softmax(-attn, tau=0.1)
        else:
            score = F.gumbel_softmax(attn, tau=0.1)
    elif 'linear' in args.mapping:
        if 'inv' in args.mapping:
            score = -attn 
        else:
            score = attn 
    elif 'uniform' in args.mapping:
        score = torch.rand(b, patch_num*patch_num)
    else:
        raise NotImplementedError

    # generate mask for each sample
    for b in range(B):
        idx = np.linspace(0,P-1, P).astype(int)
        if args.rand_patch:
            mask_id = np.random.choice(idx, int(P*args.patch_prob), replace=False, p=(score[b]/score[b].sum()).numpy()) # selec high score with high prob 
        else: 
            mask_id = np.argpartition(score[b], int(P*(1-args.patch_prob)))[-int(P*args.patch_prob):] # select high score 
            ### for debug
            # no_mask = np.argpartition(score[b], int(P*(1-args.patch_prob)))[:-int(P*args.patch_prob)]
        # print(len(mask_id))
        # generate transform
        ### for debug 
        # patches[b, no_mask] = torch.zeros_like(patches[b,no_mask])
        transform_func = construct_transform(args)
        if args.same_aug:
            patches[b,mask_id] = transform_func(patches[b, mask_id])
        else:
            for m_id in mask_id:
                patches[b, m_id] = transform_func(patches[b, m_id])
    
    samples = patches.reshape(B, patch_num, patch_num, 3, patch_size, patch_size).permute(0,3,1,4,2,5).reshape(B, C, H, W)
        # if 'cut' in args.patch_aug:
    ### for debug 
    # to_pil = transforms.ToPILImage()
    # for i in range(samples.shape[0]):
    #     plt.figure()
    #     plt.imshow(to_pil(samples[i].cpu()))
    #     plt.savefig('{}.png'.format(time.time()))
    return samples 
    # if args.combine_aug:
        # in one image, do all three augments 

    

def denorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.tensor(std).cuda()
    mean = torch.tensor(mean).cuda()
    img.mul_(std[None,:,None, None]).add_(mean[None,:,None,None])
    return img 

def norm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.tensor(std).cuda()
    mean = torch.tensor(mean).cuda()
    img.sub_(mean[None,:,None,None]).div_(std[None,:,None,None])   
    return img 

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
 
    
    
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
    
    
def new_data_aug_generator(args = None):
    img_size = args.input_size
    remove_random_resized_crop = args.src
    # named_loss = args.named_loss
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    # secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
    #                                           Solarization(p=1.0),
    #                                           GaussianBlur(p=1.0)])]
   
    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl + final_tfl)
    # return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
