# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.transforms import stereo_augment, stereo_random_crop_hw,stereo_single_random_crop_hw,random_scale
from basicsr.utils import FileClient, imfrombytes, img2tensor
import os
import numpy as np
from basicsr.utils.registry import DATASET_REGISTRY
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data
from torchvision import transforms

from PIL import Image


def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    # rgb = img.view(3, -1).permute(1, 0)
    return coord


@DATASET_REGISTRY.register(suffix='basicsr')
class ContinuousStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(ContinuousStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'

        self.gt_files = os.listdir(self.gt_folder)
        self.gt_files = list(filter(lambda file: file.find('L')!=-1, self.gt_files ))

        self.nums = len(self.gt_files)
        self.patch_size = opt['patch_size'] if 'patch_size' in opt else None
        self.min_scale = opt['min_scale'] if 'min_scale' in opt else 1
        self.max_scale = opt['max_scale'] if 'max_scale' in opt else 4
        self.random_sample = opt['random_sample'] if 'random_sample' in opt else False
    
    def resize_fn(self, img, size):
        img = torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        resized_tensor = transforms.ToTensor()(
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))
        return cv2.cvtColor(resized_tensor.permute(1, 2, 0).numpy(),cv2.COLOR_RGB2BGR)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        L_filename = self.gt_files[index]
        R_filename = L_filename.replace('L', 'R')

        gt_path_L = os.path.join(self.gt_folder, L_filename)
        gt_path_R = os.path.join(self.gt_folder, R_filename)

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            # img_gt_L = cv2.imread(gt_path_L)/255.0
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            # img_gt_R = cv2.imread(gt_path_R)/255.0
            # H, W ,C
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        s = random.uniform(self.min_scale, self.max_scale)
        if self.patch_size is not None:
            h_lr, w_lr = self.patch_size
            h_hr, w_hr = round(h_lr*s), round(w_lr * s)
        else:
            h_lr, w_lr = math.floor(img_gt_L.shape[0] / s + 1e-9), math.floor(img_gt_L.shape[1] / s + 1e-9)
            h_hr, w_hr = round(h_lr*s), round(w_lr * s)
        x0 = random.randint(0, img_gt_L.shape[0] - h_hr)
        y0 = random.randint(0, img_gt_L.shape[1] - w_hr)
        crop_hr_L = img_gt_L[x0: x0 + h_hr, y0: y0 + w_hr, :]
        crop_hr_R = img_gt_R[x0: x0 + h_hr, y0: y0 + w_hr, :]
        crop_lr_L = self.resize_fn(crop_hr_L, (h_lr, w_lr))
        crop_lr_R = self.resize_fn(crop_hr_R, (h_lr, w_lr))
    
        img_gt = np.concatenate([crop_hr_L, crop_hr_R], axis = -1)   # [H,W,C]
        img_lq = np.concatenate([crop_lr_L, crop_lr_R], axis = -1)
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # flip, rotation
            imgs, status = stereo_augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt, img_lq = imgs

        img_gt = torch.cat([img2tensor(img_gt[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_gt[:,:,3:],bgr2rgb=True,float32=True)], dim=0)
        img_lq = torch.cat([img2tensor(img_lq[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_lq[:,:,3:],bgr2rgb=True,float32=True)], dim=0)
        hr_coord = make_coord(img_gt.shape[-2:])

        # sequential sampling
        if self.random_sample:
            sample_x = torch.randint(0, h_hr, (h_lr,))
            sample_y = torch.randint(0, w_hr, (w_lr,))
            grid_x, grid_y = torch.meshgrid(sample_x, sample_y, indexing='ij')  # h_lr × w_lr
            hr_coord = hr_coord[grid_x, grid_y]  # h_lr × w_lr × 2
            img_gt = img_gt[:, grid_x, grid_y]   # C × h_lr × w_lr
        else:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            hr_coord = hr_coord[x0: x0 + h_lr, y0: y0 + w_lr,:]
            img_gt = img_gt[:, x0: x0 + h_lr, y0: y0 + w_lr]



        cell = torch.ones_like(hr_coord)
        cell[ :, :, 0] = cell[ :, :, 0] * 2 / img_gt.shape[-2]
        cell[ :, :, 1] = cell[ :, :, 1] * 2 / img_gt.shape[-1]
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'coord': hr_coord,
            'cell': cell,
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


@DATASET_REGISTRY.register(suffix='basicsr')
class TestContinuousStereoImageDataset(data.Dataset):
    '''
        Testdata set for filckr1024 KITTI2012 KITTI2015 Middlebury
    '''
    def __init__(self, opt):
        super(TestContinuousStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)
        self.patch_size = opt['patch_size'] if 'patch_size' in opt else None
        self.min_scale = opt['min_scale'] if 'min_scale' in opt else 1
        self.max_scale = opt['max_scale'] if 'max_scale' in opt else 4

    def resize_fn(self, img, size):
        img = torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        resized_tensor = transforms.ToTensor()(
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))
        return cv2.cvtColor(resized_tensor.permute(1, 2, 0).numpy(),cv2.COLOR_RGB2BGR)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        s = random.uniform(self.min_scale, self.max_scale) if self.opt['scale'] is None else self.opt['scale']
        if self.patch_size is not None:
            h_lr, w_lr = self.patch_size
            h_hr, w_hr = round(h_lr*s), round(w_lr * s)
        else:
            h_lr, w_lr = math.floor(img_gt_L.shape[0] / s + 1e-9), math.floor(img_gt_L.shape[1] / s + 1e-9)
            h_hr, w_hr = round(h_lr*s), round(w_lr * s)
        x0 = random.randint(0, img_gt_L.shape[0] - h_hr)
        y0 = random.randint(0, img_gt_L.shape[1] - w_hr)
        crop_hr_L = img_gt_L[x0: x0 + h_hr, y0: y0 + w_hr, :]
        crop_hr_R = img_gt_R[x0: x0 + h_hr, y0: y0 + w_hr, :]
        crop_lr_L = self.resize_fn(crop_hr_L, (h_lr, w_lr))
        crop_lr_R = self.resize_fn(crop_hr_R, (h_lr, w_lr))
    
        img_gt = np.concatenate([crop_hr_L, crop_hr_R], axis = -1)   # [H,W,C]
        img_lq = np.concatenate([crop_lr_L, crop_lr_R], axis = -1)

        img_gt = torch.cat([img2tensor(img_gt[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_gt[:,:,3:],bgr2rgb=True,float32=True)], dim=0)
        img_lq = torch.cat([img2tensor(img_lq[:,:,:3],bgr2rgb=True,float32=True), img2tensor(img_lq[:,:,3:],bgr2rgb=True,float32=True)], dim=0)
        hr_coord = make_coord(img_gt.shape[-2:])

        cell = torch.ones_like(hr_coord)
        cell[:, :, 0] = cell[:, :, 0] * 2 / img_gt.shape[-2]
        cell[:, :, 1] = cell[:, :, 1] * 2 / img_gt.shape[-1]

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'coord': hr_coord,
            'cell': cell,
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums
    