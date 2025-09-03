#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: crop_stereo_img_to_subimgs.py
#@time: 12/15/22 4:12 PM

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor as Pool
from functools import partial
from tqdm import tqdm
import random
np.random.seed(0)


#x4
patch_num_h = 18
patch_num_w = 11
HR_SLIPSIZE = 80
HR_CROPSIZE = 320
scale = 4
LR_SLIPSIZE = HR_SLIPSIZE // scale
LR_CROPSIZE = HR_CROPSIZE // scale

#x2
# patch_num_h = 18
# patch_num_w = 11
# HR_SLIPSIZE = 80
# HR_CROPSIZE = 320
# scale = 2
# LR_SLIPSIZE = HR_SLIPSIZE // scale
# LR_CROPSIZE = HR_CROPSIZE // scale


def crop_and_save(fn,hrframe_folder,lr_frame_foloder,hrpatch_folder,lrpatch_folder):
    # read images
    hrimg = cv2.imread(os.path.join(hrframe_folder, fn), cv2.IMREAD_COLOR)
    lrimg = cv2.imread(os.path.join(lr_frame_foloder, fn), cv2.IMREAD_COLOR)

    H, W, _ = hrimg.shape
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            if i * HR_SLIPSIZE + HR_CROPSIZE > H or j * HR_SLIPSIZE + HR_CROPSIZE > W:
                continue
            else:
                hrpatch = hrimg[i * HR_SLIPSIZE:i * HR_SLIPSIZE + HR_CROPSIZE,
                          j * HR_SLIPSIZE:j * HR_SLIPSIZE + HR_CROPSIZE]
                lrpatch = lrimg[i * LR_SLIPSIZE:i * LR_SLIPSIZE + LR_CROPSIZE,
                          j * LR_SLIPSIZE:j * LR_SLIPSIZE + LR_CROPSIZE]
                savename = fn.split('.')[0].split("_")[0] + '_{}_{}_{}.png'.format(i, j,fn.split('.')[0].split("_")[1])

                cv2.imwrite(os.path.join(hrpatch_folder, savename), hrpatch)
                cv2.imwrite(os.path.join(lrpatch_folder, savename), lrpatch)
def multi_scale_crop_and_save(fn,hrframe_folder,lr_frame_foloder,hrpatch_folder,lrpatch_folder):
    # read images
    hrimg = cv2.imread(os.path.join(hrframe_folder, fn), cv2.IMREAD_COLOR)
    lrimg = cv2.imread(os.path.join(lr_frame_foloder, fn), cv2.IMREAD_COLOR)

    scale = random.uniform(0.8, 1.2)
    hrimg = cv2.resize(hrimg, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    H, W, _ = hrimg.shape
    lrimg = cv2.resize(lrimg, dsize=(int(0.25*W), int(0.25*H)), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    for i in range(patch_num_h):
        for j in range(patch_num_w):
            if i * HR_SLIPSIZE + HR_CROPSIZE > H or j * HR_SLIPSIZE + HR_CROPSIZE > W:
                continue
            else:
                hrpatch = hrimg[i * HR_SLIPSIZE:i * HR_SLIPSIZE + HR_CROPSIZE,
                          j * HR_SLIPSIZE:j * HR_SLIPSIZE + HR_CROPSIZE]
                lrpatch = lrimg[i * LR_SLIPSIZE:i * LR_SLIPSIZE + LR_CROPSIZE,
                          j * LR_SLIPSIZE:j * LR_SLIPSIZE + LR_CROPSIZE]
                savename = fn.split('.')[0].split("_")[0] + 'multiscale_{}_{}_{}.png'.format(i, j,fn.split('.')[0].split("_")[1])

                cv2.imwrite(os.path.join(hrpatch_folder, savename), hrpatch)
                cv2.imwrite(os.path.join(lrpatch_folder, savename), lrpatch)

def x4_data():
    indirhr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/NTIRE2024/Track2-Train/Train/HR"
    indirlr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/NTIRE2024/Track2-Train/Train/LR_x4"

    outdir = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/NTIRE2024/Track2/X4_"
    outdirlr = os.path.join(outdir,"LR_patch")
    outdirhr = os.path.join(outdir,"HR_patch")

    os.makedirs(outdirlr,exist_ok=True)
    os.makedirs(outdirhr,exist_ok=True)

    filenames = os.listdir(indirlr)

    special_crop = partial(crop_and_save,hrframe_folder=indirhr,lr_frame_foloder=indirlr,
                           hrpatch_folder=outdirhr,lrpatch_folder=outdirlr)
    pool = Pool(max_workers=30)
    pbar = tqdm(total=len(filenames))
    results = pool.map(special_crop,filenames)
    for result in results:
        pbar.set_description(result)
        pbar.update()

def x4_data_multiscale():
    indirhr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/HR"
    indirlr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/lr_x4"

    outdir = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/X4_multiscale"
    outdirlr = os.path.join(outdir,"LR_patch")
    outdirhr = os.path.join(outdir,"HR_patch")

    os.makedirs(outdirlr,exist_ok=True)
    os.makedirs(outdirhr,exist_ok=True)

    filenames = os.listdir(indirlr)

    special_crop = partial(multi_scale_crop_and_save,hrframe_folder=indirhr,lr_frame_foloder=indirlr,
                           hrpatch_folder=outdirhr,lrpatch_folder=outdirlr)
    pool = Pool(max_workers=30)
    pbar = tqdm(total=len(filenames))
    results = pool.map(special_crop,filenames)
    for result in results:
        pbar.set_description(result)
        pbar.update()


def x2_data():
    indirhr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/HR"
    indirlr = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/lr_x2"

    outdir = os.path.expanduser("~") + "/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/X2"
    outdirlr = os.path.join(outdir,"LR_patch")
    outdirhr = os.path.join(outdir,"HR_patch")

    os.makedirs(outdirlr,exist_ok=True)
    os.makedirs(outdirhr,exist_ok=True)

    filenames = os.listdir(indirlr)

    special_crop = partial(crop_and_save,hrframe_folder=indirhr,lr_frame_foloder=indirlr,
                           hrpatch_folder=outdirhr,lrpatch_folder=outdirlr)
    pool = Pool(max_workers=10)
    pbar = tqdm(total=len(filenames))
    results = pool.map(special_crop,filenames)
    for result in results:
        pbar.set_description(result)
        pbar.update()

if __name__ == '__main__':
    #x4_data()
    x4_data_multiscale()
    # x2_data()

