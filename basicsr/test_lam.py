import logging
import torch
from os import path as osp
import os
import numpy as np
import math

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.archs import build_network
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

from lam.utils import prepare_clips, vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, \
    make_pil_grid, pil_to_cv2, cv2_to_pil, PIL2Tensor, Tensor2PIL, prepare_stereo, gini
from lam.core import attr_grad
from lam.SSR_BackProp import GaussianBlurPath
from lam.SSR_BackProp import attribution_objective, Path_gradient, Path_gradient_continuous
from lam.SSR_BackProp import saliency_map_PG as saliency_map
import cv2


def main(root_path, imgpath, img_path2, w=50, h=300, window_size=32, fold=50, sigma=1.2, l=9, alpha=0.3, zoomfactor=4, kde=False):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    
    # create model
    model = build_model(opt)
    net = model.net_g

    # 准备图像
    # Prepare images
    img_lr, img_hr, coord, cell = prepare_stereo(imgpath, img_path2, scale=zoomfactor)
    tensor_lr = PIL2Tensor(img_lr)
    tensor_hr = PIL2Tensor(img_hr)
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
    cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)
    

    b, c, orig_h, orig_w = tensor_lr.shape

    frame_index = math.ceil(b / 2) - 1

    w, h, position_pil = click_select_position(img_hr[frame_index], window_size)

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient_continuous(tensor_lr.numpy(), coord, cell, net, attr_objective,
                                                                            gaus_blur_path_func, frame_index, cuda=True)
    
    interpolated_grad_numpy = np.stack([interpolated_grad_numpy[:,:3,:,:],interpolated_grad_numpy[:,-3:,:,:]], axis=0)
    result_numpy = result_numpy
    # result_numpy = np.concatenate([result_numpy[:,:,:3,:,:],result_numpy[:,:,-3:,:,:]], axis=1)

    for index in range(b):
        grad_numpy, result = saliency_map(interpolated_grad_numpy[index], result_numpy)
        abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
        saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
        saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
        blend_abs_and_input = cv2_to_pil(
            pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
        blend_kde_and_input = cv2_to_pil(
            pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
        pil = make_pil_grid(
            [position_pil,
            saliency_image_abs,
            blend_abs_and_input,
            blend_kde_and_input
            ]
        )
        # pil.show()
        result_dir = "visual/" + opt['name']+ '/'
        os.makedirs(result_dir, exist_ok=True)
        pil.save(result_dir + str(index) + ".png")

        gini_index = gini(abs_normed_grad_numpy)
        diffusion_index = (1 - gini_index) * 100
        print(f"The DI of {index} is {diffusion_index}")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path, imgpath="demo/demo2/lr0.png", img_path2="demo/demo2/lr1.png")