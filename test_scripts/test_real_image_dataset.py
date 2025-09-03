import math
import os
import torchvision.utils
from basicsr.utils.options import parse_options,yaml_load
from basicsr.data import build_dataloader, build_dataset
import cv2
import numpy as np

def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """

    opt = yaml_load("/home/bobo/Temp_new/ZYB/IV_WORKING/codes/GeneralSR/SCGLANet/options/train/StereoSR/train_RealSCGLANet_L_x4_debug.yml")
    opt['dataset_enlarge_ratio'] = 1
    opt['is_train'] = True
    os.makedirs('tmp', exist_ok=True)
    opt_dataset = opt['datasets']['train']
    opt_dataset['phase'] = 'train'
    opt_dataset['scale'] = 4
    dataset = build_dataset(opt_dataset)

    data_loader = build_dataloader(dataset, opt_dataset, num_gpu=0, dist=False, sampler=None)
    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)


        gt = data['gt']
        kernel1 = data['kernel1']
        kernel2 = data['kernel2']
        sinc_kernel = data['sinc_kernel']

        kernel1 = kernel1.numpy().squeeze()
        kernel2 = kernel2.numpy().squeeze()
        sinc_kernel = sinc_kernel.numpy().squeeze()
        cv2.imshow('gg',np.hstack([kernel1,kernel2,sinc_kernel]))
        cv2.waitKey(0)
        print(kernel1,kernel2,sinc_kernel)

if __name__ == '__main__':
    main()
