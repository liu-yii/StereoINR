from os import path as osp
from PIL import Image
from tqdm import tqdm
from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = '/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/X4/HR_patch'
    meta_info_txt = '/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/StereoSR/iPASSR_trainset/Train/X4/meta_info_sub_HR_path.txt'

    img_list = sorted(list(scandir(gt_folder)))
    pbar = tqdm(total=len(img_list))
    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            #print(idx + 1, info)
            f.write(f'{info}\n')
            pbar.update()


if __name__ == '__main__':
    generate_meta_info_div2k()
