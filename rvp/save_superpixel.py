import argparse
import os
import numpy as np
import torch

import cv2
from PIL import Image

from multiprocessing import Pool
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmseg.datasets import PascalVOCDataset

# slic methods
from skimage.segmentation import slic
from fast_slic import Slic as fslic

superpixel_base_dir = "superpixel_img"
    
def compute_slic_fast(img_base_dir, img_name):
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    slic = fslic(num_components=150, compactness=6)
    segments_slic = slic.iterate(im).astype(np.uint8)
    
    im = Image.fromarray(segments_slic)
    im.save(
        "./superpixels_slicfast/{}".format(img_name)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='For Generation of Transferrable Attack')
    parser.add_argument('--data_root', type=str, default='../data', help='Location of the data.')
    parser.add_argument('--slic_mode', type=str, default='scikit', help='[scikit | fast] Superpixel method mode.')
    parser.add_argument('--seg_num', type=int, default=30, help='Superpixel method mode.')
    args = parser.parse_args()
    
    assert args.slic_mode in ['scikit', 'fast'], "Slic mode not supported: {}".format(args.slic_mode)
    
    val_dataset = PascalVOCDataset(
        data_root = os.path.join(args.data_root, 'VOCdevkit/VOC2012'),
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val100.txt',
    )

    slic_method_name = args.slic_mode + str(args.seg_num)
    superpixel_dir = os.path.join(args.data_root, superpixel_base_dir)
    superpixel_dir = os.path.join(superpixel_dir, slic_method_name)
    os.makedirs(superpixel_dir, exist_ok=True)
    
    print ('Save superpixel images path: {}'.format(superpixel_dir))
    print ('Slic method name: {}'.format(slic_method_name))

    for i in tqdm(range(len(val_dataset))):
        datasample = val_dataset[i]
        img = cv2.imread(datasample['img_path'])
        # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_name = datasample['img_path'].split('/')[-1]
        save_path = os.path.join(superpixel_dir, img_name)
        
        if args.slic_mode == 'scikit':
            sp = slic(img, n_segments=args.seg_num, compactness=6, sigma=3.0, start_label=0).astype(np.uint8)
            
        elif args.slic_mode == 'fast':
            slic_method = fslic
            slic = fslic(num_components=args.seg_num, compactness=6)
            sp = slic.iterate(img).astype(np.uint8)
        else:
            raise ValueError('Slice Method is not Supported: {}'.format(args.slic_mode))

        # sp = cv2.cvtColor(sp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, sp)
