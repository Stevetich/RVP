import os
import numpy as np
import torch

import cv2
from PIL import Image

from multiprocessing import Pool
from skimage.segmentation import slic
from torch.utils.data import DataLoader

from mmseg.datasets import PascalVOCDataset

from tqdm import tqdm

def compute_slic(img_base_dir, img_name):
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    segments_slic = slic(
        im, n_segments=150, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    # im = Image.fromarray(segments_slic)
    # im.save(
    #     "./superpixels_slic/{}".format(img_name)
    # )
    return segments_slic
    
def compute_slic_30(img_base_dir, img_name):
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    segments_slic = slic(
        im, n_segments=30, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    # im = Image.fromarray(segments_slic)
    # im.save(
    #     "./superpixels_slic30/{}".format(img_name)
    # )
    return segments_slic
    
def compute_slic_fast(img_base_dir, img_name):
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    slic = Slic(num_components=150, compactness=6)
    segments_slic = slic.iterate(im).astype(np.uint8)
    
    im = Image.fromarray(segments_slic)
    im.save(
        "./superpixels_slicfast/{}".format(img_name)
    )

def render_in_img(im, superpixels, superpixel_id, out_base_dir, img_name):
    mask = (superpixels == superpixel_id).astype(np.uint8)
    # red_mask = np.zeros_like(im)
    # red_mask[:, :, 2] = 255
    color = np.array([0,0,255])
    color = color[None, None, :]

    mask = mask * color
    masked_img = cv2.addWeighted(im, 0.7, mask.astype(np.uint8), 0.3, 0)
    os.makedirs(os.path.join(out_base_dir, img_name.split(".")[0]), exist_ok=True)
    cv2.imwrite(os.path.join(out_base_dir, img_name.split(".")[0], str(superpixel_id)+".jpg"), masked_img)
    
    
def render_superpixels(img_base_dir, superpixels_base_dir, img_name, out_base_dir):

    im = cv2.imread(os.path.join(img_base_dir, img_name))
    superpixels = cv2.imread(os.path.join(superpixels_base_dir, img_name))
    superpixels_ids = np.unique(superpixels)

    for id in superpixels_ids:
        render_in_img(im, superpixels, id, out_base_dir, img_name)


def compute_slic_and_render_superpixels(img_base_dir, img_name, out_base_dir):

    im = cv2.imread(os.path.join(img_base_dir, img_name))
    superpixels = slic(
        im, n_segments=150, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)

    superpixels_ids = np.unique(superpixels)
    superpixels = superpixels.unsqueeze(-1).repeat(1,1,3)

    for id in superpixels_ids:
        render_in_img(im, superpixels, id, out_base_dir, img_name)


if __name__ == "__main__":
    val_dataset = PascalVOCDataset(
        data_root = '../data/VOCdevkit/VOC2012',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
    )

    for i in tqdm(range(len(val_dataset))):
        datasample = val_dataset[i]
        img = cv2.imread(datasample['img_path'])
        img_name = datasample['img_path'].split('/')[-1]
        save_path = os.path.join('../data/superpixel_img/slic150', img_name)
        
        # scikit 
        superpixel = slic(img, n_segments=150, compactness=6, sigma=3.0, start_label=0).astype(np.uint8)
        superpixel = Image.fromarray(superpixel)
        superpixel.save(save_path)
        tqdm.write(save_path)
