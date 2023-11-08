import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm


voc_base_dir = "VOCdevkit/VOC2012/JPEGImages"
superpixel_base_dir = "superpixel_img"
rendered_base_dir = "rendered_img"


def save_rendered_imgs(voc_dir, superpixel_dir, rendered_dir, color_mode):
    """
    Save rendered images (img_name/id.jpg)
    Args:
        voc_dir (str): Directory of voc images.
        superpixel_dir (str): Directory of corresponding superpixel images.
    """
    val_img_name_list = os.listdir(superpixel_dir)

    for img_name in tqdm(val_img_name_list, desc="Processing VOC2012:"):
        img = cv2.imread(os.path.join(voc_dir, img_name))
        superpixels = cv2.imread(os.path.join(superpixel_dir, img_name))
        superpixels_ids = np.unique(superpixels)
        
        # Stack superpixel masks 
        # superpixels_masks.shape: (ids, C, H, W)
        superpixels_masks = []
        for superpixel_id in superpixels_ids:
            mask = (superpixels == superpixel_id)
            superpixels_masks.append(torch.tensor(mask))

        superpixels_masks = torch.stack(superpixels_masks, dim=0)
        superpixels_masks = superpixels_masks.permute(0,3,1,2)
        img_repeated = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).expand_as(superpixels_masks)
        
        color = torch.tensor([[0,0,255], [0,255,0], [255,0,0]]) # R G B
        Red = color[0].reshape(1, 3, 1, 1)
        Green = color[1].reshape(1, 3, 1, 1)
        Blue = color[2].reshape(1, 3, 1, 1)
        
        mask_imgs = torch.zeros_like(img_repeated)
        remain_imgs = torch.zeros_like(img_repeated)
        mask_imgs[superpixels_masks] = img_repeated[superpixels_masks]
        remain_imgs[superpixels_masks.logical_not()] = img_repeated[superpixels_masks.logical_not()]
        
        # Render images
        # Mix-up (Background brightness unchanged)
        if color_mode == 'R':
            rendered_imgs = (mask_imgs * 0.6 + superpixels_masks * Red * 0.4) + remain_imgs
        elif color_mode == 'G':
            rendered_imgs = (mask_imgs * 0.6 + superpixels_masks * Green * 0.4) + remain_imgs
        elif color_mode == 'B':
            rendered_imgs = (mask_imgs * 0.6 + superpixels_masks * Blue * 0.4) + remain_imgs
        else:
            raise ValueError('Color not supported: {}'.format(color_mode))

        # Point prompt
        _, _, H, W = superpixels_masks.shape
        y_coords = torch.arange(H)[None, None, :, None].expand_as(superpixels_masks)
        x_coords = torch.arange(W)[None, None, None, :].expand_as(superpixels_masks)

        y_weighted = superpixels_masks * y_coords
        x_weighted = superpixels_masks * x_coords

        y_center = y_weighted.sum(dim=(2, 3)) / torch.where(y_weighted != 0, 1, 0).sum(dim=(2, 3))
        x_center = x_weighted.sum(dim=(2, 3)) / torch.where(x_weighted != 0, 1, 0).sum(dim=(2, 3))        
        
        rendered_img_save_dir = os.path.join(rendered_dir, img_name.split('.')[0])
        os.makedirs(rendered_img_save_dir, exist_ok=True)
        
        # Save rendered images of all ids per img
        for id in range(img_repeated.shape[0]):
            # rendered_img = rendered_imgs[id].permute(1, 2, 0).numpy().astype(np.uint8)
            rendered_img = img_repeated[id].permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.circle(rendered_img, (int(np.round(x_center[id][0].item())), int(np.round(y_center[id][0].item()))), 8, (255, 0, 0), -1)
            
            img_path = os.path.join(rendered_img_save_dir, "{}.jpg".format(id))
            cv2.imwrite(img_path, rendered_img)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For Generation of Semantic Segmentation Predictions.')
    parser.add_argument('--data_root', type=str, default='../data', help='Location of the data.')
    parser.add_argument('--slic_mode', type=str, default='scikit', help='[scikit | fast] Superpixel method mode.')
    parser.add_argument('--seg_num', type=int, default=30, help='Superpixel method mode.')
    parser.add_argument('--color', type=str, default='B', help='[R | G | B] Color of superpixel mask.')
    args = parser.parse_args()
    
    assert args.slic_mode in ['scikit', 'fast'], "Slic mode not supported: {}".format(args.slic_mode)
    assert args.color in ['R', 'G', 'B'], "Color not supported: {}".format(args.color)
    
    # Directories
    slic_method_name = args.slic_mode + str(args.seg_num)
    voc_dir = os.path.join(args.data_root, voc_base_dir)
    superpixel_dir = os.path.join(args.data_root, superpixel_base_dir, slic_method_name)
    rendered_dir = os.path.join(args.data_root, rendered_base_dir, slic_method_name, args.color)
    # voc_dir = "../data/VOCdevkit/VOC2012/JPEGImages"
    # superpixel_dir = "../data/superpixel_img/scikit30"
    
    assert os.path.isdir(voc_dir), 'Not a valid voc image dir: {}'.format(voc_dir)
    assert os.path.isdir(superpixel_dir), 'Not a valid superpixel image dir: {}'.format(superpixel_dir)

    print ('Voc original images dir: {}'.format(voc_dir))
    print ('Superpixel images dir: {}'.format(superpixel_dir))
    print ('Rendered images dir: {}'.format(rendered_dir))
    
    print ('Slic method name: {}'.format(slic_method_name))
    print ('Color mode: {}'.format(args.color))
    
    save_rendered_imgs(voc_dir, superpixel_dir, rendered_dir, args.color)
