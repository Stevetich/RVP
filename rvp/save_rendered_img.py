import os
import torch
import cv2
import numpy as np
from tqdm import tqdm



def save_rendered_imgs(voc_img_base_dir, superpixel_base_dir):
    """
    Save rendered images (img_name/id.jpg)
    Args:
        voc_img_base_dir (str): Directory of voc images.
        superpixel_base_dir (str): Directory of corresponding superpixel images.
    """
    val_img_name_list = os.listdir(superpixel_base_dir)

    for img_name in tqdm(val_img_name_list, desc="Processing VOC2012:"):
        img = cv2.imread(os.path.join(voc_img_base_dir, img_name))
        superpixels = cv2.imread(os.path.join(superpixel_base_dir, img_name))
        superpixels_ids = np.unique(superpixels)
        
        # Gather per superpixel masks
        superpixels_masks = []
        for superpixel_id in superpixels_ids:
            mask = (superpixels == superpixel_id)
            superpixels_masks.append(torch.tensor(mask))

        superpixels_masks = torch.stack(superpixels_masks, dim=0)
        superpixels_masks = superpixels_masks.permute(0,3,1,2)
        color = torch.tensor([[0,0,255], [0,255,0], [255,0,0]]) # R G B
        # Red = color[0].reshape(1, 3, 1, 1)
        # Green = color[1].reshape(1, 3, 1, 1)
        Blue = color[2].reshape(1, 3, 1, 1)

        # superpixels_masks = superpixels_masks * Red

        img_repeated = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).expand_as(superpixels_masks)
        
        # 图片混合
        # rendered_imgs = img_repeated * 0.6 + superpixels_masks * Red * 0.4
        # rendered_imgs = img_repeated * 0.6 + superpixels_masks * Green * 0.4
        rendered_imgs = img_repeated * 0.6 + superpixels_masks * Blue * 0.4
        # rendered_imgs = K.enhance.add_weighted(img_repeated, 0.6, superpixels_masks, 0.4, 0)
        
        rendered_img_save_dir = '../data/rendered_img/scikit30/B'
        rendered_img_save_dir = os.path.join(rendered_img_save_dir, img_name.split('.')[0])
        os.makedirs(rendered_img_save_dir, exist_ok=True)
        
        for id in range(rendered_imgs.shape[0]):
            rendered_img = rendered_imgs[id].permute(1, 2, 0).numpy().astype(np.uint8)
            img_path = os.path.join(rendered_img_save_dir, "{}.jpg".format(id))
            
            # print(img_path)
            cv2.imwrite(img_path, rendered_img)
            
if __name__ == '__main__':
    voc_img_base_dir = "../data/VOCdevkit/VOC2012/JPEGImages"
    superpixel_base_dir = "../data/superpixel_img/scikit30"
    save_rendered_imgs(voc_img_base_dir, superpixel_base_dir)
