import numpy as np
import os
import cv2
from PIL import Image
from multiprocessing import Pool
from skimage.segmentation import slic

# from fast_slic import SlicAvx2
from fast_slic import Slic

import time

def compute_slic(img_base_dir, img_name):
    
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    segments_slic = slic(
        im, n_segments=150, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    im = Image.fromarray(segments_slic)
    im.save(
        "./superpixels_slic/{}".format(img_name)
    )

def compute_slic_30(img_base_dir, img_name):
    
    # im = Image.open(os.path.join(img_base_dir, img_name))
    im = cv2.imread(os.path.join(img_base_dir, img_name))
    segments_slic = slic(
        im, n_segments=30, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    im = Image.fromarray(segments_slic)
    im.save(
        "./superpixels_slic30/{}".format(img_name)
    )

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
    # import ipdb
    # ipdb.set_trace()



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

    img_base_dir = "./test_images"
    img_name_list = os.listdir(img_base_dir)

    os.makedirs("./superpixels_slic", exist_ok=True)
    os.makedirs("./superpixels_slic30", exist_ok=True)
    os.makedirs("./superpixels_slicfast", exist_ok=True)


    # print("############### Library slic time cost #################")
    # for img_name in img_name_list:
    #     sec1 = time.time()
    #     compute_slic(img_base_dir, img_name)
    #     sec2 = time.time()
    #     print("Per cost 1 : {}".format(sec2 - sec1))

    # for img_name in img_name_list:
    #     compute_slic_30(img_base_dir, img_name)

    print("############### Slic fast time cost #################")
    for img_name in img_name_list:
        sec3 = time.time()
        compute_slic_fast(img_base_dir, img_name)
        sec4 = time.time()
        print("Per cost 2 : {}".format(sec4 - sec3))

    # superpixels_base_dir = "./superpixels_slic30"
    # out_base_dir = "./rendered_superpixels_30"
    # os.makedirs("./rendered_superpixels_30", exist_ok=True)

    # for img_name in img_name_list:
    #     render_superpixels(img_base_dir, superpixels_base_dir, img_name, out_base_dir)

    # superpixels_base_dir = "./superpixels_slic"
    # out_base_dir = "./rendered_superpixels"
    # os.makedirs("./rendered_superpixels", exist_ok=True)

    # for img_name in img_name_list:
    #     render_superpixels(img_base_dir, superpixels_base_dir, img_name, out_base_dir)

    superpixels_base_dir = "./superpixels_slicfast"
    out_base_dir = "./rendered_superpixelsfast"
    os.makedirs("./rendered_superpixelsfast", exist_ok=True)

    for img_name in img_name_list:
        render_superpixels(img_base_dir, superpixels_base_dir, img_name, out_base_dir)