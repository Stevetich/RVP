from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import torch
import shutil
# from mmseg.datasets import PascalVOCDataset
import os
import cv2
import numpy as np
from tqdm import tqdm

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'

model_dir = '../Qwen-VL-Chat'
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# 指定生成超参数（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# Init pascal voc dataset

# val_dataset = PascalVOCDataset(
#         data_root = 'data/VOCdevkit/VOC2012',
#         data_prefix=dict(
#             img_path='JPEGImages', seg_map_path='SegmentationClass'),
#         ann_file='ImageSets/Segmentation/val.txt',
#     )

classes=['background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor']
classes_str = ""
for name in classes:
    classes_str += (name + ", ")
classes_str = classes_str[:-2]
name2id, id2name = dict(), dict()

for id, class_name in enumerate(classes):
    name2id[class_name] = id
    id2name[id] = class_name

prompt_template = "<img>{img_path}</img>The blue mask in the figure covers part of an object, \
    please analyze what is the most likely category of this object? Please select from the categories given below: {class_names}.\
    Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."

# Rendering all hyperpixels of the same image once at a time
voc_img_base_dir = "../data/VOCdevkit/VOC2012/JPEGImages"
superpixel_base_dir = "../data//superpixel_img/slic150"
val_img_name_list = os.listdir(superpixel_base_dir)

for img_name in tqdm(val_img_name_list, desc="Processing VOC2012:"):

    img = cv2.imread(os.path.join(voc_img_base_dir, img_name.split('.')[0]+".jpg"))
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
    
    

    pred_sem_seg = torch.zeros(img.shape[0], img.shape[1])
    tmp_save_dir = "../data/tmp_results2"
    os.makedirs(tmp_save_dir, exist_ok=True)
    img_paths = []
    for id in range(len(rendered_imgs)):
        rendered_img = rendered_imgs[id].permute(1,2,0).numpy().astype(np.uint8)
        img_path = os.path.join(tmp_save_dir, "{}.jpg".format(id))
        img_paths.append(img_path)
        
        cv2.imwrite(img_path, rendered_img)
        
    queries = []
    for i, img_path in enumerate(img_paths):
        if i == 6:
            break
        queries.append(prompt_template.format(img_path=img_path, class_names=classes_str))
        
    response, history = model.chat(tokenizer=tokenizer, queries=queries, history=None)
    # tqdm.write(response)
    try:
        class_id = name2id[response]
        superpixel_mask = superpixels_masks[id, 0]
        pred_sem_seg[superpixel_mask] = class_id
        # import ipdb
        # ipdb.set_trace()
    except:
        continue

    shutil.rmtree(tmp_save_dir)

    # pred_sem_seg_save_dir = "./sem_seg_preds/VOC2012/G/"
    # pred_sem_seg_save_dir = "../data/sem_seg_preds/VOC2012/B/"
    # os.makedirs(pred_sem_seg_save_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(pred_sem_seg_save_dir, img_name), pred_sem_seg.numpy().astype(np.uint8))
        

    

    


# # superpixel exps
# # image_path = 'rendered_superpixels_30/cat_teddy/6.jpg'
# image_path = 'rendered_superpixels/family/58.jpg'
# import ipdb
# ipdb.set_trace()
# response1, history = model.chat(tokenizer, query=f'<img>{image_path}</img>图中红色掩码覆盖了某个物体的一部分，请分析这个物体最可能的类别是什么?请从以下给出类别中挑选：人、盘子、窗户、墙', history=None)

# # image_path = 'rendered_superpixels_30/cat_teddy/7.jpg'
# image_path = 'rendered_superpixels/family/61.jpg'
# response2, history = model.chat(tokenizer, query=f'<img>{image_path}</img>图中红色掩码覆盖了某个物体的一部分，请分析这个物体最可能的类别是什么?请从以下给出类别中挑选：人、盘子、窗户、墙', history=None)

# # image_path = 'rendered_superpixels_30/cat_teddy/10.jpg'
# image_path = 'rendered_superpixels/family/65.jpg'
# response3, history = model.chat(tokenizer, query=f'<img>{image_path}</img>图中红色掩码覆盖了某个物体的一部分，请分析这个物体最可能的类别是什么?请从以下给出类别中挑选：人、盘子、窗户、墙', history=None)

# # image_path = 'rendered_superpixels_30/cat_teddy/12.jpg'
# image_path = 'rendered_superpixels/family/6.jpg'
# response4, history = model.chat(tokenizer, query=f'<img>{image_path}</img>图中红色掩码覆盖的部分所属的物体的类别是什么?', history=None)

# print(response1)
# print(response2)
# print(response3)
# print(response4)