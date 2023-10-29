from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import shutil
# from mmseg.datasets import PascalVOCDataset
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'
model_dir = '../Qwen-VL-Chat'


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

voc_img_base_dir = "../data/VOCdevkit/VOC2012/JPEGImages"
superpixel_base_dir = "../data/superpixel_img/scikit30"
val_img_name_list = os.listdir(superpixel_base_dir)


class RenderedImageDataset(Dataset):
    def __init__(self, rendered_img_dir: str='../data/rendered_img/scikit30'):
        super().__init__()
        # self.tokenizer = tokenizer
        # self.prompt = prompt
        self.rendered_img_dir = rendered_img_dir
        
        self.img_names = os.listdir(rendered_img_dir)
    
    def __getitem__(self, idx):
        id_img_path = os.path.join(self.rendered_img_dir, self.img_names[idx])
        return id_img_path, self.img_names[idx], idx
    
    def __len__(self):
        return len(self.img_names)
        

def main():
    torch.manual_seed(1234)
    
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
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
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()
    
    
    batch_size = 8
    val_dataset = RenderedImageDataset()
    sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler)
    
    for rendered_img_path, img_name, idx in val_loader:
        print ("idx: {}".format(idx.item()))

        img = cv2.imread(os.path.join(voc_img_base_dir, img_name[0]+".jpg"))
        superpixels = cv2.imread(os.path.join(superpixel_base_dir, img_name[0]+".jpg"))
        superpixels_ids = np.unique(superpixels)

        # Gather per superpixel masks
        superpixels_masks = []
        for superpixel_id in superpixels_ids:
            mask = (superpixels == superpixel_id)
            superpixels_masks.append(torch.tensor(mask))

        superpixels_masks = torch.stack(superpixels_masks, dim=0)
        superpixels_masks = superpixels_masks.permute(0,3,1,2)
        pred_sem_seg = torch.zeros(img.shape[0], img.shape[1])
        
        id_imgs = os.listdir(rendered_img_path[0])
        num_batches = int(np.ceil(len(id_imgs) / batch_size))
        id_imgs_batch = []
        

        
        for i in range(num_batches):
            batch = id_imgs[i*batch_size: min((i+1)*batch_size, len(id_imgs))]
            id_imgs_batch.append(batch)
        
        for batch in id_imgs_batch:
            queries = []
            for id in batch:
                queries.append(prompt_template.format(img_path=os.path.join(rendered_img_path[0], id), class_names=classes_str))

            responses, history = model.chat(tokenizer=tokenizer, queries=queries, history=None)
            
            for response, id in zip(responses, batch):
                try:
                    class_id = name2id[response]
                    superpixel_mask = superpixels_masks[id, 0]
                    pred_sem_seg[superpixel_mask] = class_id
                    # import ipdb
                    # ipdb.set_trace()
                except:
                    continue
        
        
        pred_sem_seg_save_dir = "../data/sem_seg_preds/VOC2012/B/"
        os.makedirs(pred_sem_seg_save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(pred_sem_seg_save_dir, img_name[0]+".jpg"), pred_sem_seg.numpy().astype(np.uint8))

    
if __name__ == '__main__':
    main()