import argparse
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import os
import cv2
import numpy as np

import torch
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

voc_base_dir = "VOCdevkit/VOC2012/JPEGImages"
superpixel_base_dir = "superpixel_img"
rendered_base_dir = "rendered_img"
semseg_base_dir = "sem_seg_preds"


class RenderedImageDataset(Dataset):
    def __init__(self, rendered_dir: str):
        """
        Encapsulation for DDP (or DistributedDataSampler).
        Args:
            rendered_dir (str): Directory of rendered images.
        """
        super().__init__()
        if not os.path.isdir(rendered_dir):
            raise NotADirectoryError('Not a valid rendered image dir: {}'.format(rendered_dir))
        self.rendered_dir = rendered_dir
        self.img_names = os.listdir(rendered_dir)
    
    def __getitem__(self, idx):
        id_img_path = os.path.join(self.rendered_dir, self.img_names[idx])
        return id_img_path, self.img_names[idx], idx
    
    def __len__(self):
        return len(self.img_names)
        

def main():
    parser = argparse.ArgumentParser(description='For Generation of Semantic Segmentation Predictions.')
    parser.add_argument('--data_root', type=str, default='../data', help='Location of the data.')
    parser.add_argument('--slic_mode', type=str, default='scikit', help='[scikit | fast] Superpixel method mode.')
    parser.add_argument('--seg_num', type=int, default=30, help='Superpixel method mode.')
    parser.add_argument('--color', type=str, default='B', help='[R | G | B] Color of superpixel mask.')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size.')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    
    assert args.slic_mode in ['scikit', 'fast'], "Slic mode not supported: {}".format(args.slic_mode)
    assert args.color in ['R', 'G', 'B'], "Color not supported: {}".format(args.color)
    
    batch_size = args.batch_size
    torch.manual_seed(1234)
    
    # Directories
    slic_method_name = args.slic_mode + str(args.seg_num)
    voc_dir = os.path.join(args.data_root, voc_base_dir)
    superpixel_dir = os.path.join(args.data_root, superpixel_base_dir, slic_method_name)
    rendered_dir = os.path.join(args.data_root, rendered_base_dir, slic_method_name, args.color)
    semseg_save_dir = os.path.join(args.data_root, semseg_base_dir, slic_method_name, args.color)
    if not os.path.isdir(semseg_save_dir):
        os.makedirs(semseg_save_dir, exist_ok=True)
    
    assert os.path.isdir(voc_dir), 'Not a valid voc image dir: {}'.format(voc_dir)
    assert os.path.isdir(superpixel_dir), 'Not a valid superpixel image dir: {}'.format(superpixel_dir)
    
    print ('Voc original images dir: {}'.format(voc_dir))
    print ('Superpixel images dir: {}'.format(superpixel_dir))
    print ('Semantic segmentation predictions save dir: {}'.format(semseg_save_dir))
        
    print ('Slic method name: {}'.format(slic_method_name))
    print ('Color mode: {}'.format(args.color))
    
    # DDP
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Tokenizer and VL-Model
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
    
    
    # Dataset
    val_dataset = RenderedImageDataset(rendered_dir)
    sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler)
    
    for rendered_img_path, img_name, idx in val_loader:
        # img_name: without postfix (.jpg)
        print ("Index: {}".format(idx.item()))

        img = cv2.imread(os.path.join(voc_dir, img_name[0]+".jpg"))
        superpixels = cv2.imread(os.path.join(superpixel_dir, img_name[0]+".jpg"))
        superpixels_ids = np.unique(superpixels)

        # Gather superpixel masks
        superpixels_masks = []
        for superpixel_id in superpixels_ids:
            mask = (superpixels == superpixel_id)
            superpixels_masks.append(torch.tensor(mask))

        superpixels_masks = torch.stack(superpixels_masks, dim=0)
        superpixels_masks = superpixels_masks.permute(0,3,1,2)
        pred_sem_seg = torch.zeros(img.shape[0], img.shape[1])
        
        # Generate ID Batch for the single rendered image.
        id_imgs = os.listdir(rendered_img_path[0])
        num_batches = int(np.ceil(len(id_imgs) / batch_size))
        id_imgs_batch = []
        for i in range(num_batches):
            batch = id_imgs[i*batch_size: min((i+1)*batch_size, len(id_imgs))]
            id_imgs_batch.append(batch)
        
        # Batch process
        for batch in id_imgs_batch:
            queries = []
            for id in batch:
                queries.append(prompt_template.format(img_path=os.path.join(rendered_img_path[0], id), class_names=classes_str))

            responses, history = model.chat(tokenizer=tokenizer, queries=queries, history=None)
            
            for response, id in zip(responses, batch):
                # id: with postfix (.jpg)
                try:
                    id = int(id.split('.')[0])
                    class_id = name2id[response]
                    superpixel_mask = superpixels_masks[id, 0]
                    pred_sem_seg[superpixel_mask] = class_id
                except:
                    continue
            
        cv2.imwrite(os.path.join(semseg_save_dir, img_name[0]+".jpg"), pred_sem_seg.numpy().astype(np.uint8))

    
if __name__ == '__main__':
    main()