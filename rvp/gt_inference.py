import argparse
import sys
sys.path.append('/remote-home/zhangjiacheng/MiniGPT-4/')
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# from minigpt4.common.config import Config
# from minigpt4.common.registry import registry
# from minigpt4.common.eval_utils import prepare_texts
# from minigpt4.conversation.conversation import CONV_VISION_minigptv2
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

from scipy.ndimage import label
from sklearn.cluster import KMeans

from mmseg.datasets.transforms import *
from mmseg.datasets import PascalVOCDataset
from mmengine.structures import PixelData
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import shutil
import ipdb
import warnings
warnings.filterwarnings("ignore")

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'
model_dir = '../Qwen-VL-Chat'
# finetune_dir = '/remote-home/zhangjiacheng/Qwen-VL/output_qwen_attn_perb'
finetune_dir = '/home/jy/mm/Qwen-VL/output_qwen'

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

# prompt_template = "<img>{img_path}</img>The green mask in the figure covers part of an object, \
# please analyze what is the most likely category of this object? Please select from the categories given below: {class_names}.\
# Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."
# 'The green glow on the picture is a manually added object mark.'
# prompt_template = '<img>{img_path}</img> \
# What is the most likely category of the object under the green glow? \
# Choose your answer from this list: {class_names}.'
prompt_template = '<img>{img_path}</img> \
What is the most likely category of the object marked by the green dot? \
Choose your answer from this list: {class_names}.'

# prompt_template = "<img>{img_path}</img> What is the object under the green mask? \
# Choose your answer from this list: {class_names}."

# prompt_template = '<img>{}</img>You are a professional semantic segmentation model. \
# In the image, the green glow covers an object or part of an object.\
# From the following list: [background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor], \
# select the name that you think best represents the object or part of the object category under the green glow (or green mask). \
# You can guess the name of the covered object by the following steps: \
# 1. Think "Is the object under the green glow (or green mask) complete?" \
# 2. If complete, think "What is the object?" \
# 3. If not complete, think "What is the complete object of this component?" \
# 4. Strictly choose your answer from the above list. \
# 5. If you are not certain with the object under the mask, just reply with "background".'

# question = ("The blue mask in the figure covers part of an object, \
# please analyze what is the most likely category of this object? Please select from the categories given below: {}.\
# Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure.".format(classes_str), )
    
# prompt_template = "<img>{img_path}</img>In the above provided image, I have marked a specific location with a green dot. \
#     Please analyze the content at the red dot's position and tell me what it most likely belongs from categories given below: {class_names}. \
#     Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."
#     # Please just response with category names from the list I give you."

def main():
    # DDP
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Tokenizer and VL-Model
    # QWen
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
    model = AutoModelForCausalLM.from_pretrained(finetune_dir, device_map="cuda", trust_remote_code=True).eval()
    
    class SimulateArgs:
        def __init__(self):
            self.config = '../mmsegmentation/my_configs/pascal_voc12_base.py'
            self.launcher = 'none'
            self.cfg_options = None
            self.work_dir = None
            self.checkpoint = None
            self.tta = False
            self.out = None

    args = SimulateArgs()


    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True
    
    runner = Runner.from_cfg(cfg)
    val_dataloader = runner.val_dataloader
    evaluator = runner.val_evaluator

    runner.call_hook('before_val')
    runner.call_hook('before_val_epoch')
    runner.model.eval()
    
    color = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]]).cuda() # R G B
    Red = color[0].reshape(1, 3, 1, 1)
    Green = color[1].reshape(1, 3, 1, 1)
    Blue = color[2].reshape(1, 3, 1, 1)
    
    rendered_base_dir = "rendered_img"
    semseg_base_dir = "sem_seg_preds"
    data_root = "../data"
    rendered_dir = os.path.join(data_root, rendered_base_dir)
    semseg_save_dir = os.path.join(data_root, semseg_base_dir, 'scikit30', 'G')
    if not os.path.isdir(semseg_save_dir):
        os.makedirs(semseg_save_dir, exist_ok=True)
    
    for idx, data_sample in enumerate(val_dataloader):
        img_path = data_sample['data_samples'][0].img_path
        img_name = img_path.split('/')[-1].split('.')[0]
        # print (f'{idx}: {img_name}')
        
        # Image: 1, C, H, W
        img = cv2.imread(data_sample['data_samples'][0].img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2).cuda()
        
        # # Mask: 1, C, H, W
        # mask = data_sample['data_samples'][0].gt_sem_seg.data.clone()
        # # mask[mask == 255] = 0
        # # mask[mask != 0] = 1
        # mask[(mask < 255) & (mask != 0)] = 1
        # mask[mask == 255] = 0
        # mask = mask[:, None].repeat(1, 3, 1, 1).bool().cuda()
        
        gt_mask = data_sample['data_samples'][0].gt_sem_seg.data.clone().cuda()
        resized_mask = TF.resize(gt_mask, (448, 448), interpolation=InterpolationMode.NEAREST)
        # resized_mask[(resized_mask < 255) & (resized_mask != 0)] = 1
        resized_mask[resized_mask == 255] = 0
        
        # Render Images
        # mask_img = torch.zeros_like(img).cuda()
        # remain_img = torch.zeros_like(img).cuda()
        # mask_img[mask] = img[mask]
        # remain_img[mask.logical_not()] = img[mask.logical_not()]

        # rendered_img = (mask_img * 0.6 + mask * Green * 0.4) + remain_img
        
        pred_sem_seg = torch.zeros(img.shape[2], img.shape[3])
        rendered_name_dir = os.path.join(rendered_dir, img_name)
        os.makedirs(rendered_name_dir, exist_ok=True)
        
        for l in torch.unique(resized_mask):
            if (l == 0):
                continue
            l = l.item()
            
            # ptr = torch.zeros(32, 32, dtype=torch.bool)
            # ptr_mask = torch.rand(32, 32) < 0.1
            # for i in range(32):
            #     for j in range(32):
            #         lbl = torch.unique(resized_mask[0, i*14:(i+1)*14, j*14:(j+1)*14])
            #         if torch.any(lbl == l): 
            #             ptr[i, j] = True
            # ptr[ptr_mask] ^= True

            # mask = TF.resize(ptr[None], (gt_mask.shape[1], gt_mask.shape[2]), interpolation=InterpolationMode.NEAREST)
            mask = torch.where(gt_mask == l, 1, 0)
            mask = mask[:, None].repeat(1, 3, 1, 1).bool().cuda()
            
            # 计算质心
            _, _, H, W = mask.shape
            y_coords = torch.arange(H)[None, None, :, None].expand_as(mask).cuda()
            x_coords = torch.arange(W)[None, None, None, :].expand_as(mask).cuda()

            y_weighted = mask * y_coords
            x_weighted = mask * x_coords

            y_center = y_weighted.sum(dim=(2, 3)) / torch.where(y_weighted != 0, 1, 0).sum(dim=(2, 3))
            x_center = x_weighted.sum(dim=(2, 3)) / torch.where(x_weighted != 0, 1, 0).sum(dim=(2, 3))
            
            # ipdb.set_trace()
            # 随机取点
            coords = torch.nonzero(mask)
            rand_coords = coords[torch.randint(0, len(coords), (1, ))][0]
            y_center, x_center = rand_coords[2][None, None], rand_coords[3][None, None]

            mask_img = torch.zeros_like(img).cuda()
            remain_img = torch.zeros_like(img).cuda()
            mask_img[mask] = img[mask]
            remain_img[mask.logical_not()] = img[mask.logical_not()]

            # rendered_img = mask * Green * 0.4 + img * 0.6
            # rendered_img = (mask_img * 0.6 + mask * Green * 0.4) + remain_img
            rendered_img = img[0].clone().permute(1,2,0).cpu().numpy().astype(np.uint8)
            cv2.circle(rendered_img, (int(np.round(x_center[0][0].item())), int(np.round(y_center[0][0].item()))), 8, (0, 255, 0), -1)
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
            
            tmp_img_path = os.path.join(rendered_name_dir, '{}.jpg'.format(l))
            cv2.imwrite(tmp_img_path, rendered_img)
            with torch.inference_mode():
                responses, _ = model.chat(tokenizer, queries=[prompt_template.format(img_path=tmp_img_path, class_names=classes_str)], history=None)
                print (responses)
            try:
                class_id = name2id[responses[0]]
                pred_sem_seg[mask[0, 0]] = class_id
            except:
                pass
        
        
        # rendered_img = rendered_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
        # rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        # img_path = os.path.join(rendered_name_dir, "{}.jpg".format(0))
        # cv2.imwrite(img_path, rendered_img)

        # with torch.inference_mode():
        #     responses, _ = model.chat(tokenizer, queries=[prompt_template.format(img_path=img_path, class_names=classes_str)], history=None)
        #     print (responses)
        # try:
        #     class_id = name2id[responses[0]]
        #     pred_sem_seg[mask[0, 0]] = class_id
        # except:
        #     pass
        
        seg_save_path = os.path.join(semseg_save_dir, img_name+".npy")
        np.save(seg_save_path, pred_sem_seg.numpy().astype(np.uint8)[None])
        # print (f'seg_save_path: {seg_save_path}')
        
if __name__ == '__main__':
    main()