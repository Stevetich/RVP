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
import torchvision.transforms as T
import torch.nn.functional as F

from sklearn.cluster import KMeans

from mmseg.datasets import PascalVOCDataset
import shutil
import ipdb

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

prompt_template = "<img>{}</img>The blue mask in the figure covers part of an object, \
please analyze what is the most likely category of this object? Please select from the categories given below: {}.\
Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."

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

img_path = '/remote-home/zhangjiacheng/RVP/data/rendered_img/scikit30/B/0.jpg'
# feature = [None]
# def hook_fn(m, fea_in, fea_out):
#     feature[0] = fea_out.detach().cpu()
# # Hook visual features
# with torch.inference_mode():
#     # hooker = FeatureHooker(layer=model.transformer.visual.transformer.resblocks[47])
#     handle = model.transformer.visual.transformer.resblocks[47].register_forward_hook(hook_fn)
    
#     queries = ['<img>{}</img>Describe this image.'.format(img_path)]
#     _, _ = model.chat(tokenizer=tokenizer, queries=queries, history=None)
#     # feature = hooker.fea[:, 0].float()
#     feature = feature[0][:, 0].float()
    
#     # hooker.handle.remove()
#     handle.remove()
img = cv2.imread(img_path)
print (img.shape)
torch.cuda.empty_cache()
responses, _ = model.chat(tokenizer, queries=[prompt_template.format(img_path, classes_str)], history=None)