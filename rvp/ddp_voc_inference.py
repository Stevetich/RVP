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

# # imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

from collections import Counter
from scipy.ndimage import label
from sklearn.cluster import KMeans
from mmseg.datasets import PascalVOCDataset
import shutil
import ipdb
ipdb.launch_ipdb_on_exception()

import warnings
warnings.filterwarnings("ignore")

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'
model_dir = '../Qwen-VL-Chat'
# finetune_dir = '/remote-home/zhangjiacheng/Qwen-VL/output_qwen_attn_perb'
model_dir = '/home/jy/mm/Qwen-VL/output_qwen_pnt_1sample'
n_clusters = 4
island = 3
valley = 10
kernel_size = 3

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

# prompt_template = "<img>{img_path}</img>The blue mask in the figure covers part of an object, \
# please analyze what is the most likely category of this object? Please select from the categories given below: {class_names}.\
# Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."

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

# prompt_template = '<img>{img_path}</img> \
# What is the most likely category of the object under the green glow? \
# Choose your answer from this list: {class_names}.'
prompt_template = '<img>{img_path}</img> \
What is the most likely category of the object marked by the green dot? \
Choose your answer from this list: {class_names}.'

# question = ("The blue mask in the figure covers part of an object, \
# please analyze what is the most likely category of this object? Please select from the categories given below: {}.\
# Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure.".format(classes_str), )
    
# prompt_template = "<img>{img_path}</img>In the above provided image, I have marked a specific location with a green dot. \
#     Please analyze the content at the red dot's position and tell me what it most likely belongs from categories given below: {class_names}. \
#     Please distinguish as many categories of objects as possible, and do not be affected by the main objects in the figure."
#     # Please just response with category names from the list I give you."

voc_base_dir = "VOCdevkit/VOC2012/JPEGImages"
superpixel_base_dir = "superpixel_img"
rendered_base_dir = "rendered_img"
semseg_base_dir = "sem_seg_preds"

color_full = {
    'B': 'blue',
    'G': 'green',
    'R': 'red'
}


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

class FeatureHooker:
    def __init__(self, layer):
        self.layer = layer
        self.fea = None
        self.handle = None

        self.register_hook()
    
    def hook_fn(self, m, fea_in, fea_out):
        self.fea = fea_out.detach().cpu()
        
    def register_hook(self):
        self.handle = self.layer.register_forward_hook(self.hook_fn)        

def main():
    # DDP
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    parser = argparse.ArgumentParser(description='For Generation of Semantic Segmentation Predictions.')
    parser.add_argument('--data_root', type=str, default='../data', help='Location of the data.')
    parser.add_argument('--slic_mode', type=str, default='scikit', help='[scikit | fast] Superpixel method mode.')
    parser.add_argument('--seg_num', type=int, default=30, help='Superpixel method mode.')
    parser.add_argument('--color', type=str, default='B', help='[R | G | B] Color of superpixel mask.')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size.')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--cluster_method', type=str, default='superpixel', help='[superpixel | feature_cluster]')
    args = parser.parse_args()
    
    assert args.slic_mode in ['scikit', 'fast'], "Slic mode not supported: {}".format(args.slic_mode)
    assert args.color in ['R', 'G', 'B'], "Color not supported: {}".format(args.color)
    
    batch_size = args.batch_size
    torch.manual_seed(1234)
    
    # Directories
    slic_method_name = args.slic_mode + str(args.seg_num)
    voc_dir = os.path.join(args.data_root, voc_base_dir)
    superpixel_dir = os.path.join(args.data_root, superpixel_base_dir, slic_method_name)
    rendered_dir = os.path.join(args.data_root, rendered_base_dir)
    semseg_save_dir = os.path.join(args.data_root, semseg_base_dir, slic_method_name, args.color)
    if not os.path.isdir(semseg_save_dir):
        os.makedirs(semseg_save_dir, exist_ok=True)
    
    assert os.path.isdir(voc_dir), 'Not a valid voc image dir: {}'.format(voc_dir)
    # assert os.path.isdir(superpixel_dir), 'Not a valid superpixel image dir: {}'.format(superpixel_dir)
    
    if rank == 0:
        print ('Voc original images dir: {}'.format(voc_dir))
        print ('Superpixel images dir: {}'.format(superpixel_dir))
        print ('Semantic segmentation predictions save dir: {}'.format(semseg_save_dir))
            
        print ('Slic method name: {}'.format(slic_method_name))
        print ('Color mode: {}'.format(color_full[args.color]))
    
    
    
    # Tokenizer and VL-Model
    # QWen
    tokenizer = AutoTokenizer.from_pretrained('../Qwen-VL-Chat', trust_remote_code=True)
    if not hasattr(tokenizer, 'model_dir'):
        tokenizer.model_dir = '../Qwen-VL-Chat'
    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
    # 使用CPU进行推理，需要约32GB内存
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
    # 默认gpu进行推理，需要约24GB显存
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()

    # MiniGPT V2
    # class SimulateArgs:
    #     def __init__(self):
    #         self.cfg_path = '/remote-home/zhangjiacheng/MiniGPT-4/eval_configs/minigptv2_eval.yaml'
    #         self.gpu_id = 0
    #         self.options = None

    # simulate_args = SimulateArgs()
    # device = 'cuda:{}'.format(simulate_args.gpu_id)

    # print('Initializing Chat')
    # cfg = Config(simulate_args)
    # model_config = cfg.model_cfg
    # model_config.device_8bit = simulate_args.gpu_id
    # model_cls = registry.get_model_class(model_config.arch)
    # model = model_cls.from_config(model_config).to(device)
    # model = model.eval()

    # vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    # vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # conv_temp = CONV_VISION_minigptv2.copy()
    # conv_temp.system = ""
    
    # Dataset
    # val_dataset = RenderedImageDataset(rendered_dir)
    val_dataset = PascalVOCDataset(
        data_root = os.path.join(args.data_root, 'VOCdevkit/VOC2012'),
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
    )
    
    def custom_collate_fn(batch):
        """
        Custom collate function to filter out None values.
        Args:
            batch (Any): batch data.
        """
        batch = [{k: v for k, v in item.items() if v is not None} for item in batch]
        return torch.utils.data.dataloader.default_collate(batch)
    
    sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler, collate_fn=custom_collate_fn)
    
    color = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]]).cuda() # R G B
    Red = color[0].reshape(1, 3, 1, 1)
    Green = color[1].reshape(1, 3, 1, 1)
    Blue = color[2].reshape(1, 3, 1, 1)
    
    if args.cluster_method == 'feature_cluster':
        for data in val_loader:
            img_path = data['img_path'][0]
            img_name = img_path.split('/')[-1].split('.')[0]
            print (img_name)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[0], img.shape[1]
            img = torch.tensor(img.transpose(2, 0, 1)).cuda()
            
            feature = [None]
            def hook_fn(m, fea_in, fea_out):
                feature[0] = fea_out.detach().cpu()
            # Hook visual features
            with torch.inference_mode():
                # hooker = FeatureHooker(layer=model.transformer.visual.transformer.resblocks[47])
                handle = model.transformer.visual.transformer.resblocks[47].register_forward_hook(hook_fn)
                
                queries = ['<img>{}</img>Describe this image.'.format(img_path)]
                _, _ = model.chat(tokenizer=tokenizer, queries=queries, history=None)
                
                # image = Image.open(img_path).convert('RGB')
                # image = vis_processor(image)[None]
                # text = prepare_texts(('Describe this image.', ), conv_temp)
                # _ = model.generate(image, text, max_new_tokens=100, do_sample=False)

                # feature = hooker.fea[:, 0].float()
                feature = feature[0][:, 0].float()
                # feature = feature[0][0, 1:, ...].float()
                
                # hooker.handle.remove()
                handle.remove()
            
            
            # # Features clustering
            num_clusters = n_clusters
            kmeans = KMeans(n_clusters=num_clusters)
            cluster_ids = kmeans.fit_predict(feature)
            
            # 离群点和空缺处理
            cluster_ids = np.resize(cluster_ids, (32, 32))
            masks = []
            for id in np.unique(cluster_ids):
                mask = (cluster_ids == id)
                masks.append(torch.tensor(mask))
            masks = torch.stack(masks)
            
            structure = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]

            for mask in masks:
                l, n = label(mask, structure=structure)
                for lb in range(1, n + 1):
                    if np.sum(lb == l) <= island:
                        mask[lb == l] = 0
                
                reversed_mask = mask.logical_not()
                l, n = label(reversed_mask, structure=structure)
                for lb in range(1, n + 1):
                    if np.sum(lb == l) <= valley:
                        mask[lb == l] = 1
            
            
       
            
            masks = masks[:, None].repeat(1, 3, 1, 1).cuda()
            masks = TF.resize(masks, (H, W), interpolation=InterpolationMode.NEAREST)
            
            # 正常不处理
            # cluster_img = np.zeros((448, 448, 3))
            # for i in range(32):
            #     for j in range(32):
            #         patch_label = cluster_ids[i * 32 + j]
            #         cluster_img[i*14:(i+1)*14, j*14:(j+1)*14] = patch_label
                    
            # # masks.shape: (ids, C, H, W)
            # masks = []
            # for id in np.unique(cluster_ids):
            #     mask = (cluster_img == id)
            #     masks.append(torch.tensor(mask))
            # masks = torch.stack(masks).permute(0, 3, 1, 2)
            # masks = TF.resize(masks, size=(H, W))
            
            img_repeated = img[None].expand_as(masks)

            
            mask_imgs = torch.zeros_like(img_repeated).cuda()
            remain_imgs = torch.zeros_like(img_repeated).cuda()
            mask_imgs[masks] = img_repeated[masks]
            remain_imgs[masks.logical_not()] = img_repeated[masks.logical_not()]
            
            # Render images
            # Mix-up (Background brightness unchanged)
            color_mode = args.color
            if color_mode == 'R':
                rendered_imgs = (mask_imgs * 0.6 + masks * Red * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Red * 0.4
            elif color_mode == 'G':
                rendered_imgs = (mask_imgs * 0.6 + masks * Green * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Green * 0.4
            elif color_mode == 'B':
                rendered_imgs = (mask_imgs * 0.6 + masks * Blue * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Blue * 0.4
            else:
                raise ValueError('Color not supported: {}'.format(color_mode))
            
            pred_sem_seg = torch.zeros(img.shape[1], img.shape[2]).cuda()
            rendered_name_dir = os.path.join(rendered_dir, img_name)
            os.makedirs(rendered_name_dir, exist_ok=True)
            for id in range(num_clusters):
                rendered_img = rendered_imgs[id].permute(1,2,0).cpu().numpy().astype(np.uint8)
                img_path = os.path.join(rendered_name_dir, "{}.jpg".format(id))
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, rendered_img)

                with torch.inference_mode():
                    # image = Image.open(img_path).convert('RGB')
                    # image = vis_processor(image)[None]

                    # text = prepare_texts(question, conv_temp)
                    # responses = model.generate(image, text, max_new_tokens=100, do_sample=False)
                    responses, _ = model.chat(tokenizer, queries=[prompt_template.format(img_path=img_path, class_names=classes_str)], history=None)

                    print (responses)
                    # if img_name == '2007_001175':
                    # #     ipdb.set_trace()

                    #     print (answer)
                try:
                    class_id = name2id[responses[0]]
                    pred_sem_seg[masks[id, 0]] = class_id
                except:
                    continue
            if img_name == '2007_001175':
                np.save('./test.npy', pred_sem_seg.cpu().numpy().astype(np.uint8))
            np.save(os.path.join(semseg_save_dir, img_name+".npy"), pred_sem_seg.cpu().numpy().astype(np.uint8)[None])
    elif args.cluster_method == 'point':
        for data in val_loader:
            img_path = data['img_path'][0]
            img_name = img_path.split('/')[-1].split('.')[0]
            print (img_name)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[0], img.shape[1]
            img = torch.tensor(img.transpose(2, 0, 1)).cuda()
            
            feature = [None]
            def hook_fn(m, fea_in, fea_out):
                feature[0] = fea_out.detach().cpu()
            # Hook visual features
            with torch.inference_mode():
                # hooker = FeatureHooker(layer=model.transformer.visual.transformer.resblocks[47])
                handle = model.transformer.visual.transformer.resblocks[47].register_forward_hook(hook_fn)
                
                queries = ['<img>{}</img>Describe this image.'.format(img_path)]
                _, _ = model.chat(tokenizer=tokenizer, queries=queries, history=None)
                
                # image = Image.open(img_path).convert('RGB')
                # image = vis_processor(image)[None]
                # text = prepare_texts(('Describe this image.', ), conv_temp)
                # _ = model.generate(image, text, max_new_tokens=100, do_sample=False)

                # feature = hooker.fea[:, 0].float()
                feature = feature[0][:, 0].float()
                # feature = feature[0][0, 1:, ...].float()
                
                # hooker.handle.remove()
                handle.remove()
                
            num_clusters = n_clusters
            kmeans = KMeans(n_clusters=num_clusters)
            cluster_ids = kmeans.fit_predict(feature)
            
            # 离群点和空缺处理
            cluster_ids = np.resize(cluster_ids, (32, 32))
            masks = []
            for id in np.unique(cluster_ids):
                mask = (cluster_ids == id)
                masks.append(torch.tensor(mask))
            masks = torch.stack(masks).byte()
            
            structure = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]

            for mask in masks:
                l, n = label(mask, structure=structure)
                for lb in range(1, n + 1):
                    if np.sum(lb == l) <= island:
                        mask[lb == l] = 0
                
                reversed_mask = mask.logical_not()
                l, n = label(reversed_mask, structure=structure)
                for lb in range(1, n + 1):
                    if np.sum(lb == l) <= valley:
                        mask[lb == l] = 1
            
            # 加了一步形态学操作
            # kernel = np.ones((kernel_size, kernel_size)).astype(np.uint8)
            # for i in range(masks.shape[0]):
            #     mask = masks[i].cpu().numpy()
            #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            #     # mask = cv2.dilate(mask, kernel, iterations = 1)
            #     # mask = cv2.erode(mask, kernel, iterations = 1)
            #     masks[i] = torch.from_numpy(mask)
                    
            masks = masks[:, None].repeat(1, 3, 1, 1).cuda()
            masks = TF.resize(masks, (H, W), interpolation=InterpolationMode.NEAREST)
            img_repeated = img[None].expand_as(masks)
            
            # 计算质心（弃用了）
            _, _, H, W = masks.shape
            y_coords = torch.arange(H)[None, None, :, None].expand_as(masks).cuda()
            x_coords = torch.arange(W)[None, None, None, :].expand_as(masks).cuda()

            y_weighted = masks * y_coords
            x_weighted = masks * x_coords

            y_center = y_weighted.sum(dim=(2, 3)) / torch.where(y_weighted != 0, 1, 0).sum(dim=(2, 3))
            x_center = x_weighted.sum(dim=(2, 3)) / torch.where(x_weighted != 0, 1, 0).sum(dim=(2, 3))
            
            color_mode = args.color
            mask_imgs = torch.zeros_like(img_repeated).cuda()
            remain_imgs = torch.zeros_like(img_repeated).cuda()
            mask_imgs[masks] = img_repeated[masks]
            remain_imgs[masks.logical_not()] = img_repeated[masks.logical_not()]
            if color_mode == 'R':
                rendered_imgs = (mask_imgs * 0.6 + masks * Red * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Red * 0.4
            elif color_mode == 'G':
                rendered_imgs = (mask_imgs * 0.6 + masks * Green * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Green * 0.4
            elif color_mode == 'B':
                rendered_imgs = (mask_imgs * 0.6 + masks * Blue * 0.4) + remain_imgs
                # rendered_imgs = img_repeated * 0.6 + masks * Blue * 0.4
            else:
                raise ValueError('Color not supported: {}'.format(color_mode))
            
            # print ("x_center: {}".format(x_center))
            # print ("y_center: {}".format(y_center))
            # print ("masks: {}".format(masks.shape))
            # print ("x: {}".format(torch.where(x_weighted != 0, 1, 0).sum(dim=(2, 3))))
            # print ("y: {}".format(torch.where(y_weighted != 0, 1, 0).sum(dim=(2, 3))))
            
            pred_sem_seg = torch.zeros(img.shape[1], img.shape[2]).cuda()
            rendered_name_dir = os.path.join(rendered_dir, img_name)
            os.makedirs(rendered_name_dir, exist_ok=True)
            for id in range(num_clusters):
                coords = torch.nonzero(masks[id][0])
                n = min(7, coords.shape[0])
                sampled_coords = coords[torch.randperm(coords.shape[0])[:n]]
                results = []
                for y, x in sampled_coords:
                    
                    rendered_img = img.clone().permute(1,2,0).cpu().numpy().astype(np.uint8)
                    img_path = os.path.join(rendered_name_dir, "{}.jpg".format(id))
                    # try:
                    #     cv2.circle(rendered_img, (int(np.round(x_center[id][0].item())), int(np.round(y_center[id][0].item()))), 6, (0, 255, 0), -1)
                    # except:
                    #     continue
                    cv2.circle(rendered_img, (int(x), int(y)), 6, (0, 255, 0), -1)
                    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, rendered_img)
                    

                    with torch.inference_mode():
                        # image = Image.open(img_path).convert('RGB')
                        # image = vis_processor(image)[None]

                        # text = prepare_texts(question, conv_temp)
                        # responses = model.generate(image, text, max_new_tokens=100, do_sample=False)
                        responses, _ = model.chat(tokenizer, queries=[prompt_template.format(img_path=img_path, class_names=classes_str)], history=None)
                        try:
                            results.append(responses[0])
                        except:
                            continue
                        # print (responses)
                        # if img_name == '2007_001175':
                        # #     ipdb.set_trace()

                        #     print (answer)
                if len(results) == 0:
                    continue
                
                counter = Counter(results)
                response = counter.most_common(1)[0][0]
                # print (results)
                try:
                    rendered_img = rendered_imgs[id].permute(1,2,0).cpu().numpy().astype(np.uint8)
                    img_path = os.path.join(rendered_name_dir, "{}_{}.jpg".format(id, response))
                    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, rendered_img)
                    
                    class_id = name2id[response]
                    # if class_id == 0:
                    #     class_id = 255
                    # pred_sem_seg[int(np.round(y_center[id][0].item())), int(np.round(x_center[id][0].item()))] = class_id
                    pred_sem_seg[masks[id, 0]] = class_id
                except:
                    continue
            np.save(os.path.join(semseg_save_dir, img_name+".npy"), pred_sem_seg.cpu().numpy().astype(np.uint8)[None])
            
            
    elif args.cluster_method == 'superpixel':
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
                    queries.append(prompt_template.format(img_path=os.path.join(rendered_img_path[0], id), color=color_full[args.color], class_names=classes_str))

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
    else:
        raise ValueError('Cluster method not supported: {}'.format(args.cluster_method))
    
if __name__ == '__main__':
    main()