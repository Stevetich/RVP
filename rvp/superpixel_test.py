import itertools
import numpy as np
import argparse
import os
import os.path as osp
import cv2
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from multiprocessing import Pool
from skimage.segmentation import slic

# from fast_slic import SlicAvx2
from fast_slic import Slic

from mmseg.datasets.transforms import *
from mmseg.datasets import PascalVOCDataset
from mmengine.structures import PixelData

import time

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import warnings
warnings.simplefilter(action='ignore', category=Warning)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('--config', default='../mmsegmentation/my_configs/pascal_voc12_base.py', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    
    parser.add_argument('--data_root', type=str, default='../data', help='Location of the data.')
    parser.add_argument('--slic_mode', type=str, default='scikit', help='[scikit | fast] Superpixel method mode.')
    parser.add_argument('--seg_num', type=int, default=30, help='Superpixel method mode.')
    parser.add_argument('--color', type=str, default='B', help='[R | G | B] Color of superpixel mask.')
    
    
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()
    
    # DDP
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Directories
    voc_base_dir = "VOCdevkit/VOC2012/JPEGImages"
    superpixel_base_dir = "superpixel_img"
    rendered_base_dir = "rendered_img"
    semseg_base_dir = "sem_seg_preds"
    
    slic_method_name = args.slic_mode + str(args.seg_num)
    voc_dir = os.path.join(args.data_root, voc_base_dir)
    superpixel_dir = os.path.join(args.data_root, superpixel_base_dir, slic_method_name)
    semseg_dir = os.path.join(args.data_root, semseg_base_dir, slic_method_name, args.color)
    
    assert os.path.isdir(voc_dir), 'Not a valid voc image dir: {}'.format(voc_dir)
    assert os.path.isdir(superpixel_dir), 'Not a valid superpixel image dir: {}'.format(superpixel_dir)
    assert os.path.isdir(semseg_dir), 'Not a valid semseg prediction image dir: {}'.format(semseg_dir)


    if rank == 0:
        print ('Voc original images dir: {}'.format(voc_dir))
        print ('Superpixel images dir: {}'.format(superpixel_dir))
        print ('Semantic segmentation predictions save dir: {}'.format(semseg_dir))
        
        print ('Slic method name: {}'.format(slic_method_name))
        print ('Color mode: {}'.format(args.color))

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
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

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



    # build the runner from config
    runner = Runner.from_cfg(cfg)
    val_dataloader = runner.val_dataloader
    evaluator = runner.val_evaluator

    runner.call_hook('before_val')
    runner.call_hook('before_val_epoch')
    runner.model.eval()

    # sem_pred_dir_R = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/"
    # sem_pred_dir_G = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/G/"
    # sem_pred_dir_B = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/B/"
    sem_pred_dir_B = semseg_dir
    
    # print("=== DataLoader Details ===")
    # for key, value in val_dataloader.__dict__.items():
    #     print(f"{key}: {value}")

    img_names = []
    for idx, data_batch in enumerate(val_dataloader):
        runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)
        
        pseudo_outputs = data_batch['data_samples'][0]
        img_name = pseudo_outputs.img_path.split('/')[-1]
        img_names.append(img_name)
        # print (img_name)
        
        
        # sem_pred_R = cv2.imread(os.path.join(sem_pred_dir_B, img_name))
        # sem_pred_R = torch.tensor(sem_pred_R)[:,:,[0]].permute(2,0,1)
        # sem_pred_G = cv2.imread(os.path.join(sem_pred_dir_B, img_name))
        # sem_pred_G = torch.tensor(sem_pred_G)[:,:,[0]].permute(2,0,1)
        
        sem_pred_B = cv2.imread(os.path.join(sem_pred_dir_B, img_name))
        sem_pred_B = torch.tensor(sem_pred_B)[:,:,[0]].permute(2,0,1)

        # sem_pred = torch.zeros_like(sem_pred_R)
        # sem_pred_merge = torch.cat([sem_pred_R, sem_pred_G, sem_pred_B], dim=0)
        
        # # vote
        # for i in range(sem_pred_merge.shape[1]):
        #     for j in range(sem_pred_merge.shape[2]):
        #         sem_pred[:, i, j] = torch.argmax(torch.bincount(sem_pred_merge[:, i, j]))
        # # print ('sem_pred: {}, shape: {}'.format(sem_pred, sem_pred.shape))

        pred_sem_seg = PixelData()
        # pred_sem_seg.data = pseudo_outputs.gt_sem_seg.data
        pred_sem_seg.data = sem_pred_B
        pseudo_outputs.pred_sem_seg = pred_sem_seg
        pseudo_outputs = [pseudo_outputs]


        evaluator.process(data_samples=pseudo_outputs, data_batch=data_batch)
        runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=pseudo_outputs)
        
        if idx == 99:
            break
        
    # dist.barrier()
    
    # merge_img_names = [None for _ in range(world_size)]
    # dist.all_gather_object(merge_img_names, img_names)
    # merge_img_names = [x for x in itertools.chain.from_iterable(merge_img_names)]
    # if rank == 0:
    #     print (len(merge_img_names))
    
    # dist.barrier()
        
    metrics = evaluator.evaluate(len(val_dataloader.dataset))
    runner.call_hook('after_val_epoch', metrics=metrics)
    runner.call_hook('after_val')
    

if __name__ == "__main__":
    main()