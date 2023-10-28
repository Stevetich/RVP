import numpy as np
import argparse
import os
import os.path as osp
import cv2
import torch

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
    masked_img = cv2.addWeighted(im, 0.6, mask.astype(np.uint8), 0.4, 0)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == "__main__":

    # test_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    #     # add loading annotation after ``Resize`` because ground truth
    #     # does not need to do resize data transform
    #     dict(type='LoadAnnotations'),
    #     dict(type='PackSegInputs')
    # ]

    # val_dataset = PascalVOCDataset(
    #     data_root = 'data/VOCdevkit/VOC2012',
    #     data_prefix=dict(
    #         img_path='JPEGImages', seg_map_path='SegmentationClass'),
    #     ann_file='ImageSets/Segmentation/val.txt',
    #     pipeline=test_pipeline
    # )

    args = parse_args()

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

    sem_pred_dir_R = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/"
    sem_pred_dir_G = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/G/"
    sem_pred_dir_B = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/jiaoyang/Qwen-VL/sem_seg_preds/VOC2012/B/"

    for idx, data_batch in enumerate(val_dataloader):
        runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)
        
        # # outputs = runner.model.val_step(data_batch)
        # from mmcv.transforms import LoadImageFromFile
        # from mmseg.datasets.transforms import LoadAnnotations

        # pp1 = LoadImageFromFile(ignore_empty=False, to_float32=False, color_type='color', imdecode_backend='cv2', backend_args=None)
        # pp2 = LoadAnnotations(reduce_zero_label=None, imdecode_backend='pillow', backend_args=None)

        # import ipdb
        # ipdb.set_trace()

        # pp1(data_batch)

        
        pseudo_outputs = data_batch['data_samples'][0]
        img_name = pseudo_outputs.img_path.split('/')[-1]
        # sem_pred = cv2.imread(os.path.join(sem_pred_dir, img_name.split('.')[0]+".png"))
        # sem_pred = torch.tensor(sem_pred)[:,:,[0]].permute(2,0,1)
        sem_pred_R = cv2.imread(os.path.join(sem_pred_dir_R, img_name.split('.')[0]+".png"))
        sem_pred_R = torch.tensor(sem_pred_R)[:,:,[0]].permute(2,0,1)
        sem_pred_G = cv2.imread(os.path.join(sem_pred_dir_G, img_name.split('.')[0]+".png"))
        sem_pred_G = torch.tensor(sem_pred_G)[:,:,[0]].permute(2,0,1)
        sem_pred_B = cv2.imread(os.path.join(sem_pred_dir_B, img_name.split('.')[0]+".png"))
        sem_pred_B = torch.tensor(sem_pred_B)[:,:,[0]].permute(2,0,1)

        sem_pred = torch.zeros_like(sem_pred_R)
        sem_pred_merge = torch.cat([sem_pred_R, sem_pred_G, sem_pred_B], dim=0)
        for i in range(sem_pred_merge.shape[1]):
            for j in range(sem_pred_merge.shape[2]):
                sem_pred[:, i, j] = torch.argmax(torch.bincount(sem_pred_merge[:, i, j]))

        pred_sem_seg = PixelData()
        # pred_sem_seg.data = pseudo_outputs.gt_sem_seg.data
        pred_sem_seg.data = sem_pred
        pseudo_outputs.pred_sem_seg = pred_sem_seg
        pseudo_outputs = [pseudo_outputs]

        # import ipdb
        # ipdb.set_trace()
        
        # Substitute the 'pred_sem_seg.data' filed reservied in outputs
        # Below is an example when the batchsize is 1
        # outputs[0].pred_sem_seg.data = outputs[0].gt_sem_seg.data
        
        # evaluator.process(data_samples=outputs, data_batch=data_batch)

        # runner.call_hook(
        #     'after_val_iter',
        #     batch_idx=idx,
        #     data_batch=data_batch,
        #     outputs=outputs)

        evaluator.process(data_samples=pseudo_outputs, data_batch=data_batch)
        runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=pseudo_outputs)

        if idx == 9:
            break
        
    metrics = evaluator.evaluate(len(val_dataloader.dataset))
    runner.call_hook('after_val_epoch', metrics=metrics)
    runner.call_hook('after_val')
    
        

    

    # slic = Slic(num_components=150, compactness=6)
    
    # ## render color selected as R(ed)
    # # out_base_dir = "./tmp_results/VOC2012/R/"
    # # os.makedirs(out_base_dir, exist_ok=True)

    # # color = np.array([0,0,255])
    # # color = color[None, None, :]

    # ## save superpixels
    # out_base_dir = "./tmp_results/VOC2012/sp/"
    # os.makedirs(out_base_dir, exist_ok=True)

    # for id in range(len(val_dataset)):
    #     datasample = val_dataset[id]
    #     img = cv2.imread(datasample['img_path'])
    #     img_name = datasample['img_path'].split('/')[-1]
    #     superpixels = slic.iterate(img).astype(np.uint8)

    #     # superpixels_ids = np.unique(superpixels)
        
    #     # for superpixel_id in superpixels_ids:
    #     #     mask = (superpixels == superpixel_id).astype(np.uint8)
    #     #     mask = mask * color
    #     #     masked_img = cv2.addWeighted(img, 0.6, mask.astype(np.uint8), 0.4, 0)
    #     #     os.makedirs(os.path.join(out_base_dir, img_name.split(".")[0]), exist_ok=True)
    #     #     cv2.imwrite(os.path.join(out_base_dir, img_name.split(".")[0], str(superpixel_id)+".jpg"), masked_img)
    #     # import ipdb
    #     # ipdb.set_trace()
    #     cv2.imwrite(os.path.join(out_base_dir, img_name.split(".")[0]+".png"), superpixels)    