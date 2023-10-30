_base_ = [
    '../configs/_base_/models/fcn_r50-d8.py', 
    '../configs/_base_/default_runtime.py',
]

dataset_type = 'PascalVOCDataset'
data_root = '/home/jy/mm/RVP/data/VOCdevkit/VOC2012/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))