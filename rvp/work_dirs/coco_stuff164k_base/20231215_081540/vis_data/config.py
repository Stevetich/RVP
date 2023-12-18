crop_size = (
    512,
    512,
)
data_preprocessor = dict(size=(
    512,
    512,
))
data_root = '/home/jy/mm/coco_stuff164k/dataset'
dataset_type = 'COCOStuffDataset'
launcher = 'none'
load_from = None
model = dict(
    auxiliary_head=dict(num_classes=21),
    data_preprocessor=dict(size=(
        512,
        512,
    )),
    decode_head=dict(num_classes=21))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        data_root='/home/jy/mm/coco_stuff164k/dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='COCOStuffDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
work_dir = './work_dirs/coco_stuff164k_base'
