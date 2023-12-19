#!/bin/bash
DATASET='context'
rm -rf ../data/$DATASET/rendered_img/*
rm -rf ../data/$DATASET/sem_seg_preds/*
# rm -rf ../data/$DATASET/superpixel_img/*

COLOR="G"
SLIC_MODE="scikit"
SEG_NUM=30
DATA_ROOT="../data"
GPU_NUM=2

SAVE_SP_COMMAND="python save_superpixel.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM "

SAVE_RENDERED_COMMAND="python save_rendered.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR "

# env CUDA_VISIBLE_DEVICES=5,7 指定显卡加这个
SAVE_SEMSEG_COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM ddp_voc_inference.py \
--data_root $DATA_ROOT --dataset $DATASET --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR --batch_size 1 --cluster_method point"

FEATURES_COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM save_features.py \
--data_root $DATA_ROOT --dataset $DATASET --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR --batch_size 1 --cluster_method point"

# TEST_COMMAND="python superpixel_test.py \
# --data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
# --color $COLOR "

# env CUDA_VISIBLE_DEVICES=5,7 
TEST_COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM superpixel_test.py \
--data_root $DATA_ROOT --dataset $DATASET --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR "




# $SAVE_SP_COMMAND
# $SAVE_RENDERED_COMMAND

### 推理
# $SAVE_SEMSEG_COMMAND

### 获取cluster
$FEATURES_COMMAND

### gt推理
# python -m torch.distributed.launch --nproc_per_node=$GPU_NUM gt_inference.py

### 测试
# echo "Testing..."
# $TEST_COMMAND
