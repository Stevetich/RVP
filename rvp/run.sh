#!/bin/bash
rm -rf ../data/rendered_img/*
rm -rf ../data/sem_seg_preds/*
rm -rf ../data/superpixel_img/*

COLOR="G"
SLIC_MODE="scikit"
SEG_NUM=30
DATA_ROOT="../data"
GPU_NUM=4

SAVE_SP_COMMAND="python save_superpixel.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM "

SAVE_RENDERED_COMMAND="python save_rendered.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR "

SAVE_SEMSEG_COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM ddp_voc_inference.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR --batch_size 1 --cluster_method feature_cluster"

# TEST_COMMAND="python superpixel_test.py \
# --data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
# --color $COLOR "

TEST_COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM superpixel_test.py \
--data_root $DATA_ROOT --slic_mode $SLIC_MODE --seg_num $SEG_NUM \
--color $COLOR "



# echo "Saving super pixel images..." 
# $SAVE_SP_COMMAND

# echo "Saving rendered images..."
# $SAVE_RENDERED_COMMAND

echo "Saving semantic segmentation predictions..."
$SAVE_SEMSEG_COMMAND

echo "Testing..."
$TEST_COMMAND
