#!/bin/bash

export DIFF_CKPT=/home/nianyli/Desktop/code/thesis/DiffViewTrans/saved_experiments/3d_rgb_depth_large_training_size/diff_large_epoch_379.ckpt
export AUTOENCODER_CKPT=/home/nianyli/Desktop/code/thesis/DiffViewTrans/saved_experiments/3d_rgb_depth_large_training_size/vqgan_rgb_depth_epoch_69.ckpt
export DIFF_CONFIG=/home/nianyli/Desktop/code/thesis/DiffViewTrans/saved_experiments/3d_rgb_depth_large_training_size/3d_view_translation_config_rgb_depth.yaml
export SAVE_DIR=/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/view_translation_demo_vid
export N_SAMPLES=100
export VID_DURATION=10
export FPS=1

python carla_view_translation_demo.py \
    --diff_ckpt=${DIFF_CKPT} \
    --autoencoder_ckpt=${AUTOENCODER_CKPT} \
    --diff_config=${DIFF_CONFIG} \
    --save_dir=${SAVE_DIR} \
    --fps=${FPS} \
    --vid_duration=${VID_DURATION} \
    --show 
