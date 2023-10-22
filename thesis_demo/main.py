"""The purpose of this script is to collect data for training a view translation
model specifically to run vehicle to vehicle translation and control with.

This script will randomly spawn sensors and collect translation data like in
collect_randomized_points_3d_mult_sensors_v2.py, but this one will modify the
translation parameters.

Additionally, this will randomly spawn vehicles in the road so that the translation
will learn to reproduce vehicles in the road so that the translation algorithm
can be used for obstacle avoidance.
"""

# ==============================================================================
# -- set carla path -------------------------------------------------------------------
# ==============================================================================

import os
import sys

CARLA_EGG_PATH = os.environ.get('CARLA_EGG')

try:
    sys.path.append(CARLA_EGG_PATH)
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import random
import time
import numpy as np
import cv2
import shutil
import pygame
import queue
import json
import glob
from tqdm import tqdm
import argparse

from sync_mode import CarlaSyncMode

IM_HEIGHT = 256
IM_WIDTH = 256
    
def main():

    try:
        # carla boilerplate variables
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        clock = pygame.time.Clock()
        
        ########################################################################
        # define sensor parameters (fine tune data control)
        ########################################################################
        
        # the model was trained to perform translation to distances within 3D
        # x, y, z, and yaw bounds. define these bounds here so that we can
        # try to keep the drone within these distance bounds from the vehicle
        
        # how far can an auxiliary sensor spawn from the initial sensor
        translation_distance_bounds = {
            'x': [3, 5],
            'y': [-3, 3],
            'z': [-5.5, 0],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [-10, 10]
        }
        
        blueprint_attributes = {
            'image_size_x': f"{IM_HEIGHT}",
            'image_size_y': F"{IM_WIDTH}",
            'fov': "90"
        }
        
        # define the sensors to spawn for the vehicle and drone
        sensor_types = [
            "sensor.camera.depth",                      # * depth should always be first
            # "sensor.camera.semantic_segmentation",
            # "sensor.camera.instance_segmentation",
            "sensor.camera.rgb"
        ]
        
        ########################################################################
        # start the demo
        ########################################################################
        
        with CarlaSyncMode(world, fps=30, sensor_types) as sync_mode:
            # ensure that if you add to existing datasets, your dataset parameters
            # are the same
            
            # run simulation and collect data
            while train_img_count < num_images:
                
                # get frame information
                clock.tick()
                world_data  = sync_mode.tick(timeout=2.0)        
                snapshot = world_data[0]
                sensor_data = world_data[1:]
                sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                true_fps = clock.get_fps()
                
                # define dictionary for saving the frame data
                sensor_group_data = dict()
                images = []
                
                # save the rest of the training imgs
                save_labels = True
                for idx, data in enumerate(sensor_data):
                    # if initial sensor, specify from, else to
                    if idx // len(sensor_types) == 0:
                        data_type = 'from'
                    else:
                        data_type = 'to'
                    
                    # add image to images list
                    img = get_carla_sensor_img(data)
                    sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    train_img_name = f'{data_type}_{sensor_type}_{str(train_img_count).zfill(6)}.png'
                    images.append({
                        'img_name': train_img_name,
                        'img': img
                    })
                    # sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    # train_img_name = f'{data_type}_{sensor_type}_{str(train_img_count).zfill(6)}.png'
                    # img = save_carla_sensor_img(data,
                    #                   train_img_name,
                    #                   DATASET_PATH)
                    
                    # filter out images with extreme occlusions
                    if sensor_type == 'depth' and data_type == 'from':
                        # grab the semantic information as well
                        local_idx = idx
                        while sensor_types[local_idx % len(sensor_types)].split('.')[-1] != 'semantic_segmentation':
                            local_idx += 1
                        semantic_map = get_carla_sensor_img(sensor_data[local_idx])
                        if not proximity_filter(img,
                                                semantic_map):
                            save_labels = False
                            break
                    
                    # get img information to save to labels dictionary
                    train_img_info = dict()
                    train_img_info['img_name'] = train_img_name
                    train_img_info['location'] = sync_mode.sensor_info[idx]
                    train_img_info['sensor_type'] = sensor_type
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                    # increment train_img_count every -len(sensor_types)- img saves
                    if (idx+1) % len(sensor_types) == 0:
                        train_img_count += 1
                
                if save_labels:
                    # add frame dict to label dict
                    sensor_groups.append(sensor_group_data)
                    
                    # save images
                    for save_dict in images:
                        img_name = save_dict['img_name']
                        img = save_dict['img']
                        full_save_path = os.path.join(DATASET_PATH,
                                    img_name)
                        cv2.imwrite(full_save_path, img)
                    
                    frame_count += 1
                    pbar.update(len(sensor_data)//len(sensor_types))
                    
                # randomize sensor locations
                sync_mode.randomize_sensors()
                
            labels['data'] = sensor_groups
            if prev_label_data:
                labels['data'] += prev_label_data
            labels['sensor_params'] = sensor_params
            
            labels_path = os.path.join(DATASET_PATH,
                                       'labels.json')
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
                
        # create a synchronous mode context
        time.sleep(1)
        
    finally:
        for actor in actor_list:
            actor.destroy()
        
        # save the labels if there is an error that broke the simulation before
        # completing the data collection
        labels_path = os.path.join(DATASET_PATH,
                                       'labels.json')
        if not os.path.exists(labels_path):
            labels['data'] = sensor_groups
            labels['sensor_params'] = sensor_params
            
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
            
        print("All cleaned up!")
        
def verify_dataset():
    """This function is meant to ensure that the above collected dataset adheres
    to the following rules:
    
    1. Every img name in the json labels file has a corresponding saved image in
    the dataset file path.
    2. Every saved image file has a corresponding json labels file img name.
    3. Every collected image from each sensor is sequential with no gaps.
    
    """
    # get the dataset information
    with open(DATASET_PATH+'\\labels.json', 'r') as file:
        dataset = json.load(file)
        
    sensor_params = dataset['sensor_params']
    data = dataset['data']
    
    sensor_types = len(sensor_params['sensor_types'])
    
    # get img names in the dataset
    dataset_img_names = set()
    for group_info in data:
        for img in group_info:
            dataset_img_names.add(group_info[img]['img_name'])
    
    # get all of the saved images
    saved_img_paths = glob.glob(DATASET_PATH+'\\*.png')
    
    for idx, saved_img_path in enumerate(saved_img_paths):
        saved_img_paths[idx] = saved_img_path.split('\\')[-1]
        
    saved_depth_imgs = []
    saved_rgb_imgs = []
    saved_semantic_segmentation_imgs = []
    
    for saved_img_path in saved_img_paths:
        sensor_type = saved_img_path.split('_')[1]
        if 'depth' == sensor_type:
            saved_depth_imgs.append(saved_img_path)
        elif 'rgb' == sensor_type:
            saved_rgb_imgs.append(saved_img_path)
        elif 'semantic' == sensor_type:
            saved_semantic_segmentation_imgs.append(saved_img_path)
        else:
            print('there is an error in your algorithm dummy')
    
    def get_substr(string):
        return string[-10:-4]
    
    saved_depth_imgs.sort(key=lambda x: x[-10:-4])
    saved_rgb_imgs.sort(key=lambda x: x[-10:-4])
    saved_semantic_segmentation_imgs.sort(key=lambda x: x[-10:-4])
    
    # verify that every saved image has a corresponding image name in the labels
    # file and that the images are saved in sequential order with no numbers
    # missing
    if len(dataset_img_names) != len(saved_img_paths):
        print('Mismatched saved images and labels')
    
    sensor_types = ['depth', 'rgb', 'semantic_segmentation']
    
    depth_imgs_not_saved = []
    rgb_imgs_not_saved = []
    semantic_segmentation_imgs_not_saved = []
    
    count = 0
    idx = 0
    while idx < len(saved_depth_imgs):
        img_path = saved_depth_imgs[idx]
        if int(img_path[-10:-4]) == count: 
            idx += 1
        else:
            depth_imgs_not_saved.append(str(count).zfill(6))
            
        count += 1
        
    count = 0
    idx = 0
    while idx < len(saved_rgb_imgs):
        img_path = saved_rgb_imgs[idx]
        if int(img_path[-10:-4]) == count: 
            idx += 1
        else:
            depth_imgs_not_saved.append(str(count).zfill(6))
            
        count += 1
        
    count = 0
    idx = 0
    while idx < len(saved_semantic_segmentation_imgs):
        img_path = saved_semantic_segmentation_imgs[idx]
        if int(img_path[-10:-4]) == count: 
            idx += 1
        else:
            depth_imgs_not_saved.append(str(count).zfill(6))
            
        count += 1
        
    # loop through each saved image and ensure that there is a corresponding label
    # for that image
    missing_depth_labels = []
    missing_rgb_labels = []
    missing_semantic_segmentation_labels = []
    
    for img_name in saved_depth_imgs:
        if img_name not in dataset_img_names:
            missing_depth_labels.append(img_name)
            
    for img_name in saved_rgb_imgs:
        if img_name not in dataset_img_names:
            missing_rgb_labels.append(img_name)
            
    for img_name in saved_semantic_segmentation_imgs:
        if img_name not in dataset_img_names:
            missing_semantic_segmentation_labels.append(img_name)
    
    print(f'skipped depth imgs: {len(depth_imgs_not_saved)}')
    print(f'skipped rgb imgs: {len(rgb_imgs_not_saved)}')
    print(f'skipped semantic segmentation imgs: {len(semantic_segmentation_imgs_not_saved)}')
    
    print(f'missing depth labels: {len(missing_depth_labels)}')
    print(f'missing rgb labels: {len(missing_depth_labels)}')
    print(f'missing semantic segmentation labels: {len(missing_semantic_segmentation_labels)}')
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--verify_dataset", 
                        action='store_true', 
                        help='Run dataset verification')
    
    args = parser.parse_args()
    
    try:
        if args.verify_dataset:
            verify_dataset()
        else:
            main()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')