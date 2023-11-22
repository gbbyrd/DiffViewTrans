import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import cv2
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision
import random
from tqdm import tqdm

# dataset imports
from ldm.datasets.custom_datasets import RGBDepthDatasetVal, RGBDepthDatasetTrain, RGBDepthDatasetTrainMultiGPUHack, RGBDepthDatasetValMultiGPUHack

# import functions for testing performance
from fid_score.fid_score import FidScore
from skimage.metrics import structural_similarity as ssim

"""_summary_

The purpose of this script is to evaluate the performance of various conditional
translation models using FID and SSIM.

"""

rescale = lambda x: (x + 1.) / 2.
    
def normalize_labels(sensor_limits, labels):
    """Normalizes all of the LOCATION VALUES ONLY for every label in a translation
    dataset.
    """
    normalize_dict = {}
    for key in sensor_limits:
        if key == 'num_sensors' or key == 'num_sensor_types':
            continue
        if sensor_limits[key][0] == sensor_limits[key][1]:
            normalize_dict[key] = False
        else:
            normalize_dict[key] = True

    # normalize all sensor location data
    for idx, label in enumerate(labels):
        for label_idx, img_name in enumerate(label):
            for key in label[img_name]['location']:
                # TODO: Unhack the below two lines
                if key == 'sensor_bp':
                    continue
                if normalize_dict[f'{key}_limits']:
                    min_lim = sensor_limits[f'{key}_limits'][0]
                    max_lim = sensor_limits[f'{key}_limits'][1]

                    # normalize between 0 and 1
                    label[img_name]['location'][key] = (label[img_name]['location'][key] - min_lim) / (max_lim-min_lim)

                    # normalize between -1 and 1
                    label[img_name]['location'][key] = label[img_name]['location'][key] * 2 - 1
                else:
                    label[img_name]['location'][key] = 0

        labels[idx] = label
    
    return labels

def dataset_post_processing(output_dict, max_instances, num_semantic_classes):

    carla_semantic_color_map = {
        0: (0,0,0),
        1: (70,70,70),
        2: (100,40,40),
        3: (55,90,80),
        4: (220,20,60),
        5: (153,153,153),
        6: (157,234,50),
        7: (128,64,128),
        8: (244,35,232),
        9: (107,142,35),
        10: (0,0,142),
        11: (102,102,156),
        12: (220,220,0),
        13: (70,130,180),
        14: (81,0,81),
        15: (150,100,100),
        16: (230,150,140),
        17: (180,165,180),
        18: (180,165,180),
        19: (110,190,160),
        20: (170,120,150),
        21: (45,60,150),
        22: (145,170,100),
        23: (180,23,34),
        24: (120, 60, 80),
        25: (140, 0, 190),
        26: (255, 255, 255),
        27: (40, 190, 230)
    }

    # load the images, and present the depth, instance segmentation, and
    # semantic segmentation from each image

    # TODO: change the naming convention here to make it less confusing..
    # trans vs translated img is hard and confusing to distiguish..

    # translation images
    # these are the 'from' images
    trans_combined_image = output_dict['trans']

    depth_trans = np.zeros_like(trans_combined_image)
    instance_trans = np.zeros_like(trans_combined_image)
    semantic_trans = np.zeros_like(trans_combined_image)

    # denormalize the depth and break it up into RGB values to get a more
    # accurate image.. read the sensor documentation here to understand
    # https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth_trans = trans_combined_image[:,:,0]
    depth_trans = ((depth_trans + 1) / 2 * (256 * 256 * 256 - 1)).astype('uint32')
    depth_trans = np.expand_dims(depth_trans, axis=-1)
    # use bitwise to break up into different RGB values
    R = 0xFF & depth_trans
    G = (0xFF00 & depth_trans)>>8
    B = (0xFF0000 & depth_trans)>>16

    # concatenate together to get proper BGR depth img
    depth_trans = np.concatenate((B,G,R), axis=2).astype('uint8')

    # denormalize the instance labels
    instance_trans[:,:,2] = trans_combined_image[:,:,1]
    instance_trans = np.rint(((instance_trans + 1) / 2 * max_instances)).astype('uint32')

    h, w, c = instance_trans.shape
    id2color_map = dict()
    # go through all of the instance labels creating unique colors for
    # each label
    for i in range(h):
        for j in range(w):
            instance_id = instance_trans[i,j,2]
            if instance_id not in id2color_map:
                # give the identifier a random color
                id2color_map[instance_id] = np.random.randint(0, 256, size=(1, 1, 3))
            instance_trans[i, j, :] = id2color_map[instance_id]

    # denormalize the semantic image
    semantic_trans[:,:,2] = trans_combined_image[:,:,2]
    semantic_trans = np.rint(((semantic_trans + 1) / 2 * num_semantic_classes)).astype('uint8')

    semantic_id = semantic_trans[i,j,2]
    semantic_trans[i,j,:] = np.array((carla_semantic_color_map[semantic_id][2],
                                    carla_semantic_color_map[semantic_id][1],
                                    carla_semantic_color_map[semantic_id][0]))

    # ground truth
    ground_truth_combined_img = output_dict['ground']

    depth_ground = np.zeros_like(ground_truth_combined_img)
    instance_ground = np.zeros_like(ground_truth_combined_img)
    semantic_ground = np.zeros_like(ground_truth_combined_img)

    # denormalize the depth and break it up into RGB values to get a more
    # accurate image.. read the sensor documentation here to understand
    # https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth_ground = ground_truth_combined_img[:,:,0]
    depth_ground = ((depth_ground + 1) / 2 * (256 * 256 * 256 - 1)).astype('uint32')
    depth_ground = np.expand_dims(depth_ground, axis=-1)
    # use bitwise to break up into different RGB values
    R = 0xFF & depth_ground
    G = (0xFF00 & depth_ground)>>8
    B = (0xFF0000 & depth_ground)>>16

    # concatenate together to get proper BGR depth img
    depth_ground = np.concatenate((B,G,R), axis=2).astype('uint8')

    # denormalize the instance labels
    instance_ground[:,:,2] = ground_truth_combined_img[:,:,1]
    instance_ground = np.rint(((instance_ground + 1) / 2 * max_instances)).astype('uint32')

    # go through instance labels and give them colors corresponding to
    # their identifier
    for i in range(h):
        for j in range(w):
            instance_id = instance_ground[i,j,2]
            if instance_id not in id2color_map:
                # give the identifier the color black for unknown instance
                instance_ground[i, j, :] = np.array((0,0,0))
            else:
                instance_ground[i, j, :] = id2color_map[instance_id]

    # denormalize the semantic image
    semantic_ground[:,:,2] = ground_truth_combined_img[:,:,2]
    semantic_ground = np.rint(((semantic_ground + 1) / 2 * num_semantic_classes)).astype('uint8')
    for i in range(h):
        for j in range(w):
            semantic_id = semantic_ground[i,j,2]
            semantic_ground[i,j,:] = np.array((carla_semantic_color_map[semantic_id][2],
                                            carla_semantic_color_map[semantic_id][1],
                                            carla_semantic_color_map[semantic_id][0]))

    # translated img
    translated_combined_img = output_dict['translated_img']

    depth_translated = np.zeros_like(translated_combined_img)
    instance_translated = np.zeros_like(translated_combined_img)
    semantic_translated = np.zeros_like(translated_combined_img)

    # denormalize the depth and break it up into RGB values to get a more
    # accurate image.. read the sensor documentation here to understand
    # https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth_translated = translated_combined_img[:,:,0]
    depth_translated = ((depth_translated + 1) / 2 * (256 * 256 * 256 - 1)).astype('uint32')
    depth_translated = np.expand_dims(depth_translated, axis=-1)
    # use bitwise to break up into different RGB values
    R = 0xFF & depth_translated
    G = (0xFF00 & depth_translated)>>8
    B = (0xFF0000 & depth_translated)>>16

    # concatenate together to get proper BGR depth img
    depth_translated = np.concatenate((B,G,R), axis=2).astype('uint8')

    # denormalize the instance labels
    instance_translated[:,:,2] = translated_combined_img[:,:,1]
    instance_translated = np.rint(((instance_translated + 1) / 2 * max_instances)).astype('uint32')

    # go through instance labels and give them colors corresponding to
    # their identifier
    for i in range(h):
        for j in range(w):
            instance_id = instance_translated[i,j,2]
            if instance_id not in id2color_map:
                # give the identifier the color black for unknown instance
                instance_translated[i, j, :] = np.array((0,0,0))
            else:
                instance_translated[i, j, :] = id2color_map[instance_id]

    # denormalize the semantic image
    semantic_translated[:,:,2] = translated_combined_img[:,:,2]
    semantic_translated = np.rint(((semantic_translated + 1) / 2 * num_semantic_classes)).astype('uint8')
    # for i in range(h):
    #     for j in range(w):
    #         semantic_id = semantic_translated[i,j,2]
    #         semantic_translated[i,j,:] = np.array((carla_semantic_color_map[semantic_id][2],
    #                                         carla_semantic_color_map[semantic_id][1],
    #                                         carla_semantic_color_map[semantic_id][0]))
    
    instance_trans = instance_trans.astype('uint8')
    semantic_trans = semantic_trans.astype('uint8')

    instance_ground = instance_ground.astype('uint8')
    semantic_ground = semantic_ground.astype('uint8')

    instance_translated = instance_translated.astype('uint8')
    semantic_translated = semantic_translated.astype('uint8')
    
    # save the images

    # concatenate the ground and trans and display them in one image
    concatenated_trans = np.concatenate((depth_trans, instance_trans, semantic_trans), axis=1)
    concatenated_ground = np.concatenate((depth_ground, instance_ground, semantic_ground), axis=1)
    concatenated_translated = np.concatenate((depth_translated, instance_translated, semantic_translated), axis=1)
    full_concat = np.concatenate((concatenated_trans, concatenated_ground, concatenated_translated), axis=0)

    return full_concat
        
def run_translation_rgb_depth(model, opt):
    """Runs inference using rgbd images and model. Creates concatenated images
    of the results for visualization.
    """

    # create dataloader
    dataset = RGBDepthDatasetVal(opt.sample_data_folder) # for depth instance sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # sample batches from the dataloader and run inference
    img_count = 0
    for batch in dataloader:
        z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
        

        samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                shape=(3, 64, 64), 
                                                conditioning=c, verbose=False,
                                                translation_label=translation_label, 
                                                eta=1.0)

        # decode from latent with pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            to_view = torch.clamp(imgs[i], -1., 1.)
            to_view = to_view.cpu().numpy()
            to_view = np.transpose(to_view, (1, 2, 0))
            to_view = denormalize(to_view)

            # get ground truth and conditioning imgs
            ground_truth = batch['to'][i].numpy()
            from_view = batch['from'][i].numpy()

            # denormalize data images loaded from dataloader
            ground_truth = denormalize(ground_truth)
            from_view = denormalize(from_view)

            # uncomment below to display images for 5 seconds

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)
            # cv2.waitKey(5000)

            # concatenate together
            full_img = np.concatenate([from_view, ground_truth, to_view], axis=1)

            # break each 4 channel image into depth and rgb images
            full_rgb_img = np.concatenate([from_view[:, :, :3], ground_truth[:, :, :3], to_view[:, :, :3]], axis=1)
            full_depth_img = np.concatenate([from_view[:, :, -1], ground_truth[:, :, -1], to_view[:, :, -1]], axis=1)

            # save the imgs
            # cv2.imwrite(os.path.join(opt.save_dir, f'processed_sample_{img_count}.png',),
            #             processed_data)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_rgb.png'),
                        full_rgb_img)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_depth.png'),
                        full_depth_img)
            img_count += 1
            
            # break out if number of samples reached
            if img_count >= opt.n_samples:
                break
        
        if img_count >= opt.n_samples:
                break
            
def run_translation_rgb_depth_for_benchmark(model, opt, split):
    """Runs inference using depth - instance segmentation images and model. Saves
    the target images and generated images in their own folders as this the
    pytorch FID implementation that we are using requires this.
    """

    # create dataloader
    if split == 'train':
        dataset = RGBDepthDatasetTrainMultiGPUHack(opt.start, opt.total, opt.sample_data_folder)
    elif split == 'val':
        dataset = RGBDepthDatasetValMultiGPUHack(opt.start, opt.total, opt.sample_data_folder)
        
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    
    # create benchmarking dataset folders
    ground_truth_depth_dataset_path = os.path.join(opt.experiment_folder_path,
                                                   'benchmark',
                                                   split,
                                                   'ground_truth',
                                                   'depth')
    ground_truth_rgb_dataset_path = os.path.join(opt.experiment_folder_path,
                                                   'benchmark',
                                                   split,
                                                   'ground_truth',
                                                   'rgb')
    translated_depth_dataset_path = os.path.join(opt.experiment_folder_path,
                                                   'benchmark',
                                                   split,
                                                   'translated',
                                                   'depth')
    translated_rgb_dataset_path = os.path.join(opt.experiment_folder_path,
                                                   'benchmark',
                                                   split,
                                                   'translated',
                                                   'rgb')
    
    if not os.path.exists(ground_truth_depth_dataset_path):
        os.makedirs(ground_truth_depth_dataset_path)
    if not os.path.exists(ground_truth_rgb_dataset_path):
        os.makedirs(ground_truth_rgb_dataset_path)
    if not os.path.exists(translated_depth_dataset_path):
        os.makedirs(translated_depth_dataset_path)
    if not os.path.exists(translated_rgb_dataset_path):
        os.makedirs(translated_rgb_dataset_path)

    # if opt.vanilla_sample:
    #     print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    # else:
    #     model.num_timesteps = opt.custom_steps
    #     print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    # TODO: See what happens when I change the below value
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # for multi-gpu, get device ids
    # if opt.gpus is not None:
    #     gpus = opt.gpus
    #     if gpus[-1] == ',':
    #         gpus = gpus[:-1]
    #     device_ids = [int(gpu_id) for gpu_id in gpus.split(',')]
    #     sampler.model = nn.DataParallel(sampler.model, device_ids=device_ids)
        # device = torch.device('cuda:'+gpus)
        # model.to(device)
    
    
    
    # sample batches from the dataloader and run inference
    img_count = opt.start
    for batch in tqdm(dataloader):
        z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
        

        samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                shape=(3, 64, 64), 
                                                conditioning=c, verbose=False,
                                                translation_label=translation_label, 
                                                eta=1.0)

        # decode from latent with pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            to_view = torch.clamp(imgs[i], -1., 1.)
            to_view = to_view.cpu().numpy()
            to_view = np.transpose(to_view, (1, 2, 0))
            to_view = denormalize(to_view)

            # get ground truth and conditioning imgs
            ground_truth = batch['to'][i].numpy()
            from_view = batch['from'][i].numpy()

            # denormalize data images loaded from dataloader
            ground_truth = denormalize(ground_truth)
            from_view = denormalize(from_view)
            
            # get ground truth rgb and depth
            ground_truth_rgb = ground_truth[:, :, :3]
            ground_truth_depth = ground_truth[:, :, 3:]
            
            translated_view_rgb = to_view[:, :, :3]
            translated_view_depth = to_view[:, :, 3:]
            
            # save the translated and ground truth images in their own datsets
            # for FID score
            
            img_names = {
                'ground_truth_rgb': os.path.join(ground_truth_rgb_dataset_path,
                                                 f'ground_truth_rgb_{str(img_count).zfill(6)}.png'),
                'ground_truth_depth': os.path.join(ground_truth_depth_dataset_path,
                                                 f'ground_truth_depth_{str(img_count).zfill(6)}.png'),
                'translated_rgb': os.path.join(translated_rgb_dataset_path,
                                                 f'translated_rgb_{str(img_count).zfill(6)}.png'),
                'translated_depth': os.path.join(translated_depth_dataset_path,
                                                 f'translated_depth_{str(img_count).zfill(6)}.png'),
            }
            
            cv2.imwrite(img_names['ground_truth_rgb'], ground_truth_rgb)
            cv2.imwrite(img_names['ground_truth_depth'], ground_truth_depth)
            cv2.imwrite(img_names['translated_rgb'], translated_view_rgb)
            cv2.imwrite(img_names['translated_depth'], translated_view_depth)
            
            img_count += 1
            
def run_visualization_benchmarks(model, opt):
    
    dataset = RGBDepthDatasetVal(opt.sample_data_folder)
    
    visual_benchmark_folder_path = os.path.join(opt.experiment_folder_path,
                                                    'visual_benchmark')
    
    if not os.path.exists(visual_benchmark_folder_path):
            os.makedirs(visual_benchmark_folder_path)
        
    ############################################################################
    #                    ONE TO MANY VISUALIZATION BENCHMARK                   #
    ############################################################################
    num_samples = 5
    for i in range(num_samples):
        dataset_idx = random.randint(0, len(dataset))
        batch = dataset[dataset_idx]
        condition = denormalize(batch['from'])
        condition_rgb = condition[:, :, :3]
        condition_depth = condition[:, :, 3:]
        batch['to'] = torch.tensor(batch['to']).unsqueeze(0)
        batch['from'] = torch.tensor(batch['from']).unsqueeze(0)
        batch['translation_label'] = torch.tensor(batch['translation_label']).unsqueeze(0)
        
        full_rgb, full_depth = run_one_to_many_visualization_benchmark(model, opt, batch)
                
        # save the images
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'one_to_many_rgb_dataset_idx_{dataset_idx}.png'),
                    full_rgb)
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'one_to_many_depth_dataset_idx_{dataset_idx}.png'),
                    full_depth)
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'one_to_many_rgb_condition_dataset_idx_{dataset_idx}.png'),
                    condition_rgb)
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'one_to_many_depth_condition_dataset_idx_{dataset_idx}.png'),
                    condition_depth)
            
    ############################################################################
    #                                    END                                   #
    ############################################################################
    
    ############################################################################
    #                     CONTINUOUS VISUALIZATION BENCHMARK                   #
    ############################################################################
    
    axes = ['x', 'y', 'z']
    for axis in axes:
        full_rgb_img_axis = None
        full_depth_img_axis = None
        num_samples = 5
        for i in range(num_samples):
            dataset_idx = random.randint(0, len(dataset))
            batch = dataset[dataset_idx]
            condition = denormalize(batch['from'])
            condition_rgb = condition[:, :, :3]
            condition_depth = condition[:, :, 3:]
            batch['to'] = torch.tensor(batch['to']).unsqueeze(0)
            batch['from'] = torch.tensor(batch['from']).unsqueeze(0)
            batch['translation_label'] = torch.tensor(batch['translation_label']).unsqueeze(0)
            full_rgb, full_depth = run_continuous_visualization_benchmark(model, opt, batch, axis, 
                                                                          condition_rgb, condition_depth)
            
            if full_rgb_img_axis is None:
                full_rgb_img_axis = full_rgb
            elif axis == 'x' or axis == 'y' or axis == 'yaw' or axis == 'z':
                full_rgb_img_axis = np.concatenate((full_rgb_img_axis, full_rgb), axis=0)
            else:
                full_rgb_img_axis = np.concatenate((full_rgb_img_axis, full_rgb), axis=1)
                
            if full_depth_img_axis is None:
                full_depth_img_axis = full_depth
            elif axis == 'x' or axis == 'y' or axis == 'yaw' or axis == 'z':
                full_depth_img_axis = np.concatenate((full_depth_img_axis, full_depth), axis=0)
            else:
                full_depth_img_axis = np.concatenate((full_depth_img_axis, full_depth), axis=1)
                
        
        # save the images
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'continuous_rgb_axis_{axis}.png'),
                    full_rgb_img_axis)
        cv2.imwrite(os.path.join(visual_benchmark_folder_path,
                                f'continuous_depth_{axis}.png'),
                    full_depth_img_axis)

            
def run_one_to_many_visualization_benchmark(model, opt, batch):
    """Runs inference using depth - instance segmentation images and model. Saves
    the target images and generated images in their own folders as this the
    pytorch FID implementation that we are using requires this.
    """

    # create DDIM sampler object
    # TODO: See what happens when I change the below value
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # get inputs from batch
    z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
    
    # bin the y-z axis of the Translation Limits Cube into 9 sections. Translate
    # to the middle of each section and concatenate the images. Translate to the
    # middle of the x axis of the Translation Limits Cube (x = 0 in the translation
    # label...)
    y_bins = np.linspace(-1, 1, 4)
    z_bins = np.linspace(-1, 1, 4)
    
    y_locs = []
    for idx in range(len(y_bins) - 1):
        y_locs.append((y_bins[idx] + y_bins[idx+1]) / 2)
        
    z_locs = []
    for idx in range(len(z_bins) - 1):
        z_locs.append((z_bins[idx] + z_bins[idx+1]) / 2)
        
    # reverse the z locations so that the top left location is found first.. this
    # just makes it easy to orient during concatentation
    z_locs.reverse()
    
    full_rgb_z = None
    full_depth_z = None
    for idx_z, z_loc in enumerate(z_locs):
        full_rgb_y = None
        full_depth_y = None
        for idx_y, y_loc in enumerate(y_locs):
            
            # translation_label = torch.tensor([[[0, y_loc, z_loc, 0]]])
            translation_label[0][0][0] = 0
            translation_label[0][0][1] = y_loc
            translation_label[0][0][2] = z_loc
            translation_label[0][0][3] = 0
    
            samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                        shape=(3, 64, 64), 
                                                        conditioning=c, verbose=False,
                                                        translation_label=translation_label, 
                                                        eta=1.0)
            
            # decode from latent with pretrained autoencoder
            imgs = model.decode_first_stage(samples)
            
            # process translated img
            to_view = torch.clamp(imgs[0], -1., 1.)
            to_view = to_view.cpu().numpy()
            to_view = np.transpose(to_view, (1, 2, 0))
            to_view = denormalize(to_view)
            
            translated_view_rgb = to_view[:, :, :3]
            translated_view_depth = to_view[:, :, 3:]
            
            if full_rgb_y is None:
                full_rgb_y = translated_view_rgb
            else:
                full_rgb_y = np.concatenate((full_rgb_y, translated_view_rgb), axis=1)
                
            if full_depth_y is None:
                full_depth_y = translated_view_depth
            else:
                full_depth_y = np.concatenate((full_depth_y, translated_view_depth), axis=1)
            
        if full_rgb_z is None:
            full_rgb_z = full_rgb_y
        else:
            full_rgb_z = np.concatenate((full_rgb_z, full_rgb_y), axis=0)
            
        if full_depth_z is None:
            full_depth_z = full_depth_y
        else:
            full_depth_z = np.concatenate((full_depth_z, full_depth_y), axis=0)
        
    
    
    return full_rgb_z, full_depth_z

def run_continuous_visualization_benchmark(model, opt, batch, axis, 
                                           condition_rgb, condition_depth):
    """Runs inference using depth - instance segmentation images and model. Saves
    the target images and generated images in their own folders as this the
    pytorch FID implementation that we are using requires this.
    """
    
    # define axis indexes
    axis_idxs = {
        'x': 0,
        'y': 1,
        'z': 2,
        'yaw': 3
    }
    
    axis_idx = axis_idxs[axis]

    # create DDIM sampler object
    # TODO: See what happens when I change the below value
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # get inputs from batch
    z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
    
    # bin the y-z axis of the Translation Limits Cube into 9 sections. Translate
    # to the middle of each section and concatenate the images. Translate to the
    # middle of the x axis of the Translation Limits Cube (x = 0 in the translation
    # label...)
    bins = np.linspace(-1, 1, 10)
    
    locs = []
    for idx in range(len(bins) - 1):
        locs.append((bins[idx] + bins[idx+1]) / 2)
        
    # reverse the z locations so that the top left location is found first.. this
    # just makes it easy to orient during concatentation
    if axis == 'z':
        locs.reverse()
    
    full_rgb = None
    full_depth = None
    for idx, loc in enumerate(locs):
            
        # translation_label = torch.tensor([[[0, y_loc, z_loc, 0]]])
        # zero out translation label
        translation_label[0][0][0] = 0
        translation_label[0][0][1] = 0
        translation_label[0][0][2] = 0
        translation_label[0][0][3] = 0
        translation_label[0][0][axis_idx] = loc

        samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                    shape=(3, 64, 64), 
                                                    conditioning=c, verbose=False,
                                                    translation_label=translation_label, 
                                                    eta=1.0)
        
        # decode from latent with pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # process translated img
        to_view = torch.clamp(imgs[0], -1., 1.)
        to_view = to_view.cpu().numpy()
        to_view = np.transpose(to_view, (1, 2, 0))
        to_view = denormalize(to_view)
        
        translated_view_rgb = to_view[:, :, :3]
        translated_view_depth = to_view[:, :, 3:]
        
        if full_rgb is None:
            full_rgb = translated_view_rgb
        elif axis == 'x' or axis == 'y' or axis == 'yaw' or axis == 'z':
            full_rgb = np.concatenate((full_rgb, translated_view_rgb), axis=1)
        else:
            full_rgb = np.concatenate((full_rgb, translated_view_rgb), axis=0)
            
        if full_depth is None:
            full_depth = translated_view_depth
        elif axis == 'x' or axis == 'y' or axis == 'yaw' or axis == 'z':
            full_depth = np.concatenate((full_depth, translated_view_depth), axis=1)
        else:
            full_depth = np.concatenate((full_depth, translated_view_depth), axis=0)
    
    if axis == 'x' or axis == 'y' or axis == 'yaw' or axis == 'z':
        buffer = np.ones((256, 25, 3)) * 255
        full_rgb = np.concatenate((condition_rgb, buffer, full_rgb), axis=1)
        full_depth = np.concatenate((condition_depth, buffer[:, :, :1], full_depth), axis=1)
    else:
        buffer = np.ones((25, 256, 3)) * 255
        full_rgb = np.concatenate((condition_rgb, buffer, full_rgb), axis=0)
        full_depth = np.concatenate((condition_depth, buffer[:, :, :1], full_depth), axis=0)
    
    return full_rgb, full_depth

def denormalize(img): 
    """ Takes an img normalized between [-1, 1] and denormalizes to between 
    [0, 255]
    """
    img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

    return img

def benchmark(opt, split=None):
    
    def get_ssim(paths):
        """Get the average ssim of a group of imgs.
        
        Args:
            paths (List): List of len 2 containing the directory paths of the
                source and target images.
                
        Returns:
            average_ssim (float): the average SSIM value for the dataset.
        """
        
        src_imgs = glob.glob(paths[0]+'/*.png')
        target_imgs = glob.glob(paths[1]+'/*.png')
        
        ssim_sum = 0
        
        src_imgs.sort()
        target_imgs.sort()
        
        assert len(src_imgs) == len(target_imgs), "Error: Number of src and target images are not the same."
        
        for idx in tqdm(range(len(src_imgs))):
            
            # load the images
            src_img = cv2.imread(src_imgs[idx])
            target_img = cv2.imread(target_imgs[idx])
            
            # convert to grayscale
            gray_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            
            ssim_score = ssim(gray_src_img, gray_target_img)
            
            ssim_sum += ssim_score
            
        # compute the mean ssim score
        mean_ssim = ssim_sum / len(src_imgs)
        
        return mean_ssim
        
    # compare the src images to target images
    train_imgs_path = os.path.join(opt.experiment_folder_path,
                                    'benchmark',
                                    'train')
    val_imgs_path = os.path.join(opt.experiment_folder_path,
                                    'benchmark',
                                    'val')
    # training
    if split == 'train' or split == None:
        
        paths = [os.path.join(train_imgs_path, 'translated', 'rgb'),
                    os.path.join(train_imgs_path, 'ground_truth', 'rgb')]
        
        # get fid score
        print('Calculating train FID score...')
        # fid = FidScore(paths, device='cuda', batch_size=10)
        # train_fid_score = fid.calculate_fid_score()
        # print(f'Train FID score: {train_fid_score}...')
        
        print('Calculating mean train SSIM score...')
        mean_train_ssim_score = get_ssim(paths)
        print(f'Mean train SSIM score: {mean_train_ssim_score}...')
    
    # validation
    if split == 'val' or split == None:
        
        paths = [os.path.join(val_imgs_path, 'translated', 'rgb'),
                    os.path.join(val_imgs_path, 'ground_truth', 'rgb')]
        
        # get fid score
        print('Calculating val FID score...')
        fid = FidScore(paths, device='cuda', batch_size=10)
        train_fid_score = fid.calculate_fid_score()
        print(f'Val FID score: {train_fid_score}...')
        
        print('Calculating mean val SSIM score...')
        mean_train_ssim_score = get_ssim(paths)
        print(f'Mean val SSIM score: {mean_train_ssim_score}...')
        
    print('\n\nBENCHMARK COMPLETE!')
    

        
def visualize_benchmark(opt):
    benchmark_path = opt.experiment_folder_path + '/benchmark'
    
    # get ground truth images
    ground_truth_rgb_imgs = glob.glob(os.path.join(benchmark_path,
                                                  'ground_truth',
                                                  'rgb',
                                                  '*.png'))
    ground_truth_depth_imgs = glob.glob(os.path.join(benchmark_path,
                                                  'ground_truth',
                                                  'depth',
                                                  '*.png'))
    translated_rgb_imgs = glob.glob(os.path.join(benchmark_path,
                                                  'translated',
                                                  'rgb',
                                                  '*.png'))
    translated_depth_imgs = glob.glob(os.path.join(benchmark_path,
                                                  'translated',
                                                  'depth',
                                                  '*.png'))
    
    ground_truth_depth_imgs.sort()
    ground_truth_rgb_imgs.sort()
    translated_rgb_imgs.sort()
    translated_depth_imgs.sort()
    
    for idx in range(len(translated_depth_imgs)):
        
        ground_truth_rgb_img = cv2.imread(ground_truth_rgb_imgs[idx])
        ground_truth_depth_img = cv2.imread(ground_truth_depth_imgs[idx])
        translated_rgb_img = cv2.imread(translated_rgb_imgs[idx])
        translated_depth_img = cv2.imread(translated_depth_imgs[idx])
    
        concatenated_img = np.concatenate((ground_truth_rgb_img, 
                                        translated_rgb_img, 
                                        ground_truth_depth_img, 
                                        translated_depth_img), axis=1)
        
        cv2.imshow('benchmark_imgs', concatenated_img)
        cv2.waitKey(0)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="?",
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--diff_ckpt",
        type=str,
        nargs="?",
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--autoencoder_ckpt",
        type=str,
        nargs="?",
        help="autoencoder checkpoint path",
    )
    parser.add_argument(
        "--diff_config",
        type=str,
        nargs="?",
        help="diffusion model config path",
    )
    parser.add_argument(
        "--autoencoder_config",
        type=str,
        nargs="?",
        help="autoencoder config path",
    )
    parser.add_argument(
        "--sample_data_folder",
        type=str,
        nargs="?",
        help="specify the folder containing the data to use to sample translation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="?",
        help="specify the folder to save the sampled images to",
    )
    parser.add_argument(
        "--experiment_folder_path",
        type=str,
        action='store',
        nargs="?",
        help="experiment folder",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=100
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--run_inference",
        action='store_true',
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--benchmark_performance",
        action='store_true',
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--start",
        action='store',
        type=int,
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--total",
        action='store',
        type=int,
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--split",
        action='store',
        type=str,
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="diffusion model checkpoint path",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    
        
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = opt.diff_ckpt

    opt.base = [opt.diff_config] #, opt.autoencoder_config]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    if opt.visualize:
        visualize_benchmark(opt)
        exit()

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        logdir = opt.logdir
        
    print(config)
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    
    run_visualization_benchmarks(model, opt)
    