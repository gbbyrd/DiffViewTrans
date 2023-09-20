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
import json
import cv2
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision
import random

rescale = lambda x: (x + 1.) / 2.

class SampleDataset(Dataset):
    def __init__(self, sample_data_folder_path):

        raise NotImplementedError("This dataset is outdated and needs to be tested before continuing..")
        # self.base_data_folder = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256'
        # label_json_file_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256/labels.json'
        self.base_data_folder = sample_data_folder_path
        label_json_file_path = sample_data_folder_path+'/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            data = json.load(file)

        self.labels = data['data']
        self.sensor_limits = data['sensor_limits']

        self.img_info = []

        normalize_dict = {}
        for key in self.sensor_limits:
            if key == 'num_sensors' or key =='num_sensor_types':
                continue
            if self.sensor_limits[key][0] == self.sensor_limits[key][1]:
                normalize_dict[key] = False
            else:
                normalize_dict[key] = True

        # normalize all sensor location data
        for idx, label in enumerate(self.labels):
            for label_idx, img_name in enumerate(label):
                for key in label[img_name]['location']:
                    if key == 'sensor_bp':
                        continue
                    if normalize_dict[f'{key}_limits']:
                        min_lim = self.sensor_limits[f'{key}_limits'][0]
                        max_lim = self.sensor_limits[f'{key}_limits'][1]

                        # normalize between 0 and 1
                        label[img_name]['location'][key] = (label[img_name]['location'][key] - min_lim) / (max_lim-min_lim)

                        # normalize between -1 and 1
                        label[img_name]['location'][key] = label[img_name]['location'][key] * 2 - 1
                    else:
                        label[img_name]['location'][key] = 0

            self.labels[idx] = label

        # get all of the keys
        keys = list(self.labels[0].keys())
        keys.sort()

        for label in self.labels:
            for key in label:
                if key == keys[-1]:
                    continue
                info = {
                    'ground': os.path.join(self.base_data_folder, label[keys[-1]]['img_name']),
                    'aerial': os.path.join(self.base_data_folder, label[key]['img_name']),
                    'location': label[key]['location']
                }
                self.img_info.append(info)

    def __getitem__(self, idx):
        
        # generate random noise for ground sample
        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])
        # ground_img = np.random.randn(aerial_img.shape[0], aerial_img.shape[1])

        # normalize between [-1, 1]
        aerial_img = aerial_img / 127.5 - 1
        ground_img = ground_img / 127.5 - 1

        # get the location
        x = self.img_info[idx]['location']['x']
        y = self.img_info[idx]['location']['y']
        z = self.img_info[idx]['location']['z']
        z_angle = self.img_info[idx]['location']['z_angle']

        output_dict = {
            'ground': ground_img,
            'aerial': aerial_img,
            'location': np.array([[x, y, z, z_angle]], dtype='float32')
        }

        return output_dict
    
    def __len__(self):
        return len(self.img_info)
    
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
    
class SampleInstanceDepthDatasetBase(Dataset):
    """This dataset takes in a path to the data folder, reads the labels file,
    and produces a dictionary of normalized numpy arrays in the form:
    
    output_dict = {
        'ground' (np.array): ground_img,
        'trans' (np.array): trans_img,
        'location' (np.array): relative location of the ground img from the
            trans img
    }
    """

    def __init__(self, sample_data_folder_path):
        self.base_data_folder = sample_data_folder_path
        label_json_file_path = self.base_data_folder+'/labels.json'

        # instance segmentation specifications
        self.max_instances = 255        # chosen to be the max value found in an image RGB value
        self.num_semantic_classes = 27  # carla spec

        # read label data
        with open(label_json_file_path, 'r') as file:
            data = json.load(file)

        self.labels = data['data']
        self.sensor_limits = data['sensor_limits']
        num_sensor_types = self.sensor_limits['num_sensor_types']

        # the below normalizes the sensor location information, NOT the images
        self.labels = normalize_labels(self.sensor_limits, self.labels)

        # create translation dataset

        # Note: there may be multiple translation locations translating to one ground truth

        data_pairs = []
        for label in self.labels:
            label = list(label.values())
            # get the ground truth image set (ground truth always at the end of the label list)
            ground_truth = label[-num_sensor_types:]
            translation_labels = label[:-num_sensor_types]
            idx = 0

            # get pairs of data from each translation image in the group
            while idx+(num_sensor_types-1) < len(translation_labels):
                trans_group_info = dict()
                trans_group_info['location'] = translation_labels[idx]['location']
                for _ in range(num_sensor_types):
                    img_info = translation_labels[idx]
                    sensor_name = img_info['img_name']
                    sensor_type = img_info['img_name'].split('_')[0]+'_trans'
                    trans_group_info[sensor_type] = sensor_name
                    idx += 1

                # add ground truth to image group
                for img in ground_truth:
                    sensor_name = img['img_name']
                    sensor_type = img['img_name'].split('_')[0]+'_ground'
                    trans_group_info[sensor_type] = sensor_name

                data_pairs.append(trans_group_info)

        self.data_pairs = data_pairs

    def __getitem__(self, idx):
        data = self.data_pairs[idx]

        # get the normalized instance images
        instance_ground_path = self._set_path(data['instance_ground'])
        instance_trans_path = self._set_path(data['instance_trans'])

        # return 256 x 256 x 2 np.array's of instance and semantic ids
        # channel 0: instance id
        # channel 1: semantic id
        instance_img_trans, instance_img_ground = self._preprocess_instance_segmentation(instance_trans_path, 
                                                                                         instance_ground_path)
        
        # get the depth images and normalize
        depth_img_trans_path = self._set_path(data['depth_trans'])
        depth_img_ground_path = self._set_path(data['depth_ground'])

        # return 256 x 256 x 1 np.arrays of normalized depth information
        depth_img_trans = self._preprocess_depth(depth_img_trans_path)
        depth_img_ground = self._preprocess_depth(depth_img_ground_path)

        # concatenate the images together channel wise to get your ground and
        # translation images
        ground_img = np.concatenate((depth_img_ground, instance_img_ground), axis=2)
        trans_img = np.concatenate((depth_img_trans, instance_img_trans), axis=2)

        # get the location
        x = data['location']['x']
        y = data['location']['y']
        z = data['location']['z']
        z_angle = data['location']['z_angle']

        # uncomment below to see if model is learning without translation label

        # print(f'original y: {y}')
        # # randomize y
        # y = random.random() * 2 - 1
        # print(f'new_y: {y}')

        output_dict = {
            'ground': ground_img,
            'trans': trans_img,
            'location': np.array([[x, y, z, z_angle]], dtype='float32')
        }

        return output_dict
    
    def __len__(self):
        return len(self.data_pairs)
    
    def _set_path(self, img_name):
        return os.path.join(self.base_data_folder, img_name)
    
    def _preprocess_depth(self, depth_img_path):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = cv2.imread(depth_img_path)
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        depth_img = depth_img * 2 - 1

        return np.expand_dims(depth_img, axis=-1)

    def _preprocess_instance_segmentation(self, 
                                          img_path_trans, 
                                          img_path_ground_truth):
        """Takes in a h x w x 3 instance segmentation image, where the instance
        ID is in the B and G values and the semantic id is in the R value. 
        
        1. Assigns each instance id a random id between 0 and the max number of
            instance ids allowed.
        2. Normalizes everything between 0 and 1 using the max num instaces for 
            channel 1 and max num semantic classes for channel 2. 
        3. Returns a h x w x 2 numpy array to be concatenated with the depth 
            information.
        """
        
        # get list of possible instance IDs, leave 0 value for 'unknown instance id'
        instance_labels = np.linspace(1, self.max_instances, self.max_instances).tolist()
        # we shuffle the labels to randomize the instance IDs and reduce bias in
        # the model towards small instance ID values
        random.shuffle(instance_labels)

        # get all instance ID's from the translated image first
        trans_img = cv2.imread(img_path_trans).astype('float64')

        # *** NOTE: cv2 reads images in [b, g, r] format

        h, w, _ = trans_img.shape
        id_map = dict()
        for i in range(h):
            for j in range(w):

                # get unique identifier
                id_0 = str(trans_img[i,j,0])+str(trans_img[i,j,1])

                # if new img id, add to dictionary
                if id_0 not in id_map:
                    if len(instance_labels) == 0:
                        # notify if max instances are exceeded
                        print('Max instances exceeded.. marking as 0')
                        id_map[id_0] = 0
                    else:
                        id_map[id_0] = instance_labels.pop()

                trans_img[i,j,1] = id_map[id_0]

        # modify all instance ID's for the ground truth img, label the instance id 
        # and semantic id as 0 if not seen in trans img. (we dont want the model
        # to try to predict what an unseen area is. Just that the area is unseen)
        ground_truth_img = cv2.imread(img_path_ground_truth).astype('float64')
        for i in range(h):
            for j in range(w):

                # get unique identifier
                id_0 = str(ground_truth_img[i,j,0])+str(ground_truth_img[i,j,1])

                if id_0 not in id_map:
                    ground_truth_img[i,j,1] = 0
                    ground_truth_img[i,j,2] = 0
                else:
                    ground_truth_img[i,j,1] = id_map[id_0]

        # normalize the images between [-1, 1]
        trans_img[:,:,1] = (trans_img[:,:,1] / self.max_instances) * 2 - 1
        ground_truth_img[:,:,1] = (ground_truth_img[:,:,1] / self.max_instances) * 2 - 1
        if np.max(trans_img[:,:,2]) > self.num_semantic_classes:
            print('Warning: Semantic class exceeds maximum')
        if np.max(ground_truth_img[:,:,2]) > self.num_semantic_classes:
            print('Warning: Semantic class exceeds maximum')
        trans_img[:,:,2] = (trans_img[:,:,2] / self.num_semantic_classes) * 2 - 1
        ground_truth_img[:,:,2] = (ground_truth_img[:,:,2] / self.num_semantic_classes) * 2 - 1

        return trans_img[:,:,1:], ground_truth_img[:,:,1:]

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

def run_translation(model, opt):
    """Runs inference using rgb images and model.
    """

    # TODO: VERIFY THAT THIS FUNCTION STILL WORKS (outdated function)
    raise NotImplementedError ("This is an old function and must be tested and verified before continuing!")

    # create dataloader
    dataset = SampleDataset(opt.sample_data_folder) # for rgb sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
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
        
        # decode from latent using pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            trans_img = torch.clamp(imgs[i], -1., 1.)
            trans_img = trans_img.cpu().numpy()
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = denormalize(trans_img)

            # get ground truth and conditioning imgs
            ground_truth = batch['ground'][i].numpy()
            aerial_view = batch['trans'][i].numpy()

            # denormalize images loaded froma dataloader
            ground_truth = denormalize(ground_truth)
            aerial_view = denormalize(aerial_view)

            # uncomment below to display images for 5 seconds

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)
            # cv2.waitKey(5000)

            # concatenate together
            full_img = np.concatenate([aerial_view, ground_truth, trans_img], axis=1)

            # save the img
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}.png'),
                        full_img)
            img_count += 1
            
            if img_count >= opt.n_samples:
                break
        
        if img_count >= opt.n_samples:
                break

def run_translation_depth_instance(model, opt):
    """Runs inference using depth - instance segmentation images and model.
    """

    # create dataloader
    dataset = SampleInstanceDepthDatasetBase(opt.sample_data_folder) # for depth instance sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

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
            trans_img = torch.clamp(imgs[i], -1., 1.)
            trans_img = trans_img.cpu().numpy()
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = denormalize(trans_img)

            # get ground truth and conditioning imgs
            ground_truth = batch['ground'][i].numpy()
            trans_view = batch['trans'][i].numpy()

            # post process to get separate depth, instance, and semantic images
            processed_data = dataset_post_processing({'trans': trans_view,
                                                      'ground': ground_truth,
                                                      'translated_img': trans_img},
                                                      dataset.max_instances,
                                                      dataset.num_semantic_classes)

            # denormalize data images loaded from dataloader
            ground_truth = denormalize(ground_truth)
            trans_view = denormalize(trans_view)

            # uncomment below to display images for 5 seconds

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)
            # cv2.waitKey(5000)

            # concatenate together
            full_img = np.concatenate([trans_view, ground_truth, trans_img], axis=1)

            # save the imgs
            cv2.imwrite(os.path.join(opt.save_dir, f'processed_sample_{img_count}.png',),
                        processed_data)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}.png'),
                        full_img)
            img_count += 1
            
            # break out if number of samples reached
            if img_count >= opt.n_samples:
                break
        
        if img_count >= opt.n_samples:
                break

def denormalize(img): 
    """ Takes an img normalized between [-1, 1] and denormalizes to between 
    [0, 255]
    """
    img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

    return img

def get_parser():
    parser = argparse.ArgumentParser()
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
        default=1
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

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        logdir = opt.logdir

    print(config)
    model, global_step = load_model(config, ckpt, gpu, eval_mode)

    run_translation_depth_instance(model, opt)