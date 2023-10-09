import os
from typing import Any
import cv2
import glob
import json
import random

import numpy as np
from torch.utils.data import Dataset
# from torchvision import transforms

class FixedTransDatasetBase(Dataset):
    def __init__(self, **kwargs):
        self.base_data_folder = 'data/3D_trans_diff_v1_256'
        label_json_file_path = 'data/3D_trans_diff_v1_256/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            self.labels = json.load(file)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])

        # normalize between [-1, 1]
        img = img / 127.5 - 1

        # the output of the dataset must be in this form (according to the original repo)
        output_dict = {'image': img}

        return output_dict
    
class FixedTransDatasetAerialViewTrain(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][:1300]:
            self.img_paths.append(
                os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name'])
            )

    def __len__(self):
        return len(self.img_paths)

class FixedTransDatasetGroundViewTrain(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][:1300]:
            self.img_paths.append(
                os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name'])
            )

    def __len__(self):
        return len(self.img_paths)

class FixedTransDatasetAerialViewVal(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][1300:]:
            self.img_paths.append(
                os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name'])
            )

    def __len__(self):
        return len(self.img_paths)

class FixedTransDatasetGroundViewVal(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][1300:]:
            self.img_paths.append(
                os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name'])
            )

    def __len__(self):
        return len(self.img_paths)
    
class FixedTransDatasetTrain(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][:1300]:
            img_info = {
                'ground': os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name']),
                'aerial': os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name'])
            }
            self.img_paths.append(
                img_info
            )

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_paths[idx]['ground'])
        aerial_img = cv2.imread(self.img_paths[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

        output_dict = {
            'ground': ground_img,
            'aerial': aerial_img
        }

        return output_dict

    def __len__(self):
        return len(self.img_paths)
    
class FixedTransDatasetVal(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for img_pair in self.labels['data'][1300:]:
            img_info = {
                'ground': os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name']),
                'aerial': os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name'])
            }
            self.img_paths.append(
                img_info
            )

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_paths[idx]['ground'])
        aerial_img = cv2.imread(self.img_paths[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

        output_dict = {
            'ground': ground_img,
            'aerial': aerial_img
        }

        return output_dict

    def __len__(self):
        return len(self.img_paths)
    
class FullTransDatasetTrain(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_info = []

        for img_pair in self.labels['data'][:10]:
            img_info = {
                'ground': os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name']),
                'aerial': os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name']),
                'location': img_pair['train_img_0_info']['location']
            }
            self.img_info.append(
                img_info
            )

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

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
    
class FullTransDatasetVal(FixedTransDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_info = []

        for img_pair in self.labels['data'][1300:]:
            img_info = {
                'ground': os.path.join(self.base_data_folder, img_pair['train_img_1_info']['img_name']),
                'aerial': os.path.join(self.base_data_folder, img_pair['train_img_0_info']['img_name']),
                'location': img_pair['train_img_0_info']['location']
            }
            self.img_info.append(
                img_info
            )

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

        # get the location
        x = self.img_info[idx]['location']['x']
        y = self.img_info[idx]['location']['y']
        z = self.img_info[idx]['location']['z']
        z_angle = self.img_info[idx]['location']['z_angle']

        output_dict = {
            'ground': ground_img,
            'aerial': aerial_img,
            'location': np.array([x, y, z, z_angle])
        }

        return output_dict

    def __len__(self):
        return len(self.img_info)
    
class TransDataset1D_Base(Dataset):
    def __init__(self, **kwargs):
        self.base_data_folder = 'data/1D_trans_diff_v1'
        label_json_file_path = 'data/1D_trans_diff_v1/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            data = json.load(file)

        self.labels = data['data']
        self.sensor_limits = data['sensor_limits']

class TransDataset1D_Train(TransDataset1D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_info = []

        normalize_dict = {}
        for key in self.sensor_limits:
            if key == 'num_sensors':
                continue
            if self.sensor_limits[key][0] == self.sensor_limits[key][1]:
                normalize_dict[key] = False
            else:
                normalize_dict[key] = True

        # normalize all sensor location data
        for idx, label in enumerate(self.labels):
            for label_idx, img_name in enumerate(label):
                for key in label[img_name]['location']:
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
        split_idx = len(self.img_info) // 5
        # self.img_info = self.img_info[split_idx:]

        # debugging
        self.img_info = self.img_info[0:100]

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

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
    
class TransDataset1D_Train_Autoencoder(TransDataset1D_Train):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        ground_img = cv2.imread(self.img_info[idx]['ground'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1

        output_dict = {
            'image': ground_img
        }

        return output_dict
    
class TransDataset1D_Val(TransDataset1D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_info = []

        normalize_dict = {}
        for key in self.sensor_limits:
            if key == 'num_sensors' or key == 'num_sensor_types':
                continue
            if self.sensor_limits[key][0] == self.sensor_limits[key][1]:
                normalize_dict[key] = False
            else:
                normalize_dict[key] = True

        # normalize all sensor location data
        for idx, label in enumerate(self.labels):
            for label_idx, img_name in enumerate(label):
                for key in label[img_name]['location']:
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
        split_idx = len(self.img_info) // 5
        # self.img_info = self.img_info[:split_idx]

        # debugging
        self.img_info = self.img_info[500:600]

    def __getitem__(self, idx):

        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1
        aerial_img = aerial_img / 127.5 - 1

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
    
class TransDataset1D_Val_Autoencoder(TransDataset1D_Val):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        ground_img = cv2.imread(self.img_info[idx]['ground'])

        # normalize between [-1, 1]
        ground_img = ground_img / 127.5 - 1

        output_dict = {
            'image': ground_img
        }

        return output_dict
    
class TransDepthDataset3D_Base(Dataset):
    def __init__(self, data_folder_path=None, sensor_types_to_return = None, **kwargs):
        if data_folder_path is None:
            self.base_data_folder = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/1D_trans_multi_sensor_test'
        else:
            self.base_data_folder = data_folder_path
        label_json_file_path = self.base_data_folder + '/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            data = json.load(file)

        self.labels = data['data']
        self.sensor_limits = data['sensor_limits']

        num_sensor_types = self.sensor_limits['num_sensor_types']

        self.labels = normalize_labels(self.sensor_limits, self.labels)

        data_pairs = []

        # create translation dataset
        for label in self.labels:
            label = list(label.values())
            # get the ground truth image set
            ground_truth = label[-num_sensor_types:]
            translation_labels = label[:-num_sensor_types]
            idx = 0

            # get pairs of data from each translation image in the group
            while idx+2 < len(translation_labels):
                trans_group_info = dict()
                trans_group_info['location'] = translation_labels[idx]['location']
                for i in range(num_sensor_types):
                    img_info = translation_labels[idx]
                    sensor_name = img_info['img_name']
                    sensor_type = img_info['img_name'].split('_')[0]+'_aerial'
                    trans_group_info[sensor_type] = sensor_name
                    idx += 1

                # add ground truth to image group
                for img in ground_truth:
                    sensor_name = img['img_name']
                    sensor_type = img['img_name'].split('_')[0]+'_ground'
                    trans_group_info[sensor_type] = sensor_name

                data_pairs.append(trans_group_info)

        # shuffle and get train and validation sets
        random.seed(42)

        random.shuffle(data_pairs)
        
        split_idx = len(data_pairs) // 5
        self.train_pairs = data_pairs[split_idx:]
        self.val_pairs = data_pairs[:split_idx]

        if sensor_types_to_return:
            self.sensor_types_to_return = sensor_types_to_return
        else:
            self.sensor_types_to_return = [
                'rgb',
                'semantic',
                'depth'
            ]

    def __getitem__(self, idx):
        
        data = self.data_pairs[idx]

        output = dict()

        # get the ground sensor imgs
        for sensor_type in self.sensor_types_to_return:
            sensor_key = f'{sensor_type}_ground'
            output[sensor_key] = cv2.imread(self._set_path(data[sensor_key]))
        
        # get the translation sensor imgs
        for sensor_type in self.sensor_types_to_return:
            sensor_key = f'{sensor_type}_aerial'
            output[sensor_key] = cv2.imread(self._set_path(data[sensor_key]))

        # normalize labels between [-1, 1]
        for sensor_name in output:
            sensor_type = sensor_name.split('_')[0]
            output[sensor_name] = normalize_sensor_data(sensor_type, output[sensor_name])

        # get the location
        x = data['location']['x']
        y = data['location']['y']
        z = data['location']['z']
        z_angle = data['location']['z_angle']

        output['location'] = np.array([[x, y, z, z_angle]], dtype='float32')

        return output
    
    def _set_path(self, img_name):
        return os.path.join(self.base_data_folder, img_name)
    
class TransDepthDataset3D_Train(TransDepthDataset3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_pairs = self.train_pairs

    def __len__(self):
        return len(self.data_pairs)
    
class TransDepthDataset3D_Val(TransDepthDataset3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_pairs = self.val_pairs

    def __len__(self):
        return len(self.data_pairs)
    
def normalize_sensor_data(sensor_type, sensor_data, num_semantic_classes=None):
    if sensor_type == 'rgb':
        sensor_data = sensor_data / 127.5 - 1
    elif sensor_type == 'depth':
        sensor_data = sensor_data[:,:,2] + 256 * sensor_data[:,:,1] + 256 * 256 * sensor_data[:,:,0]
        sensor_data = sensor_data / (256 * 256 * 256 - 1)
    elif sensor_type == 'semantic_segmentation':
        if num_semantic_classes is None:
            num_semantic_classes = 24
        sensor_data = sensor_data[:,:,2] / num_semantic_classes * 2 - 1
    elif sensor_type == 'instance_segmentation':
        # TODO: Add instance segmentation normalization
        pass
    else:
        raise Exception("Invalid sensor type.")
    
    return sensor_data

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

def normalize_labels_sensor_params(sensor_params, labels):
    """Normalizes all of the LOCATION VALUES ONLY for every label in a translation
    dataset.
    """
    normalize_dict = {}
    for key in sensor_params:
        if key == 'num_sensors' or key == 'num_sensor_types':
            continue
        if sensor_params[key][0] == sensor_params[key][1]:
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
                if normalize_dict[key]:
                    min_lim = sensor_params[key][0]
                    max_lim = sensor_params[key][1]

                    # normalize between 0 and 1
                    label[img_name]['location'][key] = (label[img_name]['location'][key] - min_lim) / (max_lim-min_lim)

                    # normalize between -1 and 1
                    label[img_name]['location'][key] = label[img_name]['location'][key] * 2 - 1
                else:
                    label[img_name]['location'][key] = 0

        labels[idx] = label
    
    return labels

class InstanceDepthDatasetBase(Dataset):
    """This dataset takes in a path to the data folder, reads the labels file,
    and produces a dictionary of normalized numpy arrays in the form:
    
    output_dict = {
        'ground' (np.array): ground_img,
        'trans' (np.array): trans_img,
        'location' (np.array): relative location of the ground img from the
            trans img
    }
    """

    def __init__(self, data_folder_path=None, **kwargs):
        if data_folder_path is None:
            data_folder_path = 'data/1D_trans_multi_sensor_v1_val'
        self.base_data_folder = data_folder_path
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

        # shuffle and get train and validation sets
        random.seed(42)

        random.shuffle(data_pairs)
        
        split_idx = len(data_pairs) // 5
        self.train_pairs = data_pairs[split_idx:]
        self.val_pairs = data_pairs[:split_idx]

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

class InstanceDepthDatasetTrain(InstanceDepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.train_pairs

class InstanceDepthDatasetVal(InstanceDepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.val_pairs

def preprocess_data(labels_json_path: str,
                    type: str):
    # read label data
        with open(labels_json_path, 'r') as file:
            data = json.load(file)

        labels = data['data']
        sensor_params = data['sensor_params']
        sensor_types = sensor_params['sensor_types']
        num_aux_sensors = sensor_params['num_aux_sensors']

        # the below normalizes the sensor location information, NOT the images
        labels = normalize_labels_sensor_params(sensor_params['relative_spawn_limits'], labels)
        
        # process the dataset in one to many format (one 'from' image and many 'to' images)
        if type == 'one_to_many':
            data_pairs = []
            for label in labels:
                # the first sensor location should be the one 'from' image
                label = list(label.values())

                # get all sensor data from the 'from' location
                from_img_paths = dict()
                sensor_idx = 0
                for _ in range(len(sensor_types)):
                    sensor_info = label[sensor_idx]
                    from_img_paths[sensor_info['sensor_type']] = (sensor_info['img_name'])
                    sensor_idx +=1

                # get all sensor data from the 'to' locations
                for i in range(num_aux_sensors):
                    to_img_paths = dict()
                    for _ in range(len(sensor_types)):
                        sensor_info = label[sensor_idx]
                        to_img_paths[sensor_info['sensor_type']] = (sensor_info['img_name'])
                        sensor_idx +=1

                    data_pairs.append({'from': from_img_paths,
                                       'to': to_img_paths,
                                       'translation_label': sensor_info['location']})
        else:
            raise NotImplementedError("Need to add many to one functionality.")
        
        return data_pairs


class DepthDatasetBase(Dataset):
    """
    For datasets with depth information only.

    This dataset takes in a path to the data folder, reads the labels file,
    and produces a dictionary of normalized numpy arrays in the form:
    
    output_dict = {
        'ground' (np.array): ground_img,
        'trans' (np.array): trans_img,
        'location' (np.array): relative location of the ground img from the
            trans img
    }
    """

    def __init__(self, data_folder_path=None, **kwargs):
        if data_folder_path is None:
            data_folder_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3d_trans_multi_sensor_v3_large'
        self.base_data_folder = data_folder_path
        label_json_file_path = self.base_data_folder+'/labels.json'

        data_pairs = preprocess_data(label_json_file_path,
                                     type='one_to_many')

        # shuffle and get train and validation sets
        random.seed(42)

        random.shuffle(data_pairs)
        
        split_idx = len(data_pairs) // 5
        self.train_pairs = data_pairs[split_idx:]
        self.val_pairs = data_pairs[:split_idx]

        self.data_pairs = data_pairs
    
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]

        # we only want the depth information
        from_depth = os.path.join(self.base_data_folder,
                                  data_pair['from']['depth'])
        to_depth = os.path.join(self.base_data_folder,
                                data_pair['to']['depth'])
        
        from_depth = self._preprocess_depth(from_depth)
        to_depth = self._preprocess_depth(to_depth)

        # get the location
        x = data_pair['translation_label']['x']
        y = data_pair['translation_label']['y']
        z = data_pair['translation_label']['z']
        yaw = data_pair['translation_label']['yaw']

        # uncomment below to see if model is learning without translation label

        # print(f'original y: {y}')
        # # randomize y
        # y = random.random() * 2 - 1
        # print(f'new_y: {y}')

        output_dict = {
            'to': to_depth,
            'from': from_depth,
            'translation_label': np.array([[x, y, z, yaw]], dtype='float32')
        }

        return output_dict

    def __len__(self):
        return len(self.data_pairs)
    
    def _preprocess_depth(self, depth_img_path):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = cv2.imread(depth_img_path)
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        
        # the distribution of depth values was HEAVILY skewed towards the lower end
        # therfore we will try to improve the distribution
        
        # need to test with clip_coefficient = 2
        clip_coefficient = 4

        depth_img = np.clip(depth_img, 0, 1/clip_coefficient)

        depth_img = depth_img * clip_coefficient

        depth_img = depth_img * 2 - 1

        return np.expand_dims(depth_img, axis=-1)
        
class DepthDatasetTrain(DepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.train_pairs

class DepthDatasetVal(DepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.val_pairs

class RGBDepthDatasetBase(Dataset):
    """
    For datasets with depth information only.

    This dataset takes in a path to the data folder, reads the labels file,
    and produces a dictionary of normalized numpy arrays in the form:
    
    output_dict = {
        'ground' (np.array): ground_img,
        'trans' (np.array): trans_img,
        'location' (np.array): relative location of the ground img from the
            trans img
    }
    """

    def __init__(self, data_folder_path=None, **kwargs):
        if data_folder_path is None:
            data_folder_path = 'data/vehicle_control_dataset_test'
        self.base_data_folder = data_folder_path
        label_json_file_path = self.base_data_folder+'/labels.json'

        data_pairs = preprocess_data(label_json_file_path,
                                     type='one_to_many')

        # shuffle and get train and validation sets
        random.seed(42)

        random.shuffle(data_pairs)
        
        split_idx = len(data_pairs) // 5
        self.train_pairs = data_pairs[split_idx:]
        self.val_pairs = data_pairs[:split_idx]

        self.data_pairs = data_pairs
    
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]

        # depth information
        from_depth = os.path.join(self.base_data_folder,
                                  data_pair['from']['depth'])
        to_depth = os.path.join(self.base_data_folder,
                                data_pair['to']['depth'])
        
        from_depth = self._preprocess_depth(from_depth)
        to_depth = self._preprocess_depth(to_depth)

        # rgb information
        from_rgb = cv2.imread(os.path.join(self.base_data_folder,
                                           data_pair['from']['rgb']))
        to_rgb = cv2.imread(os.path.join(self.base_data_folder,
                                         data_pair['to']['rgb']))
        
        # normalize the rgb information
        from_rgb = from_rgb / 127.5 - 1
        to_rgb = to_rgb / 127.5 - 1

        # get the location
        x = data_pair['translation_label']['x']
        y = data_pair['translation_label']['y']
        z = data_pair['translation_label']['z']
        yaw = data_pair['translation_label']['yaw']

        # concatenate the depth and rgb information to form a 4-channel image
        # with the following specs: [B, G, R, Depth] * it is in BGR format because
        # that is what cv2 reads images as
        to_rgbd = np.concatenate((to_rgb, to_depth), axis=2)
        from_rgbd = np.concatenate((from_rgb, from_depth), axis=2)

        output_dict = {
            'to': to_rgbd,
            'from': from_rgbd,
            'translation_label': np.array([[x, y, z, yaw]], dtype='float32')
        }

        return output_dict

    def __len__(self):
        return len(self.data_pairs)
    
    def _preprocess_depth(self, depth_img_path):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = cv2.imread(depth_img_path)
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        
        # the distribution of depth values was HEAVILY skewed towards the lower end
        # therfore we will try to improve the distribution
        
        # need to test with clip_coefficient = 2
        clip_coefficient = 4

        depth_img = np.clip(depth_img, 0, 1/clip_coefficient)

        depth_img = depth_img * clip_coefficient

        depth_img = depth_img * 2 - 1

        return np.expand_dims(depth_img, axis=-1)
        
class RGBDepthDatasetTrain(RGBDepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.train_pairs

class RGBDepthDatasetVal(RGBDepthDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_pairs = self.val_pairs

if __name__=='__main__':
    dataset = RGBDepthDatasetBase()
    print(len(dataset))
    import matplotlib.pyplot as plt
    while 1:
        idx = random.randint(0, len(dataset))

        yes = dataset[idx]

        a = yes['from']
        a = a.flatten()

        plt.hist(a, bins=np.linspace(-1, 1, 100))
        plt.show()