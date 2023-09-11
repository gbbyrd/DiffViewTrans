import os
import cv2
import glob
import json

import numpy as np
from torch.utils.data import Dataset
# from torchvision import transforms

class FixedTransDatasetBase(Dataset):
    def __init__(self, **kwargs):
        # self.base_data_folder = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256'
        # label_json_file_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256/labels.json'
        self.base_data_folder = 'data/3D_trans_diff_v1_256'
        label_json_file_path = 'data/3D_trans_diff_v1_256/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            self.labels = json.load(file)

        # self.transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])

        # normalize between [-1, 1]
        img = img / 127.5 - 1

        # img_tens = self.transform(img)

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
        # self.base_data_folder = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256'
        # label_json_file_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256/labels.json'
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

if __name__=='__main__':
    dataset_fixed = FixedTransDatasetGroundViewTrain()
    dataset_1d = TransDataset1D_Train_Autoencoder()

    what = 'yes'