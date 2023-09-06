import os
import cv2
import glob
import json

from torch.utils.data import Dataset

class FixedTransDatasetBase(Dataset):
    def __init__(self, **kwargs):
        self.base_data_folder = '/home/nianyli/Desktop/code/thesis/TransDiffusion/data/3D_trans_diff_v1_256'
        label_json_file_path = '/home/nianyli/Desktop/code/thesis/TransDiffusion/data/3D_trans_diff_v1_256/labels.json'
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

if __name__=='__main__':
    dataset =  FixedTransDatasetAerialViewTrain()

    what = 'yes'