import json
import random

labels_file_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256/labels.json'

with open(labels_file_path, 'r') as file:
    data = json.load(file)

what = 'yes'