import pandas as pd
import glob
import os
import detectron2
from detectron2.structures import BoxMode
import cv2
import json


class DataSet(object):
    def __init__(self, folder):
        self.folder = folder
        self.root_folder = 'images'
        self.data = pd.read_csv(os.path.join(self.root_folder, folder + '_labels.csv'))
        self.all_img_paths = glob.glob(os.path.join(self.root_folder, folder) + '/*.jpg')
        self.imgs = []

        for img_path in self.all_img_paths:
            img_name = img_path.split('/')[-1]
            self.imgs.append(img_name)

    def set(self):
        dataset_dicts = []

        for index, img_name in enumerate(self.imgs):
            record = {}

            img = cv2.imread(self.all_img_paths[index], cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2]

            record["file_name"] = self.all_img_paths[index]
            record["height"] = height
            record["width"] = width

            rows_by_img = self.data.loc[self.data['filename'] == img_name]

            objs = []
            for index_row, row in rows_by_img.iterrows():
                obj = {
                    "bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)


        return dataset_dicts
