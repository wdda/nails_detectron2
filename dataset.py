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
        images = []
        annotations = []

        for index, img_name in enumerate(self.imgs):
            img = cv2.imread(self.all_img_paths[index], cv2.IMREAD_UNCHANGED)

            images.append({
                "file_name": img_name,
                "height": img.shape[0],
                "width": img.shape[1],
                "id": index
            })

            rows_by_img = self.data.loc[self.data['filename'] == img_name]

            for index_row, row in rows_by_img.iterrows():
                width = row['xmax'] - row['xmin']
                height = row['ymax'] - row['ymin']

                annotations.append({
                    "id": 1,
                    "box_mode": BoxMode.XYXY_ABS,
                    "bbox": [
                        row['xmin'],
                        row['ymin'],
                        width,
                        height

                    ],
                    "image_id": index,
                    "segmentation": [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])],
                    "ignore": 0,
                    "area": width * height,
                    "iscrowd": 0,
                    "category_id": 0
                })

        data = {
            "type": "instances",
            "images": images,
            "categories": [
                {
                    "supercategory": "none",
                    "name": "nail",
                    "id": 0
                },
            ],
            "annotations": annotations
        }

        return json.dumps(data)
