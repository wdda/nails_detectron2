import dataset
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
import os

import cv2

# Create config
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"


def get_data(folder):
    return dataset.DataSet(folder).set()


# datasetForDetector = DatasetCatalog.register("nails_dataset", get_data)
# MetadataCatalog.get("nails_dataset").thing_classes = ["nail"]


def train():
    for d in ["train", "test"]:
        DatasetCatalog.register("nail/" + d, lambda d=d: get_data(d))
        MetadataCatalog.get("nail/" + d).set(thing_classes=["nail"])

    balloon_metadata = MetadataCatalog.get("nail/train")
    cfg.DATASETS.TRAIN = ("nail/train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def test_img():
    im = cv2.imread("test.jpg")

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)
    image = im[:, :, ::-1]

    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('test', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


train()
