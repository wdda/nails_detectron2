import dataset
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
import os
import random
import cv2

# Create config
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 15000  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.DATASETS.TRAIN = ("nail/train",)
cfg.DATASETS.TEST = ("nail/test",)

nail_metadata_test = MetadataCatalog.get(cfg.DATASETS.TEST[0])
nail_metadata_test.set(thing_classes=["nail"])


def get_data(folder):
    return dataset.DataSet(folder).set()


for d in ["train", "test"]:
    DatasetCatalog.register("nail/" + d, lambda d=d: get_data(d))
    MetadataCatalog.get("nail/" + d).set(thing_classes=["nail"])


def train():
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def test_img():
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    dataset_dicts = get_data("test")

    for d in random.sample(dataset_dicts, 3):
        test_one_img(d['file_name'])


def test_one_img(img_path):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    im = cv2.imread(img_path)

    W = 300.
    height, width, depth = im.shape
    imgScale = W / width
    newX, newY = im.shape[1] * imgScale, im.shape[0] * imgScale
    im = cv2.resize(im, (int(newX), int(newY)))

    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)
    image = im[:, :, ::-1]

    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow(img_path, v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def test_video():
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # Open video file
    video = cv2.VideoCapture('videoplayback.mp4')
    predictor = DefaultPredictor(cfg)

    while (video.isOpened()):
        ret, image = video.read()
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], metadata=nail_metadata_test, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', v.get_image()[:, :, ::-1])

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()


# test_one_img('test3.jpg')
train()
