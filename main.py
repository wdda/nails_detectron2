import dataset
from detectron2.data import DatasetCatalog


def get_data():
    return dataset.DataSet('train').set()


datasetForDetector = DatasetCatalog.register("nails_dataset", get_data)
