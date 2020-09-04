import os
import multiprocessing
import dataset.config as cfg
from dataset.functions import load_dataset, download_dataset


def dataset():

    '''
    This function checks if the dataset is not downloaded previously, it
    downloads that.

    Input:
          1- root:
                The absolute path to the main folder.
    '''
    # Path to this file
    root        = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_dir = os.path.join(root, cfg.dataset_path)

    # If the data is not downloaded yet
    if not os.path.isdir(dataset_dir):

        print(20 * '*')
        print(f"Dataset is Downloading ... \n")

        os.mkdir(dataset_dir)

        download_dataset(cfg.url, dataset_dir)
        load_dataset(multiprocessing.cpu_count(),
                     dataset_dir,
                     cfg.detection_path,
                     cfg.image_path,
                     cfg.center_path,
                     cfg.stain_path)

    else:

        print(20 * '*')
        print(f"Dataset Exists \n")
