import os
import logging
import multiprocessing
import dataset.config as cfg
from dataset.functions import load_dataset, download_dataset, load_test_dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def dataset(arg):

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
    logger      = logging.getLogger('Data')

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    if arg.mode == 'train':

        dataset_train_dir = os.path.join(dataset_dir, 'Train')
        # If the data is not downloaded yet
        if not os.path.isdir(dataset_train_dir):

            logger.info(f"Dataset is Downloading ...")

            os.mkdir(dataset_train_dir)

            download_dataset(cfg.url, dataset_train_dir)

            num_cpu = min(arg.num_workers, multiprocessing.cpu_count())
            logger.info(f"Using {num_cpu} CPUs ...")
            load_dataset(num_cpu,
                         dataset_train_dir,
                         cfg.detection_path,
                         cfg.image_path,
                         cfg.center_path,
                         cfg.stain_path)
            logger.info(f"Finished Processing Dataset.")

        else:
            logger.info(f"Dataset Exists.")

    if arg.mode == 'test':

        logger.info(f"PreProcessing Test Dataset ...")

        if arg.test_dir == None or not os.path.isdir(arg.test_dir):
            raise ValueError("The Test Directory is Wrong!")

        dataset_test_dir = os.path.join(dataset_dir, 'Test')

        if not os.path.isdir(dataset_test_dir):
            os.mkdir(dataset_test_dir)

        num_cpu = min(arg.num_workers, multiprocessing.cpu_count())
        logger.info(f"Using {num_cpu} CPUs ...")
        load_test_dataset(num_cpu,
                          dataset_test_dir,
                          arg.test_dir,
                          cfg.image_path,
                          cfg.center_path,
                          cfg.stain_path)
