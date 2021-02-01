import os
import torch
import logging
import numpy as np
from PIL import Image
import other.utils as utils
from model.SC_CNN import SC_CNN
from model.SC_CNN_v2 import SC_CNN_v2
from postprocess.postprocess import postprocess
from data.image_dataset import TestDataset
from model.model import Model
import dataset.config as cfg


class Test(object):

    def __init__(self, arg):

        # train on the GPU or on the CPU, if a GPU is not available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.logger = logging.getLogger('Testing')

        self.arg = arg
        self.root= os.path.dirname(os.path.realpath(__file__))
        self.d            = arg.d
        self.version      = arg.version
        self.num_workers  = arg.num_workers
        self.batch_size   = arg.batch_size
        self.heatmap_size = arg.heatmap_size
        self.patch_size   = arg.patch_size
        self.test_dir     = arg.test_dir

    def data_loader(self, file):

        self.logger.info(f'Loading Image <{file}> ...')
        dataset = TestDataset(self.root, self.arg, file)

        return dataset

    def detect_cell(self, file):

        dataset = self.data_loader(file)

        H, W  = dataset.dimension()
        cell  = np.zeros((H, W))
        count = np.zeros((H, W))

        [H_prime, W_prime] = self.heatmap_size

        for data in dataset:

            img, coords = data

            img = img.to(self.device)
            img = img.unsqueeze(0)

            point, h = self.SC_CNN.model(img)

            heat_map = utils.heat_map_tensor(point.view(-1, 2),
                                             h.view(-1, 1),
                                             self.device,
                                             self.d,
                                             self.heatmap_size)
            heatmap  = heat_map.cpu().detach().numpy().reshape((H_prime, W_prime))

            start_H, end_H, start_W, end_W = utils.find_out_center_coords(coords,
                                                                          self.patch_size,
                                                                          self.heatmap_size)

            cell[start_H:end_H, start_W:end_W] += heatmap

            idx = np.argwhere(heatmap != 0)
            count[idx[:,0]+start_H, idx[:,1]+start_W] += 1

        count[count==0] = 1
        cell = np.divide(cell, count)

        return cell

    def load_model(self):

        self.SC_CNN = Model(self.arg, self.logger)
        self.SC_CNN.initialize()
        self.SC_CNN.load_()
        # Test mode
        self.SC_CNN.model.train(False)

    def run(self):

        self.load_model()

        utils.delete_file(self.test_dir, '.DS_Store')
        for file in os.listdir(self.test_dir):
            cell = self.detect_cell(file)
            cell = postprocess(cell, dist=8, thresh=0.3)
            self.logger.info(f'{len(cell)} cells are detected for <{file}>.')

            center_name = os.path.splitext(file)[0] + '.txt'
            path = f"{self.root}/{cfg.dataset_path}/Test/{cfg.center_path}/{center_name}"
            utils.save_cells(cell, path)
