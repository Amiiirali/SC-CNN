import os
import cv2
import numpy as np

import torch
import torch.utils.data

from other import utils
import dataset.config as cfg

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CenterDataset(object):

    def __init__(self, root, arg):


        self.imgs      = []
        self.heat_maps = []

        self.d            = arg.d
        self.version      = arg.version
        self.patch_size   = arg.patch_size
        self.stride_size  = arg.stride_size
        self.heatmap_size = arg.heatmap_size

        self.epsilons  = 0
        self.transform = utils.transform(arg.version)

        self.dataset_dir = os.path.join(root, cfg.dataset_path)
        self.dataset_dir = os.path.join(self.dataset_dir, 'Train')
        self.Image_path  = os.path.join(self.dataset_dir, cfg.image_path)
        self.Center_path = os.path.join(self.dataset_dir, cfg.center_path)
        self.Stain_path  = os.path.join(self.dataset_dir, cfg.stain_path)

        self.process()

    def __getitem__(self, idx):

        img      = self.imgs[idx]
        heat_map = self.heat_maps[idx]
        # Number of pixels that have a nonzero value / zero values
        epsilon = self.epsilons / (len(self.imgs)*self.heatmap_size[0]*self.heatmap_size[1] - self.epsilons)
        # Normalize image
        img = self.transform(np.uint8(img))
        img = img.float()
        # Convert to Tensor
        heat_map = torch.as_tensor(heat_map, dtype=torch.float)
        epsilon  = torch.as_tensor(epsilon,  dtype=torch.float)

        return img, heat_map, epsilon


    def __len__(self):
        return len(self.imgs)

    def process(self):

        # MacOS thing
        utils.delete_file(self.Image_path, '.DS_Store')
        utils.delete_file(self.Center_path, '.DS_Store')
        utils.delete_file(self.Stain_path, '.DS_Store')

        for img_name, stain_name, center_file in zip(list(sorted(os.listdir(self.Image_path), key=lambda x:x[:-4])),
                                                     list(sorted(os.listdir(self.Stain_path), key=lambda x:x[:-10])),
                                                     list(sorted(os.listdir(self.Center_path), key=lambda x:x[:-4]))):

            # Check if the img and the centers are for same file
            assert img_name[:-4]==center_file[:4] \
                or img_name[:-4]==stain_name[:-10], \
                   f"The Image {img_name}, Center {center_file}, and {stain_name} are not same!"

            img_path    = os.path.join(self.Image_path, img_name)
            stain_path  = os.path.join(self.Stain_path, stain_name)
            center_path = os.path.join(self.Center_path, center_file)
            # Load Image
            img   = cv2.imread(img_path)
            stain = cv2.imread(stain_path)
            # Reading Centers
            center_txt = open(center_path, "r")
            center     = utils.read_center_txt(center_txt)
            # Process
            self.extract_patch_calculate_heatmap(img, stain, center)

    def extract_patch_calculate_heatmap(self, img, stain, center):

        cropped, coords = utils.patch_extraction(img,
                                                 stain,
                                                 self.patch_size,
                                                 self.stride_size,
                                                 self.version)
        # Extracting patches' centers
        patch_centers = utils.center_extraction(center,
                                                coords)
        # Finding epsilon and heatmaps
        h_map, epsilon = utils.heat_map(patch_centers,
                                        coords,
                                        self.d,
                                        self.patch_size,
                                        self.heatmap_size)
        self.imgs.extend(cropped)
        self.heat_maps.extend(h_map)
        self.epsilons += epsilon
