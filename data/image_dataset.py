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

        self.epsilons  = 0
        self.out_size  = arg.heatmap_size
        self.transform = utils.transform(arg.version)

        self.dataset_dir = os.path.join(root, cfg.dataset_path )

        self.Image_path  = os.path.join(self.dataset_dir, cfg.image_path )
        self.Center_path = os.path.join(self.dataset_dir, cfg.center_path )
        self.Stain_path  = os.path.join(self.dataset_dir, cfg.stain_path )

        # MacOS thing :)
        utils.delete_file(self.Image_path, '.DS_Store')
        utils.delete_file(self.Center_path, '.DS_Store')
        utils.delete_file(self.Stain_path, '.DS_Store')

        # load all image files, sorting them to
        total_num = len(list(os.listdir(self.Image_path)))

        for idx, (img_name, stain_name, center_file) in enumerate(zip(list(sorted(os.listdir(self.Image_path))),
                                                                      list(sorted(os.listdir(self.Stain_path))),
                                                                      list(sorted(os.listdir(self.Center_path)))
                                                                      )):

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

            cropped, coords = utils.patch_extraction(img,
                                                     stain,
                                                     arg.patch_size,
                                                     arg.stride_size,
                                                     arg.version)

            # Reading Centers
            center_txt = open(center_path, "r")
            center     = utils.read_center_txt(center_txt)

            # Extracting patches' centers
            patch_centers = utils.center_extraction(center,
                                                    coords,
                                                    arg.patch_size,
                                                    arg.heatmap_size)

            # Finding epsilon and heatmaps
            h_map, epsilon = utils.heat_map(patch_centers,
                                            coords,
                                            arg.d,
                                            arg.patch_size,
                                            arg.heatmap_size)

            self.imgs.extend(cropped)
            self.heat_maps.extend(h_map)

            self.epsilons += epsilon

            # print(idx+1, 'from', total_num, 'Images are Loaded!' ,
            #       sep=' ', end='\r', flush=True)


    def __getitem__(self, idx):

        img = self.imgs[idx]
        heat_map = self.heat_maps[idx]

        # Number of pixels that have a nonzero value / zero values
        epsilon = self.epsilons / (len(self.imgs)*self.out_size[0]*self.out_size[1] - self.epsilons)

        # Normalize image
        img = self.transform(np.uint8(img))
        img = img.float()

        heat_map = torch.as_tensor(heat_map, dtype=torch.float)
        epsilon  = torch.as_tensor(epsilon,  dtype=torch.float)

        return img, heat_map, epsilon


    def __len__(self):
        return len(self.imgs)
