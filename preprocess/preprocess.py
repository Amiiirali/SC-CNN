import os
import cv2
import numpy as np
import h5py
import logging
from other import utils
import dataset.config as cfg

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MakeH5File(object):

    def __init__(self, root, arg):


        self.imgs      = []
        self.heat_maps = []

        self.d            = arg.d
        self.version      = arg.version
        self.patch_size   = arg.patch_size
        self.stride_size  = arg.stride_size
        self.heatmap_size = arg.heatmap_size

        self.dataset_dir = os.path.join(root, cfg.dataset_path)
        self.dataset_dir = os.path.join(self.dataset_dir, 'Train')
        self.Image_path  = os.path.join(self.dataset_dir, cfg.image_path)
        self.Center_path = os.path.join(self.dataset_dir, cfg.center_path)
        self.Stain_path  = os.path.join(self.dataset_dir, cfg.stain_path)

        self.epsilons = 0
        self.hd5_name = (f"{cfg.hdf5_name}_d{self.d}_path{self.patch_size[0]}"
                         f"_stride{self.stride_size[0]}_heatmap{self.heatmap_size[0]}"
                         f"_version{self.version}.hdf5")
        self.hd5_path = os.path.join(self.dataset_dir, self.hd5_name)

        self.logger = logging.getLogger('H5')

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

    def run(self):

        if os.path.isfile(self.hd5_path):
            self.logger.info(f'HD5 Format of Dataset Exists!')

        else:
            self.logger.info(f'Creating HD5 Format of Dataset ...')
            self.process()
            # open a hdf5 file and create earrays
            f = h5py.File(self.hd5_path, mode='w')

            num_data  = len(self.imgs)
            dimension = self.imgs[0].shape[2]

            patch_shape = (num_data, self.patch_size[0], self.patch_size[1], dimension)
            heatmap_shape = (num_data, self.heatmap_size[0], self.heatmap_size[1])

            # PIL.Image: the pixels range is 0-255,dtype is uint.
            # matplotlib: the pixels range is 0-1,dtype is float.
            f.create_dataset("patch", patch_shape, np.uint8)
            f.create_dataset("heatmap", heatmap_shape, np.uint8)
            f.create_dataset("epsilon", (1,), np.float)

            for i in range(num_data):

                f["patch"][i, ...]   = self.imgs[i]
                f["heatmap"][i, ...] = self.heat_maps[i]

            epsilon = self.epsilons / (len(self.imgs)*self.heatmap_size[0]*self.heatmap_size[1] - self.epsilons)
            f["epsilon"][...] = epsilon

            f.close()
