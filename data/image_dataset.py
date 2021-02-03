import os
import cv2
import numpy as np
import h5py
import torch
import torch.utils.data
from other import utils
import dataset.config as cfg

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CenterDataset(object):

    def __init__(self, root, arg):

        self.dataset_dir = os.path.join(root, cfg.dataset_path)
        self.dataset_dir = os.path.join(self.dataset_dir, 'Train')

        self.hd5_name = (f"{cfg.hdf5_name}_d{arg.d}_path{arg.patch_size[0]}"
                         f"_stride{arg.stride_size[0]}_heatmap{arg.heatmap_size[0]}"
                         f"_version{arg.version}.hdf5")
        self.hd5_path = os.path.join(self.dataset_dir, self.hd5_name)

        self.transform = utils.transform(arg.version)

    def __getitem__(self, idx):

        dataset  = h5py.File(self.hd5_path, "r")

        img      = dataset["patch"][idx]
        heat_map = dataset["heatmap"][idx]
        # Number of pixels that have a nonzero value / zero values
        epsilon = dataset["epsilon"][0]
        # Normalize image
        img = self.transform(np.uint8(img))
        img = img.float()
        # Convert to Tensor
        heat_map = torch.as_tensor(heat_map, dtype=torch.float)
        epsilon  = torch.as_tensor(epsilon,  dtype=torch.float)

        return img, heat_map, epsilon


    def __len__(self):
        dataset  = h5py.File(self.hd5_path, "r")
        return len(dataset["patch"])


class TestDataset(object):

    def __init__(self, root, arg, file):


        self.imgs        = []
        self.coords_list = []

        self.version      = arg.version
        self.patch_size   = arg.patch_size
        self.stride_size  = arg.stride_size
        self.transform    = utils.transform(arg.version)

        self.file = file
        self.dataset_dir = os.path.join(root, cfg.dataset_path)
        self.dataset_dir = os.path.join(self.dataset_dir, 'Test')
        self.Image_path  = os.path.join(self.dataset_dir, cfg.image_path)
        self.Stain_path  = os.path.join(self.dataset_dir, cfg.stain_path)

        self.process()

    def __getitem__(self, idx):

        img    = self.imgs[idx]
        coords = self.coords_list[idx]
        # Normalize image
        img = self.transform(np.uint8(img))
        img = img.float()

        return img, coords


    def __len__(self):
        return len(self.imgs)

    def process(self):

        img_name   = self.file
        stain_name = os.path.splitext(self.file)[0] + '_stain.png'

        # Check if the img and the centers are for same file
        assert img_name[:-4]==stain_name[:-10], \
               f"The Image {img_name} and {stain_name} are not same!"

        img_path    = os.path.join(self.Image_path, img_name)
        stain_path  = os.path.join(self.Stain_path, stain_name)

        if not os.path.isfile(img_path) or not os.path.isfile(stain_path):
            raise ValueError(f"The {img_path} or {stain_path} is wrong!!")

        # Load Image
        img   = cv2.imread(img_path)
        stain = cv2.imread(stain_path)

        self.H, self.W, _ = img.shape

        # Process
        self.extract_patch_calculate_heatmap(img, stain)

    def extract_patch_calculate_heatmap(self, img, stain):

        cropped, coords = utils.patch_extraction(img,
                                                 stain,
                                                 self.patch_size,
                                                 self.stride_size,
                                                 self.version)

        self.imgs.extend(cropped)
        self.coords_list.extend(coords)

    def dimension(self):
        return self.H, self.W
