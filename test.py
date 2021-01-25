import os
import torch
import numpy as np
from PIL import Image
from numpy import asarray
import other.utils as utils
from model.SC_CNN import SC_CNN
from matplotlib import pyplot as plt
from model.SC_CNN_v2 import SC_CNN_v2
from other.functions import test_model
from postprocess.postprocess import postprocess



def test(arg):

    root = os.path.dirname(os.path.realpath(__file__))

    if arg.version==0 or arg.version==1:
        # get the original model
        model = SC_CNN(arg.M, arg.patch_size, arg.heatmap_size, arg.version)

    if arg.version==2:
        # get the new model
        model = SC_CNN_v2(arg.M, arg.heatmap_size)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # move model to the right device
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

     # load model
    _, _, model, _, _ = utils.load_model(arg.load_name,
                                         model,
                                         None,
                                         None)
    path = '/Datasets/121095_31096.png'
    # path = '/Datasets/CRCHistoPhenotypes_2016_04_28/Tissue_Images/img1.bmp'
    image_path = os.path.join(root + path)
    data = asarray(Image.open(image_path))

    cell = test_model(image_path, data, model, arg)
    cell = postprocess(cell, dist=8, thresh=0.3)

    print(f"{len(cell)} cells are detected!")
    plt.imshow(data, interpolation='nearest')
    plt.scatter(cell[:,0], cell[:,1], c='r', s=2)
    plt.show()
