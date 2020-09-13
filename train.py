import torch
import torch.nn as nn
import torchvision

from torchsummary import summary

import os
import other.utils as utils
from loss.loss import BCE_Loss
from model.SC_CNN import SC_CNN
from model.HeatMap import HeatMap
from model.SC_CNN_v2 import SC_CNN_v2
from other.functions import train_model
from data.image_dataset import CenterDataset


def train(arg):

    # train on the GPU or on the CPU, if a GPU is not available
    device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    root        = os.path.dirname(os.path.realpath(__file__))
    arg.log_dir = os.path.join(root, arg.log_dir)

    if not os.path.isdir(arg.log_dir):
        os.mkdir(arg.log_dir)

    # use our dataset and defined transformations
    print(20 * '*')
    print('Loading the Dataset:')

    dataset = CenterDataset(root,
                            arg)

    # print('\n' * 1)
    # print(20 * '*')
    # print('Checking the data:')
    # utils.printInformation(dataset, 0)

    # split the dataset in train and valid set
    torch.manual_seed(1)
    indices       = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:
                                            -int(len(indices)*arg.valid_coeff)])
    dataset_valid = torch.utils.data.Subset(dataset, indices[
                                            -int(len(indices)*arg.valid_coeff):])

    print()
    print()
    print(20 * '*')
    print("Total number of data is           :", len(dataset))
    print("Total number of training data is  :", len(dataset_train))
    print("Total number of validation data is:", len(dataset_valid))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=arg.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    valid_data_loader = torch.utils.data.DataLoader(dataset_valid,
                                                    batch_size=arg.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    dataLoaders   = {
                     'Train'     : train_data_loader,
                     'Validation': valid_data_loader
                     }

    dataset_sizes = {
                     'Train'     : len(dataset_train),
                     'Validation': len(dataset_valid)
                     }

    if arg.version==0 or arg.version==1:
        # get the original model
        model = SC_CNN(arg.M, arg.patch_size, arg.heatmap_size, arg.version)

    if arg.version==2:
        # get the new model
        model = SC_CNN_v2(arg.M, arg.heatmap_size)

    # Generating Heatmap withour my own grad
    Map = HeatMap.apply

    # move model to the right device
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # move model to the right device
    model.to(device)

    print()
    print(20 * '*')
    print('Summary of the Model:')
    summary(model, (dataset[0])[0].shape)


    # construct an optimizer
    params    = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params,
                                lr=arg.lr,
                                momentum=arg.momentum,
                                weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60,100],
                                                        gamma=0.1)


    ###########################################
    # This is Pytorch loss function
    # BCE
    criterion = nn.BCELoss(reduction='none')

    ###########################################
    # This is my own written loss function
    # SCCNN
    # criterion = BCE_Loss.apply

    # Validation criterion
    validation_criterion = nn.MSELoss(reduction='none')

    print()
    print(20 * '*')
    print('Training Starts: \n')
    # Training
    train_model(model,
                dataLoaders,
                dataset_sizes,
                Map,
                criterion,
                validation_criterion,
                optimizer,
                lr_scheduler,
                arg)
