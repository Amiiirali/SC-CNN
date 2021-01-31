import torch
import torch.nn as nn
import torchvision

from torchsummary import summary
import os
import logging
import other.utils as utils
from data.image_dataset import CenterDataset
from model.model import Model
import torch
import time
import other.utils as utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Train(object):

    def __init__(self, arg):

        # train on the GPU or on the CPU, if a GPU is not available
        self.device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.root    = os.path.dirname(os.path.realpath(__file__))
        self.log_dir = os.path.join(self.root, arg.log_dir)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = logging.getLogger('Training')

        self.arg = arg

        self.d            = arg.d
        self.epochs       = arg.epochs
        self.version      = arg.version
        self.valid_coeff  = arg.valid_coeff
        self.num_workers  = arg.num_workers
        self.batch_size   = arg.batch_size
        self.heatmap_size = arg.heatmap_size
        self.auto_grad = arg.auto_grad
        self.verbose = arg.verbose
        self.save = arg.save


    def data_loader(self):

        self.logger.info(f'Loading the Dataset ...')
        dataset = CenterDataset(self.root, self.arg)
        # split the dataset in train and valid set
        torch.manual_seed(1)
        indices       = torch.randperm(len(dataset)).tolist()
        dataset_train = torch.utils.data.Subset(dataset,
                                                indices[:-int(len(indices)*self.valid_coeff)])
        dataset_valid = torch.utils.data.Subset(dataset,
                                                indices[-int(len(indices)*self.valid_coeff):])
        # define training and validation data loaders
        train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)

        valid_data_loader = torch.utils.data.DataLoader(dataset_valid,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)

        self.dataLoaders   = {
            'Train'     : train_data_loader,
            'Validation': valid_data_loader}

        self.dataset_sizes = {
            'Train'     : len(dataset_train),
            'Validation': len(dataset_valid)}

        self.logger.info(f'Finished Loading the Dataset.')
        self.logger.info(f"Total number of patches are {len(dataset)}, where {len(dataset_train)} "
                         f"ones are training, and {len(dataset_valid)} ones are validation.")

    def model_configuration(self):

        self.SC_CNN = Model(self.arg, self.logger)
        self.SC_CNN.initialize()
        self.SC_CNN.load_()
        # print('Summary of the Model:')
        # summary(self.SC_CNN.model, (self.dataset[0])[0].shape)

    def training_validation(self):

        since = time.time()
        for epoch in range(self.SC_CNN.start_epoch, self.epochs):

            print(f"Epoch {epoch}/{self.epochs-1}\n", '-'*10, "\n")
            running_loss = 0.0

            for phase in ['Train', 'Validation']:

                self.SC_CNN.model.train(True) if phase == 'Train' else self.SC_CNN.model.train(False)
                # Iterate over data.
                for batch_idx, (inputs, heatmaps, epsilons) in enumerate(self.dataLoaders[phase]):

                    start_batch = time.time()

                    inputs, heatmaps, epsilons = inputs.to(self.device), heatmaps.to(self.device), epsilons.to(self.device)
                    # forward
                    points, h_value = self.SC_CNN.model(inputs)

                    if self.auto_grad == 'PyTorch':
                        predicted = utils.heat_map_tensor(points, h_value, self.device, self.d, self.heatmap_size)
                    else:
                        # Amirali's implementation
                        predicted = self.SC_CNN.Map(points, h_value, self.device, self.d, self.heatmap_size, self.version)

                    predicted = predicted.view(predicted.shape[0], -1)
                    heatmaps  = heatmaps.view(heatmaps.shape[0], -1)

                    feature  = heatmaps.shape[1]
                    epsilons = epsilons.unsqueeze(1)
                    epsilons = torch.repeat_interleave(epsilons, repeats=feature, dim=1)

                    weights = heatmaps + epsilons

                    if phase == 'Train':
                        loss = self.SC_CNN.criterion(predicted, heatmaps)
                        weight_loss = loss * weights
                        # Sum over one data
                        # Average over different data
                        loss = torch.sum(weight_loss, dim=1)
                        loss = torch.mean(loss)
                        running_loss += loss.item() * inputs.shape[0]
                        # Update parameters
                        self.SC_CNN.optimizer.zero_grad()
                        loss.backward()
                        self.SC_CNN.optimizer.step()

                    else:
                        loss = self.SC_CNN.val_criterion(predicted, heatmaps)
                        loss = torch.sqrt(loss)
                        loss = torch.sum(loss, dim=1)
                        loss = torch.sum(loss)
                        running_loss += loss.item()

                    end_batch = time.time()
                    if self.verbose:
                        print(f"""{phase}: epoch {epoch}: batch {batch_idx+1}/{int(np.ceil(self.dataset_sizes[phase]/self.batch_size))}:
                              {end_batch-start_batch} s {self.batch_size/(end_batch-start_batch)} data/s
                              obj: {running_loss/((batch_idx+1)*self.batch_size)}-> Loss: {loss.item()}""".replace('\n',''), flush=True)

                    # Empty Catch
                    torch.cuda.empty_cache()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                print(f"{phase} -> Loss: {epoch_loss}", flush=True)
                self.writer.add_scalars(f"{self.save}/loss",
                                   {f"{phase}": epoch_loss},
                                   epoch)
                self.writer.flush()

                if phase == 'Train':
                    self.SC_CNN.scheduler.step()

                if phase == 'Validation' and epoch_loss < self.SC_CNN.best_loss:

                    self.SC_CNN.best_loss = epoch_loss
                    # Save the best model
                    utils.save_model(epoch,
                                     self.SC_CNN.model,
                                     self.SC_CNN.optimizer,
                                     self.SC_CNN.scheduler,
                                     epoch_loss,
                                     self.save)

        self.writer.close()
        time_elapsed = time.time() - since
        self.logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        self.logger.info(f"Best loss: {self.SC_CNN.best_loss:4f}")

    def run(self):
        self.data_loader()
        self.model_configuration()
        self.training_validation()
