import torch
import torch.nn as nn
from loss.loss import BCE_Loss
from model.SC_CNN import SC_CNN
from model.HeatMap import HeatMap
from model.SC_CNN_v2 import SC_CNN_v2
import other.utils as utils



class Model(object):

    def __init__(self, arg, logger):
        self.M            = arg.M
        self.lr           = arg.lr
        self.load         = arg.load
        self.logger       = logger
        self.version      = arg.version
        self.momentum     = arg.momentum
        self.patch_size   = arg.patch_size
        self.heatmap_size = arg.heatmap_size

        self.device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def main_model(self):
        # get the new model
        if self.version==2:
            self.model = SC_CNN_v2(self.M,
                                   self.heatmap_size)
        # get the original model
        else:
            self.model = SC_CNN(self.M,
                                self.patch_size,
                                self.heatmap_size,
                                self.version)
        # move model to the right device
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def heatmap_layer(self):
        # Generating Heatmap withour my own grad
        self.Map = HeatMap.apply

    def optimizer(self):
        # construct an optimizer
        params         = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params,
                                         lr=self.lr,
                                         momentum=self.momentum,
                                         weight_decay=0.0005)
    def scheduler(self):
        # learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=[60,100],
                                                              gamma=0.1)
    def training_loss(self):
        # This is Pytorch loss function
        # BCE
        self.criterion = nn.BCELoss(reduction='none')
        # This is my own written loss function
        # SCCNN
        # self.criterion = BCE_Loss.apply

    def validation_loss(self):
        # Validation criterion
        self.val_criterion = nn.MSELoss(reduction='none')

    def load_(self):
        if self.load:
            checkpoint = torch.load(self.load) if torch.cuda.is_available() else \
                         torch.load(self.load, map_location='cpu')

            self.model.load_state_dict(utils.load_model(checkpoint))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss   = checkpoint['loss']
            self.logger.info(f"Model Loaded!")

        else:
            # Just a big number
            self.start_epoch = 0
            self.best_loss   = 10**10

    def initialize(self):
        self.main_model()
        self.heatmap_layer()
        self.optimizer()
        self.scheduler()
        self.training_loss()
        self.validation_loss()
