# Used to handle the images obtained from the dataset.
import nntplib
from PIL import Image, ImageStat, ImageOps

# Directory functions
import os
# Mathematical operations
import numpy as np
import torch

# import the building blocks for the neural nets
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
# import the optimizer function
from torch.optim import SGD
# function to generate the train and validation split
from sklearn.model_selection import train_test_split
# used for image processing
import torchvision.transforms
# import optimizer and loss function
from torch.optim import Adam, SGD
# import wrapper for tensors
from torch.autograd import Variable
# import time to measure time taken by aspects of the program
import time

from skimage.io import imread
import torchvision.transforms.functional as fn

class NN(Module):
    def __init__(self):
        super(NN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #takes as args: in_channels, out_channels ....   -- if greyscale, in_channels is 1.  If RGB it is 3.  The out_channels equals the number of in_channels to the next Conv2D layer
            #BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4*7*7, 42),  # flatten the output of the layers so that the second argument to the Linear function is the number of classes
            ReLU(inplace=True),
            Softmax(dim=1)
        )

    # Forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = x.view(1,-1)
        x = self.linear_layers(x)
        return x


model = torch.load('trafficRecognitionModel.pt')
model.eval()
