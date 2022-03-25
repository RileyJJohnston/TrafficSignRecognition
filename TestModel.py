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

# ------ Retrieve all the Testing Images -------#
test_img = []
test_lbl = []

# Path containing the testing images
path = 'data\GTSRB\Final_Test\Images'
# Obtain the set of directories
set_dir = os.scandir(path)

dir_path = set_dir

# Obtain all the images in the directory
files = os.scandir(dir_path)
for file in files:
    # if it is a jpg file, then proceed
    if file.name.endswith(".jpg"):
        if file.name.endswith(".jpg"):
            img = Image.open(dir_path + "\\" + file.name)
            #obtain the size of the image
            width, height = img.size
            #if (width < 28 or height < 28):
            #   print("Ignoring image with dimensions: " + str(height) + "x" + str(width))
            img = ImageOps.grayscale(img)
            #img.show()
            #print("before: " + str(img.size))
            img = fn.resize(img, size=28)
            img = fn.center_crop(img, output_size=[28])
            #img.show()
            #print("after: " + str(img.size))
            # Configure the conversion to tensor
            transform = torchvision.transforms.Compose([
                # convert the image to a tensor of range 0->1
                torchvision.transforms.ToTensor(),
            ])

            # apply the transform
            tensor_img = transform(img)

            #tensor_img = tensor_img.view(1, -1)
            # Add the new tensor to the list
            test_img.append(tensor_img)
            # label is stored in directory index
            #print(type(int(str(dir_name))))
            # ).view(1, -1))
            test_lbl = 
            test_lbl.append(torch.tensor([[test_lbl]]))
            # Close the file
            img.close()