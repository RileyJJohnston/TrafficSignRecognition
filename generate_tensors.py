# This code is used to generate and store all of the tensors (images) used by the machine learning algorithm.

# Used to handle the images obtained from the dataset. 
import nntplib
from PIL import Image, ImageStat

# Directory functions
import os 
# Mathematical operations
import numpy as np
from torch import tensor, save

# import the building blocks for the neural nets
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Dropout
# import the optimizer function
from torch.optim import SGD
# used for image processing
import torchvision.transforms




# ------ Retrieve all the Training Images -------#
train_img = []

# Path containing the training images 
path = 'data\GTSRB\Final_Training\Images'
# Obtain the set of directories
set_dir = os.scandir(path)

# For each image directory
for dir in set_dir: 
    #obtain the name of the directory
    dir_name = dir.name
    dir_path = path + '\\' + dir_name

    # Obtain all the images in the directory
    files = os.scandir(dir_path)
    # for each img
    for file in files: 
        # if it is a jpg file, then proceed
        if file.name.endswith(".jpg"):
            img = Image.open(dir_path + "\\" + file.name)
                     
            # Configure the conversion to tensor
            transform = torchvision.transforms.Compose([
                # convert the image to a tensor of range 0->1
                torchvision.transforms.ToTensor(), 
            ])
            # apply the transform
            tensor_img = transform(img)

            # Add the new tensor to the list
            train_img.append(tensor_img)   
            # Close the file
            img.close()





