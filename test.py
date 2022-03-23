
# Used to handle the images obtained from the dataset. 
import nntplib
from PIL import Image

# Directory functions
import os 
# Mathematical operations
import numpy as np 

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
            train_img.append(img)            
            
            
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize(
                    mean = 
                )
            ])

            img.close()



print(len(train_img))





