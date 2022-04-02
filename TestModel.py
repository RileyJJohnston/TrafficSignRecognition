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
# Softmax function
import torch.nn.functional as F
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
            # ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(4),
            # ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4*7*7, 42),  # flatten the output of the layers so that the second argument to the Linear function is the number of classes
            # ReLU(inplace=True),
            # Softmax(dim=1)
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






# ------ Retrieve all the Training Images -------#
train_img = []
train_lbl = []

# Path containing the training images
path = 'data\GTSRB\Test'
# Obtain the set of directories
set_dir = os.scandir(path)

# For each image directory
for dir in set_dir:
    if (str(dir) == '<DirEntry \'00042\'>'):
        continue # dont use this folder for now

    #obtain the name of the directory
    dir_name = dir.name

    dir_path = path + '\\' + dir_name

    print(dir_path)
    # Obtain all the images in the directory
    files = os.scandir(dir_path)


    # for each img
    for file in files:
        # if it is a jpg file, then proceed
        if file.name.endswith(".jpg"):
            img = Image.open(dir_path + "\\" + file.name) 
            #print(dir_path + "\\" + file.name)
            #obtain the size of the image
            width, height = img.size
            #img.show()
            # Convert the image to grayscale
            #img = ImageOps.grayscale(img)
            #img.show()
    
            img = fn.resize(img, size=50)
            img = fn.center_crop(img, output_size=[50])

            # Configure the conversion to tensor
            transform = torchvision.transforms.Compose([
                # convert the image to a tensor of range 0->1
                torchvision.transforms.ToTensor(), 
            ])

            # apply the transform
            tensor_img = transform(img)

            #tensor_img = tensor_img.view(1, -1)
            # Add the new tensor to the list
            train_img.append(tensor_img)
            # label is stored in directory index   

            train_lbl.append(torch.tensor([[int(str(dir_name))]]))#).view(1, -1))
            # Close the file
            img.close()

#print(train_img)
#print(train_lbl)

print("finished reading images")
correct = 0
numImages = 0
for i, lb in enumerate(train_lbl):
    # obtain the prediction using the model
    predict = model(train_img[i])
    #print(predict[0])
    #print(predict[0].size())
    # Verify if the prediction s prediction is correct and update the counter
    #print(F.softmax(predict[0], dim=0))
    #print(torch.argmax(F.softmax(predict[0], dim=0)))
    #print(lb.item())
    #print("finished")
    numImages += 1
    if lb.item() == torch.argmax(F.softmax(predict[0], dim=0)): 
        correct += 1 # increment the count 

print("Model correctly predicted " + str(correct) + " images correctly out of " + str(numImages))
print(numImages)
print("Test accuracy: " + str(100*correct/numImages) + "%")

'''
# Create a train & validation split w/ 0.1 sent to validation
train_img, val_img, train_lbl, val_lbl = train_test_split(train_img, train_lbl, test_size=0.1)

index = 1

for index in range(1,10):
    test_val1 = F.softmax(model(train_img[index]),1)
    print(train_lbl[index].item()+1)
    print(test_val1)
'''



