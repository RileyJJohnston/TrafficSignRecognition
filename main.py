
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

start = time.time()

# ------ Retrieve all the Training Images -------#
train_img = []
train_lbl = []

# Path containing the training images
path = 'data\GTSRB\Final_Training\Images'
# Obtain the set of directories
set_dir = os.scandir(path)

# For each image directory
for dir in set_dir:
    if (str(dir) == '<DirEntry \'00042\'>'):
        continue # dont use this folder for now

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
            #obtain the size of the image
            width, height = img.size

            # Convert the image to grayscale
            img = ImageOps.grayscale(img)
            #img.show()

            img = fn.resize(img, size=28)
            img = fn.center_crop(img, output_size=[28])

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




# Create a train & validation split w/ 0.1 sent to validation
train_img, val_img, train_lbl, val_lbl = train_test_split(
    train_img, train_lbl, test_size=0.1)



# Neural Net architecture
class NN(Module):
    def __init__(self):
        super(NN, self).__init__()

        '''
        # Defining the sequential layers for the NN
        self.cnn_layers = Sequential(
            # 2D Convolutional layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 2nd 2D Convolutonal Layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        '''

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

        # Place the CNN layers first
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # Flatten the output 
        x = x.view(1,-1)
        #x = torch.unsqueeze(x,0)      
        x = self.linear_layers(x)     
       
        return x

# Define the training function for the NN 
def train(epoch):
    model.train()
    tr_loss = 0
    for i, tensor_img in enumerate(train_img):
        # getting the training set
        x_train = train_img[i]
        y_train = train_lbl[i]


        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # obtain 1D tensors
        output_train = output_train[0]

        # obtain 1D labels
        y_train = y_train[0]
        y_train = y_train[0]


        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        
        train_losses.append(loss_train)
        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()

    
    # Seperate the evaluation from the training
    # Run the evaluation portion of the dataset through the neural network
    model.eval()
    for i, tensor_img in enumerate(val_img):
        # getting the validation set
        x_val = val_img[i]
        y_val = val_lbl[i]


        # prediction for validation set
        output_val = model(x_val)

        # obtain 1D labels
        y_val = y_val[0]
        y_val = y_val[0]

        # obtain 1D tensors
        output_val  = output_val[0]

        # computing the validation loss
        loss_val = criterion(output_val, y_val)
        val_losses.append(loss_val)

# construct the model defined above
model = NN()
# define the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# define the loss function
criterion = CrossEntropyLoss()


# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model

for epoch in range(n_epochs):
    train(epoch)


end = time.time()
print("Total time:"  + end-start)


# test one of the images to see if it is working correctly 
print(len(train_lbl))

test_val1 = model(train_img[4])
print(train_lbl[4])
print(test_val1)