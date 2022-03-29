
# Used to handle the images obtained from the dataset.
from cProfile import label
import enum
import nntplib
from PIL import Image, ImageStat, ImageOps

# Directory functions
import os
# Mathematical operations
import numpy as np
import torch

# import some data loading functions 
from torch.utils.data import Dataset, DataLoader

# import the building blocks for the neural nets
from torch.nn import Linear, ReLU, CrossEntropyLoss, LeakyReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
# import the optimizer function
from torch.optim import SGD
# function to generate the train and validation split
from sklearn.model_selection import train_test_split
# Functions used to summarize the structure of the nn
from torchsummary import summary
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
            # img = ImageOps.grayscale(img)
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
train_img, val_img, train_lbl, val_lbl = train_test_split(train_img, train_lbl, test_size=0.1)

# Dataset class used to handle the images used in training
class ImageDataset(Dataset):
    # Define the images and the lables fo rthe images
    def __init__(self, img, lbl): 
        self.img_labels =  lbl
        self.img = img

    # return the length of the dataset
    def __len__(self): 
        return len(self.img_labels)

    # obtain the image and the corresponding label
    def __getitem__(self, index): 
        return self.img[index], self.img_labels[index] 




# Neural Net architecture
class NN(Module):
    def __init__(self):
        super(NN, self).__init__()


        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1), #takes as args: in_channels, out_channels ....   -- if greyscale, in_channels is 1.  If RGB it is 3.  The out_channels equals the number of in_channels to the next Conv2D layer

            LeakyReLU(0.5,inplace=True),

            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(0.5,inplace=True),

            # Adding a another filter layer to the CNN
            Conv2d(4,4, kernel_size = 3, stride =1, padding = 1),
            MaxPool2d(kernel_size=2, stride=2),

            # LeakyReLU(0,inplace=True),
            
        )

        self.linear_layers = Sequential(
            # Linear(4*7*7 , 42),
            Linear(4*7*7 , 42),
            # ReLU(inplace=True),
            # Softmax(dim=1)
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
        # generate a softmax output for the final layer     
        # x = F.softmax(x,1)
        return x

# Define the training function for the NN 
def train(epoch):
    model.train()
    tr_loss = 0
    # counter used to count all the correct predictions
    correct = 0 
    
    # cycle through the datasets and loaders
    i = 1
    for i in range(1,3):
        if i == 1: 
            data_set = ImageDataset(train_img, train_lbl)         
            data_loader = DataLoader(dataset=data_set,batch_size=1, shuffle=True)
            train_set_size = data_loader.__len__() # get the size of the dataset
        else: 
            data_set =  ImageDataset(val_img, val_lbl) 
            data_loader = DataLoader(dataset=data_set,batch_size=1, shuffle=True)
            eval_set_size = data_loader.__len__() # get the size of the dataset


                # for each img/label in the batch
        for i, data in enumerate(data_loader): 
            #obtain the image and label
            img, lbl = data
            
            #!!!!! NEED to change this if larger batches are going to be used
            lbl = torch.reshape(lbl, [1])

            # zero the gradients
            optimizer.zero_grad()

            # obtain the prediction using the model
            predict = model(img)
    
            # Compute the loss
            loss = criterion(predict, lbl)
            loss.backward()

            # Adjust the learning weights
            optimizer.step()

            # Obtain the loss
            tr_loss = loss.item()

            # Verify if the prediction is prediction is correct and update the counter
            if lbl == torch.argmax(F.softmax(predict,1)): 
                correct += 1 # increment the count 

    # Obtain the final percentage of correct predictions 
    print("Correct Predictions: " + str(round(correct/(eval_set_size + train_set_size)*100,2)) + " %")

# construct the model defined above
model = NN()

# obtain a summary of the model dimensions -> currently an issue wiht the dimension used by the function
# summ =summary(model, (1,28,28))

# define the optimizer
optimizer = Adam(model.parameters(), lr=0.1)
# optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9) # Fix this later
# define the loss function
criterion = CrossEntropyLoss()


# defining the number of epochs
n_epochs = 10
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model

# Complete the training process for each epoch.
for epoch in range(n_epochs):
    print("Training in Epoch: " + str(epoch))
    train(epoch)

# Calculate the final training time for the network
end = time.time()
print("Total time: "  + str(round(end-start,2)) + " s")

# Save the model to a file
torch.save(model,'trafficRecognitionModel.pt')


# # Test a set amount of training images
# index = 1 
# for index in range(1,10):
#     test_val1 = F.softmax(model(train_img[index]),1)
#     print(train_lbl[index].item()+1)
#     print(test_val1)
#     print(torch.argmax(test_val1))






