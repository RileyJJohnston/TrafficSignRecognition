# Used to handle the images obtained from the dataset.
from PIL import Image

# Directory functions
import os
# Mathematical operations
import numpy as np
import torch

# import the building blocks for the neural nets
from torch.nn import Linear, Sequential, Conv2d, MaxPool2d, Module

from sklearn.metrics import confusion_matrix
# used for image processing
import torchvision.transforms
# Softmax function
import torch.nn.functional as F
# import time to measure time taken by aspects of the program
import torchvision.transforms.functional as fn

# import seaborn and matplotlib libraries for displaying the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

class NN(Module):
    def __init__(self):
        super(NN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #takes as args: in_channels, out_channels ....   -- if greyscale, in_channels is 1.  If RGB it is 3.  The out_channels equals the number of in_channels to the next Conv2D layer
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4*7*7, 42),  # flatten the output of the layers so that the second argument to the Linear function is the number of classes
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

# ------ Retrieve all the testing Images -------#
test_img = []
test_lbl = []
test_img_int = []
test_lbl_int = []

# Path containing the testing images
path = 'data\GTSRB\Test'
# Obtain the set of directories
set_dir = os.scandir(path)

# For each image directory
for dir in set_dir:
    if (str(dir) == '<DirEntry \'00042\'>'):
        continue

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

            #obtain the size of the image
            width, height = img.size

            img = fn.resize(img, size=50)
            img = fn.center_crop(img, output_size=[50])

            # Configure the conversion to tensor
            transform = torchvision.transforms.Compose([
                # convert the image to a tensor of range 0->1
                torchvision.transforms.ToTensor(), 
            ])

            # apply the transform
            tensor_img = transform(img)

            # Add the new tensor to the list
            test_img.append(tensor_img)

            # label is stored in directory index   
            test_lbl.append(torch.tensor([[int(str(dir_name))]]))#).view(1, -1))

            # Close the file
            img.close()
correct = 0
numImages = 0
for i, lb in enumerate(test_lbl):
    # obtain the prediction using the model
    predict = model(test_img[i])
    test_img_int.append(int(torch.argmax(F.softmax(predict[0], dim=0))))
    test_lbl_int.append(int(lb.item()))
    numImages += 1
    if lb.item() == torch.argmax(F.softmax(predict[0], dim=0)): 
        correct += 1 # increment the count 

print("Model correctly predicted " + str(correct) + " images correctly out of " + str(numImages))
print(numImages)
print("Test accuracy: " + str(100*correct/numImages) + "%")

#Generate our confusion matrix
confusionMatrix = confusion_matrix(test_img_int, test_lbl_int)
print(confusionMatrix)

#Now that we have generated our confusion matrix, use Searborn to display it
ax = sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='g')

ax.set_title('Traffic Sign Prediction Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values')

## Display the visualization of the Confusion Matrix.
plt.show()