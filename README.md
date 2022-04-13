# Traffic Sign Classification (SYSC 5108 Group 3)
## Abstract
##### Real-time road traffic sign recognition and classification is critical to the advance of autonomous driving.  Accuracy of sign interpretation is paramount, as a single misclassification could result in a potentially fatal accident.  Convolutional Neural Networks (CNNs) lend themselves naturally to the application of image recognition and are thus implemented in this situation for sign classification.   Although reliable identification is critical, real-world factors pose a challenge to a model’s accuracy.  Video footage taken from vehicles is noisy and often highly distorted due to velocity and illumination.  There is a need for a model capable of rapidly classifying sign images extracted from this raw footage. To be considered for integration with an autonomous vehicle, a machine learning classification model should be at least comparable to a human driver. 
##### This project implemented a CNN network paired with a fully-connected (FC) layer to perform multi-classification of input road sign. For the training and testing data, the German Road Traffic Recognition Benchmark (GTSRB) dataset was used. To train the model weights, cross-entropy loss as well as the stochastic gradient descent optimization method was employed. This allowed the model to be successful in 97.11% of all predictions.


## Repository Overview
### main.py
##### - Run this program to train the model
### TestModel.py
##### - Run this script to evaluate the performance of the model
### AnalyzeGUI.py
##### - Run this program to launch the performance visualizer tool
### convert_images.py
##### - Run this program to convert the .ppm images to .jpg format (this will already have been done for the training and testing images below)
### \data\GTSRB\Final_Training\Images
##### - Directory containing training dataset
### \data\GTSRB\Test
##### - Directory containing testing dataset