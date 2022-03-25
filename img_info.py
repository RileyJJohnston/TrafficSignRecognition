# this script was created to obtain information about the images used during the learning process
import os 
from PIL import Image

# Generate empty array to store the width and the height
w = []
h = []

#Root folder path
path = 'data\GTSRB\Final_Training\Images'

# Obtain the set of directories
set_dir = os.scandir(path)

# For each image directory
for dir in set_dir: 
    #obtain the name of the directory
    dir_name = dir.name

    dir_path = path + '\\' + dir_name

    # Obtain all the images in the directory
    imgs = os.scandir(dir_path)
    # for each img
    for img in imgs: 
        # if it is a ppm file, then proceed
        if img.name.endswith(".jpg"):
            # obtain the image
            im = Image.open(dir_path + '\\' + img.name)
            # obtain the size of the image
            width,height = im.size

            
            w.append(width)
            h.append(height)


print(min(w))
print(min(h))









