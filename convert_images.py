# This code is used to convert all the images from ppm file format to jpg
from PIL import Image
import os 
import string


#Root folder path
path = 'data\GTSRB\Final_Test'

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
        if img.name.endswith(".ppm"):
            # obtain the image
            im = Image.open(dir_path + '\\' + img.name)
            # change the file extension to jpg
            name = img.name.replace(".ppm", ".jpg")
            # Save the new jpg file
            im.save(dir_path + '\\' + name)           

set_dir.close()