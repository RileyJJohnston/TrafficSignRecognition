import tkinter as tk
import os
from PIL import ImageTk, Image
import random
import torch
import torchvision.transforms
import torchvision.transforms.functional as fn
# import the building blocks for the neural nets
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
# Softmax function
import torch.nn.functional as F

class NN(Module):
    def __init__(self):
        super(NN, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #takes as args: in_channels, out_channels ....   -- if greyscale, in_channels is 1.  If RGB it is 3.  The out_channels equals the number of in_channels to the next Conv2D layer
            MaxPool2d(kernel_size=2, stride=2),
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

path = 'data\\GTSRB\\Test\\00001\\00000_00010.jpg'
dirName = '00000'
signClass = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


def predict():
    global path
    global dirName
    global model

    # Fetch the currently displayed image
    print(path)
    img = Image.open(path) 
    img = fn.resize(img, size=50)
    img = fn.center_crop(img, output_size=[50])

    # Configure the conversion to tensor
    transform = torchvision.transforms.Compose([
        # convert the image to a tensor of range 0->1
        torchvision.transforms.ToTensor(), 
    ])

    # Convert the image to a tensor
    test_img = transform(img)

    # Get the label associated with this image
    test_lbl = torch.tensor([[int(str(dirName))]])

    # Close the file
    img.close()

    # Run the image through the deep learning model    
    predict = model(test_img)
    print(F.softmax(predict, dim=0))
    print(F.softmax(predict))
    print(signClass[int(int(torch.argmax(F.softmax(predict))))])
    # Display the prediction in the GUI
    outputText2.delete(0, tk.END)
    outputText2.insert(0, str(signClass[int(int(torch.argmax(F.softmax(predict))))]))

def nextImage():
    global path
    global dirName
    dirName = random.choice(os.listdir("./data/GTSRB/Test"))
    while(True):
        imageName = random.choice(os.listdir("./data/GTSRB/Test/" + str(dirName)))
        if (imageName.endswith('.jpg')):
            break
    path = 'data\\GTSRB\\Test\\' + str(dirName) + '\\' + str(imageName)
    print(path)

    # open the image
    img = Image.open(path)
    img = img.resize((410, 380), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    # update the label with the new image
    label.configure(image=img)
    label.image = img

# Load the trained CNN model
model = torch.load('trafficRecognitionModel.pt')
model.eval()

# Create the GUI
window = tk.Tk()
window.geometry("500x500")
window.title("Traffic Sign Recognition")

# Add a title
greeting = tk.Label(text="Traffic Sign Recognition", font='Helvetica 18 bold')#, bg='white')
greeting.pack(side=tk.TOP)

# Display an image in the centre of the window
# First create a frame to fit the image into
frame = tk.Frame(window, width=350, height=350)
frame.pack(side = tk.TOP)
frame.place(anchor='center', relx=0.5, rely=0.45)
# Next, open the image
img = Image.open(path)
img = img.resize((380, 380), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
# Finally, create a Label widget to display image
label = tk.Label(frame, image = img)
label.pack()

# Now create the buttons
buttonFrame = tk.Frame()
predictbutton = tk.Button(buttonFrame, text = "Predict", fg = "black", bg='grey', height=2, width=10, command = predict)  
nextbutton = tk.Button(buttonFrame, text = "Next", fg = "black", bg='grey', height=2, width=10, command = nextImage)  
predictbutton.pack( side = tk.LEFT)  
nextbutton.pack( side = tk.RIGHT)

# display the output text in an Entry widget at the bottom of the screen
outputText2 = tk.Entry(font='Helvetica 16', width = 25)
outputText2.pack(side=tk.BOTTOM)
outputText2.place(anchor='center', relx=0.5, rely=0.96)
 
# Pack the button frame onto the main window
buttonFrame.pack(side = tk.BOTTOM)
buttonFrame.place(anchor='center', relx=0.5, rely=0.88)

# Execute the main loop
window.mainloop()
