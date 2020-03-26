#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from datasets.simpleDatasetLoader import SimpleDatasetLoader
from nn.conv.minivggnet import MiniVGGNet

from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
args= vars(ap.parse_args())

# grab the lst of images that we'll be describing, then extract 
# the class label names from the image paths

print("[INFO] loading images...")
imagePaths=sorted(list(paths.list_images(args['dataset'])))

#this is incredibly inefficient but oooooookkkk
classNames=np.unique([pt.split(os.path.sep)[-2] for pt in imagePaths])

#initialize the image preprocessors
aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk and then scale the raw pixel intensities
# to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels) = sdl.load(imagePaths,verbose=50)
data=data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing

(trainX,testX,trainY,testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

######### THIS IS THE MAIN EVENT ###########
# construct the image generator for data augmentation

# Here we'll allow the images to be:
# Randomly rotated by +- 30 degrees
# Horizontally and vertically shifted by a factor of 2
# Sheared by 0.2
# Zoomed by uniformly sampling in the range [0.8,1.2]
# Randomly horizontally flipped

aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                            horizontal_flip=True,fill_mode="nearest")

#Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.05)

model = MiniVGGNet.build(width=64,height=64,depth=3,classes=len(classNames))
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])
print(model.summary())

# training the network used to train our network has to change slightly as we are now using an image generator
print("[INFO] training network...")

#the batch size you are using
batch_size=32

# num epochs
num_epochs=100

H = model.fit_generator(aug.flow(trainX,trainY,batch_size=batch_size),
    validation_data=(testX,testY), steps_per_epoch=len(trainX) //batch_size,epochs =num_epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
for key in H.history.keys():
    plt.plot(np.arange(0,num_epochs),H.history[key],label=key)

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()