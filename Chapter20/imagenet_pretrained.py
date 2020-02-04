#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the neccessary packages
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.applications import Xception #You can only use Tensorflow Backend
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications import VGG19
#imagenet utils has a set of utils that will make preprocessing images easier
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
import numpy as np 
import argparse
import cv2
import imutils

#construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",help="name of pre-trained network to use. Options: vgg16, vgg19, inception,xception,resnet")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
"vgg16": VGG16,
"vgg19": VGG19,
"inception": InceptionV3,
"xception": Xception, # TensorFlow ONLY
"resnet": ResNet50
}

# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument is invalid run with help to see available options")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
#This is what is required by VGG16, VGG19, and Resnet
inputShape = (224, 224)

#This preprocessing function does mean subtraction and normalization
preprocess = imagenet_utils.preprocess_input


#Inception and Xception require 229 x 229 image sizes
#They also require a different pre-processing function
#which performs a different kind of scaling
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


#load the network weights from disk (NOTE: if this is the first time it will take time to
# download)

print("[INFO] loading {}...".format(args["model"]))
#select the correct network class
Network = MODELS[args["model"]]

#select the weights trained on imagenet, 
#there are other weights
model = Network(weights="imagenet")

#Now load the input image using the Keras helper utility while ensuring
#the image is correctly resized to 'inputShape', the required input dimensions
#for the ImageNet pre-trained network

image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

#the image has now been reshaped to a NumPy array of shape
#(inputShape[0], inputShape[1], 3) however we need to expand the dimension by making the shape 
# (1, inputShape[0], inputShape[1], 3) so we can pass it through the network. 
#this because we/train classify images in batches
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with ’{}’...".format(args["model"]))
preds = model.predict(image)

#.decode_predictions, to give us a list of  ImageNet class label IDs, 
# “human-readable” labels, and the probability associated with each class label. The top-5 predictions 
# (i.e., the labels with the largest probabilities) are then printed to our terminal
P = imagenet_utils.decode_predictions(preds)
print(P)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
orig= imutils.resize(orig, width=600)
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)