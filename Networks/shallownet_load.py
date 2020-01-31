#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the packages we have defined in this module
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpleDatasetLoader import SimpleDatasetLoader

#import keras loader, numpy, argparse, cv2
from tensorflow.python.keras.models import load_model
from imutils import paths
import numpy as np 
import argparse
import cv2

#the load model function is responsible for accepting the path to our trained network (an HDF5)
#file decoding the weights and optimizer inside the HDF5 file, and setting the weights inside
#our architechture so we can continue training or use the network to classify new images

#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to pre-trained model")
args=vars(ap.parse_args())

#Initialize the class labels
classLabels=["dog","cat"]

#grab the list of images in the dataset and then randomly sampled indexes into the image paths list
print("[INFO] sampling images")
imagePaths=np.array(list(paths.list_images(args['dataset'])))
idxs=np.random.randint(0,len(imagePaths),size=(10,))
#using the random indexes select 10 random paths
imagePaths=imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]

#Notice how we are preprocessing our images in the exact same manner in which we preprocess our images
#during training. Failing to do this procedure can lead to incorrect classifications. Since the network
#will be presented with patterns it cannot recognize

#ALWAYS MAKE SURE TO ENSURE YOUR TESTING IMAGES ARE PREPROCESSED IN THE SAME
#WAY AS YOUR TESTING
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
print(labels[0:100])
data = data.astype("float") / 255.0

#load the pre-trained network (Keras models are HDF5 models)
print('[INFO] loading pre-trained network...')
model= load_model(args['model'])

#make predictions on the images
print("[INFO] predicting ...")
preds=model.predict(data,batch_size=32).argmax(axis=1)

#loop over the sample images
for (i,imagePath) in enumerate(imagePaths):
    #load the example image, draw the prediction and display it to our screen
    image=cv2.imread(imagePath)
    cv2.putText(image,"Label: {}".format(classLabels[preds[i]]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


