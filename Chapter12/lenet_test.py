#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import keras loader, numpy, argparse, cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K 
import numpy as np 
import argparse
import cv2
from sklearn import datasets


#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to pre-trained model")
args=vars(ap.parse_args())

#The MNIST dataset has already been preprocessed
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data
data = data.astype("float") / 255.0
#in this data set each image is represented by a 784-d vector so we need to reshape it
if K.image_data_format()=="channels_first":
    data=data.reshape(data.shape[0],1,28,28)

#Otherwise, we are using "channels last" ordering     
else:
    data=data.reshape(data.shape[0],28,28,1)


#specify the number of tests
num_tests=10
idxs=np.random.randint(0,data.shape[0],size=(num_tests,))
test_images=data[idxs]

#load the pre-trained network (Keras models are HDF5 models)
print('[INFO] loading pre-trained network...')
model= load_model(args['model'])

#make predictions on the images
print("[INFO] predicting ...")
preds=model.predict(test_images,batch_size=32).argmax(axis=1)

#display each of the predictions
for i in range(len(preds)):
    image=test_images[i]
    print(preds[i])
    cv2.imshow("Image", image)
    cv2.waitKey(0)
