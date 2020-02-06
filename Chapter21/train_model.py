#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from nn.conv.LeNet import LeNet
from preprocessing.captcha_preprocess import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import cv2 
import os

#construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-o",'--output',required=True,help="path to the location where you want to store the model")
args=vars(ap.parse_args())

#Initialize the data and labels
data=[]
labels=[]

#loop over the input images
for imagePath in paths.list_images(args['dataset']):
    #load the image, preprocess it and store it in the datalist
    image=cv2.imread(imagePath)
    #convert the image to grayscale
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #convert the image to the proper size
    image=preprocess(image,28,28)
    #make sure that the channel order is the one used by the 
    #keras backend
    image=img_to_array(image)
    data.append(image)
    #the label comes from the directory
    #root_directory/class_label/image_filename.jpg
    label = int(imagePath.split(os.path.sep)[-2])
    labels.append(label)

#Normalize the inputs into the range [0,1]
data=np.array(data,dtype='float')/255.0
labels=np.array(labels)

#partition the data into training and testing splits using 75% of the
#data for training and the remaining 25% for testing
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.01)

#convert the labels from integers into vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)
print(testY[0])
#Let's create our LeNet Model and SGD optimizer
print("[INFO] compiling the model")
model=LeNet.build(width=data[0].shape[1],height=data[0].shape[0],depth=data[0].shape[2],classes=len(trainY[0]))
opt=SGD(lr=0.01)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

# train the network
print("[INFO] training network...")
num_epochs=30
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=num_epochs, verbose=1)


# evaluate the network
print("[INFO] evaluating network...")
#predictions = model.predict(testX, batch_size=32)
#print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["output"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()