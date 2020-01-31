#import matplotlib so that figures can be saved in the background
import matplotlib
#this specifies the backend of matplotlib to agg which is specifically for writing to file 
#not for rendering to a wind. If you're only writing to a file from the command line 
#this is the backend for you.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#import the sklearn packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nn.conv.minivggnet import MiniVGGNet

#impor the packages that we defined in this directory
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader

from tensorflow.keras.optimizers import SGD
import numpy as np 
import argparse
from imutils import paths

#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to the the dataset")
ap.add_argument("-o","--output",required=True,help="path to the output loss/accuracy plot")
ap.add_argument("-m","--model",required=True,help="path to save the trained model hdf5")
args=vars(ap.parse_args())


#grab the list of images that we'll be describing
print("[INFO] loading images...")
image_paths=list(paths.list_images(args['dataset']))


#Let's create the pipeline to cload and process our dataset. 
#initialize the simple resize operator
sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreprocessor()

#load the dataset form disk then scale the raw pixel intensitites
#to the range [0,1]

sdl=SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels)=sdl.load(image_paths,verbose=500)
data=data.astype("float")/255.0

#Now that the data into training and testing splits of the data for training and the remaining 25% for testing 
#To be honest here its used more like validation data

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25, random_state=42)

#convert the labels from integers to vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

#this is me forcing them to take the softmax format
if(trainY.shape[1]==1):
    print("[INFO] converting binary output to softmax")
    trainY=np.hstack((trainY, 1 - trainY))
    testY=np.hstack((testY, 1 - testY))

#initialize the optimizer and model
print("[INFO] compiling model...")
#Include decay parameter which lowers the learning rate over time, this is 
#a form of regularization to reduce overfitting and increase higher classification accuracy
opt=SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(height=32,width=32,depth=3,classes=2)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#specify the number of epochs
num_epochs=40
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=num_epochs,verbose=1)
#This .save method takes the weights and state of the optimizer and serializes them to disk in HDF5
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])