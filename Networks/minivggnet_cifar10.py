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
from nn.conv.minivggnet import MiniVGGNet

from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.datasets import cifar10
import numpy as np 
import argparse

#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to the output loss/accuracy plot")
ap.add_argument("-m","--model",required=True,help="path to save the trained model hdf5")
args=vars(ap.parse_args())

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#normalize pixels from 0 to 1
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#convert the labels from integers to vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
# you have to do this because they come back as 0-9 when you load them above
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

#initialize the optimizer and model
print("[INFO] compiling model...")
#Include decay parameter which lowers the learning rate over time, this is 
#a form of regularization to reduce overfitting and increase higher classification accuracy
opt=SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(height=32,width=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#train the network
print("[INFO] training network...")

#specify the number of epochs
num_epochs=40
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=num_epochs,verbose=1)
#This .save method takes the weights and state of the optimizer and serializes them to disk in HDF5
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=labelNames))

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