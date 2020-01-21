#import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

#CIFAR-10 is such a common dataset that researchers benchmark machine learning and deep
#learning algorithms on, it's common to see deep learning libraries provide simple helper functions 
#to automatically load this dataset from disk. This will take a while if this is your first time downloading 
#it
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np 
import argparse

#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True, help="path to the output loss/accuracy plot")
args=vars(ap.parse_args())

#Load the training and testing data and scale it into the [0,1]
#then reshape the design matrix

print("[INFO] loading CIFAR-10 data ...")
((trainX,trainY), (testX,testY))=cifar10.load_data()
trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0
trainX=trainX.reshape((trainX.shape[0],3072))
testX=testX.reshape((testX.shape[0],3072))

#convert the labels from integers to vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

#Initialize the label names for the CIFAR-10 dataset
labelNames=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#define the 3072-1024-512-10 architechture using Keras
model=Sequential()
model.add(Dense(1024,input_shape=(3072,),activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))

#now that the architechture has been defined we can now train it
print("[INFO] training network ...")
sgd=SGD(0.01)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=32)


#evaluate the network
print("[INFO] evaluting the network...")
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


