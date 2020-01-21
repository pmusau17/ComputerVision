#import the neccessary packages
from sklearn.preprocessing import LabelBinarizer #This encodes our vectors as one hot vectors, One-hot encoding transforms categorical labels from a single integer to a vector.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

#The Sequential class indicates that our network will be feedforward and layers will
#be added to the class sequentially, one on top of the other
from keras import Sequential

#The dense is the implementation of fully connected layers
from keras.layers.core import Dense
#Stochastic gradient descent
from keras.optimizers import SGD

import matplotlib.pyplot as plt 
import argparse
import numpy as np



# construct the argument parse and parse the arguments
#it has only one argument and this is where our figure of the loss will be saved onto the disk
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")


# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits don't need to zero mean since 
#This is already the case for the MNIST data
data = dataset.data.astype("float") / 255.0

#split the training set 75% and 25%
(trainX, testX, trainY, testY) = train_test_split(data,
dataset.target, test_size=0.25)


# convert the labels from integers to one hot vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01) #initialize the SGD optimizer with a learning rate of 0.01
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# In most circumstances,
# such as when you are tuning hyperparameters or deciding on a model architecture, you will want
# your validation set to be a true validation set and not your testing data
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))


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

#With this model we are able to obtain 92% accuracy. Furthermore the training and validation curves match each other nearly identically 
#indicating there is no overfitting or issues with the training process. While 92% accuracy is pretty good with a convolutional neural network
#we can reach up to 92% accuracy.

#While on the surface it may appear that our (strictly) fully connected is performing well, we can actually do much better. And as we'll see in the next section 
#stricly fully networks applied to more challenging datasets in some cases can do just barely better than guessing randomly

