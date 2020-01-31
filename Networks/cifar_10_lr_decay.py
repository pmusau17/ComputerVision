# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from nn.conv.minivggnet import MiniVGGNet
from tensorflow.python.keras.callbacks import LearningRateScheduler #this class allows you to define your own learning rate schedulers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np 
import argparse


# Depending on how the callback is
# defined, Keras will call this function at the start or end of every epoch, mini-batch update, etc. The
# LearningRateScheduler will call step_decay at the end of every epoch, allowing us to update
# the learning prior to the next epoch starting.

def step_decay(epoch):
    #Initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha=0.01
    factor=0.25
    dropEvery= 5

    #compute learning rate for the current epoch
    # compute learning rate for the current epoch

    #starting from a learning rate and it will decrease by a factor of 1/(factor) every five
    #epochs
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

# define the set of callbacks to be passed to the model during
# training
callbacks = [LearningRateScheduler(step_decay)]

# initialize the optimizer and model
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


#define the number of epochs
num_epochs=40
# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=64, epochs=num_epochs, callbacks=callbacks, verbose=1)

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