#import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from nn.conv.minivggnet import MiniVGGNet
from tensorflow.python.keras.callbacks import ModelCheckpoint #This class will enable us to checkpoint and serialize our networks to disk
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.datasets import cifar10
import argparse
import os 

#whenever we find an incremental improvement in model performance
#Let's parse our command line arguments
#The weights the path to the output directory that will store our serialized 
#models during the training process

ap=argparse.ArgumentParser()
ap.add_argument("-w","--weights",required=True, help="path to weights directory")
args=vars(ap.parse_args())

#Load the training and testing data, then scale it into the range [0,1]

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#convert the labels from integers to vectors
lb=LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
#we are using learning rate decay, momentum with nesterov acceleration
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss
#We construct the file name template in a special way. The first number is 
#the epoch number out to three digits. The second number the the validation loss
#out to four digits
#if you wanted to monitor the other variables change val_loss to something else 
#in H.history. I
fname = os.path.sep.join([args["weights"],"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])


#save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
# mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization 
# of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the 
# name of the monitored quantity.
# period: Interval (number of epochs) between checkpoints.


checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",save_best_only=True, verbose=1)

#Let us now instantiate th callbacks

callbacks=[checkpoint]
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

