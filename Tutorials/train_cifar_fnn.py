#import the necessary packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from cifar_fnn import Cifar_FNN
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 
import numpy as np 
import argparse


# Define constants
NUM_EPOCHS = 140
BATCH_SIZE = 32

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="directory where we will output the model")
args = vars(ap.parse_args())


# load the data and scale it to the [0,1] range
print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0 

# reshape the images
trainX= trainX.reshape((-1,3072))
testX= testX.reshape((-1,3072))
# convert the labels from integers to vectors
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the lavel names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog",
            "frog","horse","ship","truck"]

# build the model 
opt=SGD(lr=0.13)
model = Cifar_FNN.build((trainX.shape[1],),trainY.shape[1])
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
print(model.summary())

#save the best performing models
fname=args['output']
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",save_best_only=True,save_weights_only=False, verbose=1)

# train the network
print("[INFO] training network...")
H= model.fit(trainX,trainY,
            validation_data=(testX,testY),
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            callbacks=[checkpoint],
            verbose=1)

# evaluate the network 
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
for key in H.history.keys():
    plt.plot(np.arange(0,NUM_EPOCHS),H.history[key],label=key)

plt.title("Training Losss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()