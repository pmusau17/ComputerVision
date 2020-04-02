# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the neccessary packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import os 

# Training Constants

BATCH_SIZE = 64
EPOCHS = 40 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o",'--output',required=True,help='path to output directory')
ap.add_argument("-m","--models",required=True, help="path to output models directory")
ap.add_argument("-n","--num-models",type=int,default=5,help="# of models to train")
args=vars(ap.parse_args())

# load the training and testing data, then scale it into the 
# range [0,1]

((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX= testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb= LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

# initialize the lavel names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog",
            "frog","horse","ship","truck"]

# Data Augmentation 
# Allow images to be randomly rotated 10 degrees, shifted by a factor of 0.1
# and randomly horizontally flipped

aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,fill_mode="nearest")

# loop over the number of models to train

for i in np.arange(0, args['num_models']):
    
    # initialize the optimizer and model
    print("[INFO] training model {}/{}".format(i+1,args['num_models']))

    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9,nesterov=True)

    model = MiniVGGNet.build(width=32, height= 32, depth=3, classes=testY.shape[1])
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

    # Train the network
    H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BATCH_SIZE),validation_data=(testX,testY),epochs=EPOCHS,steps_per_epoch=len(trainX) //64,verbose =1)

    # save the model to disk

    p = [args['models'],"model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # evaluate the network

    predictions = model.predict(testX,batch_size=BATCH_SIZE)
    report =classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames)

    # save the classification report to file
    p = [args['output'],"model_{}_report.txt".format(i)]
    f = open(os.path.sep.join(p),"w")
    f.write(report)
    f.close()

    # plot the training loss and accuracy

    p = [args['output'],"model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    
    for key in H.history.keys():
        plt.plot(np.arange(0, EPOCHS), H.history[key], label=key)
    
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()    



