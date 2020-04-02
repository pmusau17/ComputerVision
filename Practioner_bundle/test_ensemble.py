# import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from keras.datasets import cifar10
import numpy as np 
import argparse

# python glob package is a module that finds all the pathnames matching a 
# specified pattern according to the rules used in the Unix shell, the results are returned in arbitary order.
import glob
import os 

# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument('-m','--models',required=True,
                help="path to models directory")
args = vars(ap.parse_args())

# load the testing data, then scale it into the range [0,1]
(testX,testY) = cifar10.load_data()[1]
testX = testX.astype('float')/255.0

# initialize the lavel names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog",
            "frog","horse","ship","truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# construct the path used to collect the models, then utilize the 
# models list

modelPaths = os.path.sep.join([args["models"],"*.model"]) # this will produce something like this path_name/*.models
modelPaths = list(glob.glob(modelPaths))
models = []

# Let's go ahead and load each model from disk now
# loop over the model paths, loading the model, and adding it to the list of 
# models 

for (i,modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i+1,len(modelPaths)))
    models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions =[]

# loop over the models
for model in models:
    # use the current model to make predictions on the testing data,
    # then store these predictions in the aggregate predictions list
    predictions.append(model.predict(testX,batch_size=64))

# average the probabilities across all model predictions, then show 
# a classification report 

predictions = np.average(predictions,axis=0)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=labelNames))