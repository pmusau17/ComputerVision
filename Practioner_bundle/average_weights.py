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
from combine_models import CombineModels

# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument('-m','--models',required=True,
                help="path to models directory")
ap.add_argument('-sp','--searchpattern',required=True,
                help="path name patern extension such as \'*.model\'")
args = vars(ap.parse_args())


# Define Combining Models 
cm = CombineModels(args['models'],args['searchpattern'])

# define the weights to do averaging

num_models = len(cm.models)

weights = [(1)/float(num_models) for i in range(0,num_models)]

# create the average from the ensemble
model=cm.model_weight_ensemble(weights)

# load the testing data, then scale it into the range [0,1]
(testX,testY) = cifar10.load_data()[1]
testX = testX.astype('float')/255.0

# initialize the lavel names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog",
            "frog","horse","ship","truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

for j in range(1,6):
    model=cm.evaluate_n_members(j)
    _, test_acc = model.evaluate(testX,testY)
    print("num_models {}, test_accuracy {}".format(j,test_acc))



