#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder #This package converts labels represented as strings to integers where there is one unique integer per class label
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report #Helps us evaluate the performance of classifiers and print out nicely formatted table of results
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths

import argparse






#construct the argument parse and parse the arguments

ap= argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="path to input datset")

ap.add_argument("-k","--neighbors",type=int,default=1, help="#of nearest neighbors for classification")
ap.add_argument("-j",'--jobs',type=int,default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")

args= vars(ap.parse_args())

#garb the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))

#Initialized the image preprocesser, load the dataset from the disk, #and reshape the data matrix
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])

(data,labels)=sdl.load(imagePaths,verbose=500)

print(data, labels)
