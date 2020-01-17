#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the neccessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse


#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True, help="path to input dataset")
args= vars(ap.parse_args())

#grab the list of image paths
print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))

sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.reshape((data.shape[0],3072))


#encode the labels as integers
le=LabelEncoder()
labels=le.fit_transform(labels)

#partition the data into training and testing splits usung 75% of #the data for training and the remaining 25% for testing
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)


#Here we are going to train some regularizers
for r in (None,"l1","l2"):
    #train a SGD classifier using a softmax loss function and the specified regularization function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    #eta0 is the learning rate and penalty is the regularizer
    model=SGDClassifier(loss="log",penalty=r,max_iter=10,learning_rate="constant",eta0=0.01,random_state=42)
    model.fit(trainX,trainY)

    #evaluate the classifier
    acc=model.score(testX,testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r,acc*100))







