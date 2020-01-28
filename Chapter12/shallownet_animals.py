#import the neccessary sklearn packages 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#impor the packages that we defined in this directory
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet

#import keras, imutils, numpy, and finally argparse
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse


#construct the argument parser and parse the arguments

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
args=vars(ap.parse_args())

#grab the list of images that we'll be describing
print("[INFO] loading images...")
image_paths=list(paths.list_images(args['dataset']))

#Let's create the pipeline to cload and process our dataset. 
#initialize the simple resize operator
sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreprocessor()

#load the dataset form disk then scale the raw pixel intensitites
#to the range [0,1]

sdl=SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels)=sdl.load(image_paths,verbose=500)
data=data.astype("float")/255.0

#Now that the data into training and testing splits of the data for training and the remaining 25% for testing 
#To be honest here its used more like validation data

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25, random_state=42)

#convert the labels from integers to vectors
trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)
#this is me forcing them to take the softmax format
if(trainY.shape[1]==1):
    print("[INFO] converting binary output to softmax")
    trainY=np.hstack((trainY, 1 - trainY))
    testY=np.hstack((testY, 1 - testY))


#initialize the optimizer and model 

#Here we initialize the SGD optimizer with a learning rate of 0.005. The fit method is then 
#called and it requires us to pass the training and testing data. The network is trained for 
#100 epochs using minibatch sizes of 32

print("[INFO] compiling model...")
opt=SGD(lr=0.01)
model=ShallowNet.build(width=32,height=32,depth=3,classes=2)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

print("[INFO] training network....")

#specify the number of epochs
num_epochs=200

H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=256,epochs=num_epochs,verbose=1)
print(H.history)
#evaluate the network
print("INFO evaluating network...")
predictions=model.predict(testX,batch_size=64)
print(predictions[0])
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat","dog"]))

#plot the training loss and accuracy

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
plt.show()
