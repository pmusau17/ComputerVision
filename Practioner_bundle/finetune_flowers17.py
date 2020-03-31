#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# user defined preprocessing libraries and network definitions
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from datasets.simpleDatasetLoader import SimpleDatasetLoader
from nn.conv.fcheadnet import FCHeadNet
# for data augmentation
from keras.preprocessing.image import ImageDataGenerator

#optimizers
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16

# layers, model
from keras.layers import Input
from keras.models import Model 
from imutils import paths

# customary packages
import numpy as np 
import argparse
import os 

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to save the output model")
args = vars(ap.parse_args())


# initialize some constants
HEAD_TUNING_EPOCHS = 25
FINAL_TUNING_EPOCHS = 100
BATCH_SIZE = 32
#

# Initalize the ImageDataGeneraor, responsible for performing data augmentation
# when training the network

aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,
zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

# grab the list of images, then extract, 
# the class label names from the image paths

print ('[INFO] loading images...')
imagePaths=sorted(list(paths.list_images(args['dataset'])))
classNames=np.unique([pt.split(os.path.sep)[-2] for pt in imagePaths])
print(classNames)

# Initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0,1]

sdl= SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.astype("float")/255.0

# partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# now comes the main event, the network surgery

baseModel = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
print(baseModel.summary())

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel=FCHeadNet.build(baseModel,len(classNames),256)

# place the head FC model on top of the base model -- this will 
# become the actual model we will train

model = Model(inputs=baseModel.input,outputs=headModel)

print(model.summary())

# loop over all layers in the base model and freeze them so they 
# will * not* be updated during the training process.

for layer in baseModel.layers:
    layer.trainable = False

# compile our model. This needs to be done after our setting our
# layers to being non-trainable

print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become 
# initialized with actual learned values verus being purely random

print ('[INFO] training head...')
model.fit_generator(aug.flow(trainX,trainY,batch_size=BATCH_SIZE),
                    validation_data=(testX,testY),epochs=HEAD_TUNING_EPOCHS,
                    steps_per_epoch=len(trainX)//HEAD_TUNING_EPOCHS,verbose=1)

# evaluate the network after initialization 
print("[INFO] evaluating after initialization")
predictions = model.predict(testX, batch_size = BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

# now that the head FC layers have been trained/initialized, lets 
# unfreeze the final set of CONV layers and make them trainable
# the only reason that we didn't unfreeze the whole thing is that 
# VGG is a deep architechture, if classification accuracy continues
# to improve (wihtout overfitting), you may consider unfreezing more 
# layers in the body
for layer in baseModel.layers[15:]:
    layer.trainable = True


# for the changes to the model to take affect we need to recompile the
# model, this time using SGD with a *very* small learning rate

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

print ('[INFO] fine-tuning model...')
model.fit_generator(aug.flow(trainX,trainY,batch_size=BATCH_SIZE),
                    validation_data=(testX,testY),epochs=FINAL_TUNING_EPOCHS,
                    steps_per_epoch=len(trainX)//FINAL_TUNING_EPOCHS,verbose=1)

# evaluate the network after initialization 
print("[INFO] evaluating after fine-tuning")
predictions = model.predict(testX, batch_size = BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

#save te model to disk
print("[INFO] serializing model...")
model.save(args['model'])