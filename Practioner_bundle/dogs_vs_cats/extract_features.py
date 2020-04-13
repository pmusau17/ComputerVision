# import the neccessary packages

from keras.applications import ResNet50 
from keras import Input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import progressbar
from imutils import paths 
import argparse
import random
import os 
import numpy as np 

""" The basic premise of this class is that we can perform transfer learning via feature extraction 
since Image Net was trained on finer grained classifications of dogs and cats"""

# import sys so we can use packages outside of this folder in
# either python 2 or python 3
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

from input_output.hdf5datasetwriter import HDF5DatasetWriter

ap = argparse.ArgumentParser()
ap.add_argument("-d",'--dataset', required=True,help='path to input dataset')
ap.add_argument("-o",'--output', required=True,help='path to output HDF5 file')
ap.add_argument("-b",'--batch-size', type=int, default=16 ,help='path to output HDF5 file')
ap.add_argument("-s",'--buffer-size', type=int, default=1000 ,help='size of feature extraction buffer')
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args['batch_size']


# grab the list of images that we'll be be describing them randomly
# shuffle them to allow for easy training and testing splits vis 
# array slicing during training time

print ("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the 
# labels

labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le= LabelEncoder()
labels= le.fit_transform(labels)

# Load the ResNet50 Network
print("[INFO] loading network...")
model = ResNet50(weights="imagenet",include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# Because we can't have nice things and they changed the resnet definitions
# We have to add the Global Average Pooling ourselves
output = GlobalAveragePooling2D()(model.layers[-1].output)
model = Model(inputs=model.input, outputs=output)
#print(model.layers[-1].output)

print(model.summary())

# in order to store the features extracted from ResNet50 to disk, we need a HDF5 writer
# Initialize the HDF5 dataset writer then store the class label names in the dataset

dataset = HDF5DatasetWriter((len(imagePaths),2048),args['output'],dataKey="features",buffSize=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# Initialize the progressbar
widgets = ["Extracting Features: ", progressbar.Percentage()," ",
    progressbar.Bar()," ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()

# loop over the images in batches

for i in range(0,len(imagePaths),bs):
    # extract the batch of images and labels and then initialize the 
    # list of actual images that will be passed through the network
    # for feature extraction

    batchPaths = imagePaths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j,imagePath) in enumerate(batchPaths):
        # load the input image using Keras helper utility 

        # ensure that the image is resized to 224 x 224 pixels
        image = load_img(imagePath,target_size=(224,224))
        image = img_to_array(image)

        # preporcess the image by (1) expanding the dimensions and 
        # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset

        image = np.expand_dims(image,axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    
    # pass the images through the network and use the outputs as 
    # our actual features

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages,batch_size=bs)

    # reshape the features so that each image is represented by 
    # a flattened vector of the MaxPooling2D outputs

    features = features.reshape((features.shape[0],2048))

    dataset.add(features,batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()









