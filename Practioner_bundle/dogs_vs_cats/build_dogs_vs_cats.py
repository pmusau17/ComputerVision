# import the neccessary packages 

from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths 
import numpy as np 
import progressbar
import json
import cv2
import os 

# import sys so we can use packages outside of this folder in
# either python 2 or python 3, I know it's janky, chill
import sys
import os
from pathlib import Path 

#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from input_output.hdf5datasetwriter import HDF5DatasetWriter


# grab the paths to the images 

trainPaths = sorted(list(paths.list_images(config.IMAGES_PATH)))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]

# create the label encoder
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the 
# testing split from the training data

split = train_test_split(trainPaths,trainLabels,test_size=config.NUM_TEST_IMAGES,stratify=trainLabels,random_state=3020)

(trainPaths,testPaths,trainLabels,testLabels) = split


# perform another stratified sampling, this time to build the validation 
# data

split = train_test_split(trainPaths,trainLabels,test_size=config.NUM_VAL_IMAGES,stratify=trainLabels,random_state=3021)

(trainPaths,valPaths,trainLabels,valLabels) = split

# construct a list pairing their training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files

datasets = [
    ("train",trainPaths,trainLabels,config.TRAIN_HDF5),
    ("val",valPaths,valLabels,config.VAL_HDF5),
    ("test",testPaths,testLabels,config.TEST_HDF5)
]

# intialize the image preprocessor and list of RGB channel averages
aap = AspectAwarePreprocessor(256,256)
(R,G,B) = ([],[],[])

# loop over the dataset tuples

for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),256,256,3),outputPath)


    # Initialize the progress bar

    widgets = ["Building Dataset: ",progressbar.Percentage(),' ',progressbar.Bar(),' ',progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    # loop over the image paths
    for (i,(path,label)) in enumerate(zip(paths,labels)):
        # load the image and process it

        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the 
        # respective lists

        if dType == "train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label to the HDF5 dataset
        writer.add([image],[label])
        pbar.update(i)



    pbar.finish
    writer.close()

    # construct the dictionary of averages, then serialize the means to a JSON file

    print("[INFO] serializing means...")
    
    D = {"R": np.mean(R), "G": np.mean(G),"B":np.mean(B)}
    
    f = open(config.DATASET_MEAN,'w')
    f.write(json.dumps(D))
    f.close()