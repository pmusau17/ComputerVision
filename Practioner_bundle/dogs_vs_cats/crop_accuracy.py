# import the necessary packages

from config import dogs_vs_cats_config as config

# import sys so we can use packages outside of this folder in
# either python 2 or python 3
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.croppreprocessor import CropPreprocessor
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.ranked import rank_accuracy

from tensorflow.python.keras.models import load_model
import numpy as np 
import progressbar
import json 

# load the RGB means for the training set
means = json.loads((open(config.DATASET_MEAN).read()))

# initialize the image preprocesors 
sp = SimplePreprocessor(227,227)
cp = CropPreprocessor(227,227)
mp = MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ImageToArrayPreprocessor()

# load the pretrained model 
model = load_model(config.MODEL_PATH)

# Initialize the testing dataset generator, then make the predictions on the testing data

print("[INFO] predicting on test data (no crops)...")

testGen = HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],
                              classes=2)

predictions = model.predict(testGen.generator(),steps=testGen.numImages//64,max_queue_size=64 *2,verbose=1)

# compute the rank-1  accuracy

print(testGen.db['labels'][0:100])

(rank1, _) = rank_accuracy(predictions,testGen.db['labels'])

print("[ACCURACY] rank-1: {:.2f}%".format(rank1*100))

# re-initialize the testing set generator, this time excluding the the 'Simple Preprocessor'
# When we don't preprocess the images the size is 256 x 256 which is fine since w are trying now to crop 
# 10 different versions of the images at a time.

testGen = HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[mp],classes=2)
predictions = []

# Initialize the progress bar widgets
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]

# Initialize the progressbar
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64, widgets=widgets).start()

# loop over a single pass of the test data 
for (i, (images,labels)) in enumerate(testGen.generator(passes=1)):
    # loop over each of the individual images 
    for image in images: 
        # apply the crop preprocessor to the image to generate 10
        # seperate crops, then convert them from images to arrays

        crops = cp.preprocess(image) # this results in 10 different crops of the same image
        crops = np.array([iap.preprocess(c) for c in crops],dtype="float32")

        # make predictions on the crops and then average them together
        # to obtain the final prediction

        pred = model.predict(crops)
        print(pred)
        predictions.append(pred.mean(axis=0))






