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
pp = PatchPreprocessor(227,227)
mp = MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ImageToArrayPreprocessor()

# load the pretrained model 
model = load_model(config.MODEL_PATH)

# Initialize the testing dataset generator, then make the predictions on the testing data

print("[INFO] predicting on test data (no crops)...")

testGen = HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],
                              classes=2)

predictions = model.predict(testGen.generator(),steps=testGen.numImages//64,max_queue_size=64 *2,verbose=1)

# compute the rank-1 and rank-5 accuracies

(rank1, _) = rank_accuracy(predictions,testGen.db['labels'])

print("[ACCURACY] rank-1: {:.2f}%".format(rank1*100))

