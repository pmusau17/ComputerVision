# import sys so we can use packages outside of this folder in
# either python 2 or python 3,
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

# import the neccessary packages
from config import tiny_imagenet_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from utils.ranked import rank_accuracy
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import json 

# load the RGB means for the training set 
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means['G'],means['B'])
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators 
testGen = HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# make predictions on the testing data 
print("[INFO] predicting on test data")
predictions = model.predict_generator(testGen.generator(),
                                     steps=testGen.numImages // 64, max_queue_size = 64 *2)

# compute the rank-1 and rank-5 accuracies

(rank1,rank5) = rank_accuracy(predictions,testGen.db['labels'])
print("[INFO] rank-1: {:.2f}".format(rank1*100))
print("[INFO] rank-5: {:.2f}".format(rank5*100))

# close the database
testGen.close() 


