# import sys so we can use packages outside of this folder in
# either python 2 or python 3,
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from callbacks.trainingmonitor import TrainingMonitor
from callbacks.epochcheckpoint import EpochCheckpoint
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.conv.deepergooglenet import DeeperGoogLeNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K 
import argparse
import json 

# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to the output checkpoint directory")
ap.add_argument('-m','--model',type=str,help="path to *specific* model checkpoint to load")
ap.add_argument('-s','--start-epoch',type=int,default=0,help="epoch to restarting training at")
args = vars(ap.parse_args())

# construct the training image generator fof data augmentation

aug = ImageDataGenerator(rotation_range=18,zoom_range=0.15,
                         width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set 
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means['G'],means['B'])
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators 

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5,64,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen   = HDF5DatasetGenerator(config.VAL_HDF5,64,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)


# if there is no specific model checkpoint supplied, then intialize the network and compile the model 
if args['model'] is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet.build(width=64,height=64,depth=3,classes=config.NUM_CLASSES,reg=0.0002)
    opt = Adam(1e-3)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

# otherwise, load the checkpoint from disk 
else: 
    print("[INFO] loading {}...".format(args['model']))
    model = load_model(args['model'])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))

    K.set_value(model.optimizer.lr,1e-5)

    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


# construct the set of callbacks
# -- EpochCheckpoint will save a checkpoint file every 5 epochs
# -- TrainingCheckpoint will log the loss and accuracy to a log file and plot for every epoch of training

callbacks = [
    EpochCheckpoint(args['checkpoints'],every=5,startAt=args['start_epoch']),
    TrainingMonitor('logs',jsonPath='logs',startAt=args['start_epoch'])
]

# train the network 
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64, 
    validation_data= valGen.generator(),
    validation_steps= valGen.numImages // 64,
    epochs = 10,
    max_queue_size = 64 *2,
    callbacks = callbacks,
    verbose =1
)

# close the databases
trainGen.close()
valGen.close()

