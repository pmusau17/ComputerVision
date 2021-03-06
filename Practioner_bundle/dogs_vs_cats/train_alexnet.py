# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import sys so we can use packages outside of this folder in
# either python 2 or python 3
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

from config import dogs_vs_cats_config as config 
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.croppreprocessor import CropPreprocessor
from callbacks.trainingmonitor import TrainingMonitor
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.conv.alexnet import AlexNet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os 

# construct the training image generator for data augmentation 

aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,
                        height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

# load the RGB means for the training set
means = json.loads((open(config.DATASET_MEAN).read()))

# initialize the image preprocesors 
sp = SimplePreprocessor(227,227)
pp = PatchPreprocessor(227,227)
mp = MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128,aug=aug,preprocessors=[pp,mp,iap])

valGen = HDF5DatasetGenerator(config.VAL_HDF5,128,preprocessors=[sp,mp,iap])

# Initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3,beta_1=0.9,beta_2=0.999)

model = AlexNet.build(height=227,width=227,depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])

# construct the set of callbacks

path = os.path.sep.join([config.OUTPUT_PATH,"{}.png".format(os.getpid())])

callbacks = [TrainingMonitor(path)]

# train the network

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 128,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 128,
    epochs = 75,
    max_queue_size = 128 * 2,
    callbacks = callbacks, verbose =1 
) 

# save the model to model 
print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()