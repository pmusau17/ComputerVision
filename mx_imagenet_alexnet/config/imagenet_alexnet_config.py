#import the neccessary paths
from os import path

#define the base paths to where the ImageNet dataset
#devkit are stored on disk
#This works because I defined  a 
BASE_PATH="imagenet"


#based on the base path derive the images base path, image sets
#path, and devkit path

IMAGES_PATH = BASE_PATH

#This path points to the directory containing the important train_cls.txt
#and val.txt which explicitly list out the filenames
IMAGE_SETS_PATH=BASE_PATH

#the path to our DevKit lives
DEVKIT_PATH=path.sep.join([BASE_PATH,'ILSVRC/devkit/data'])

#Let's define the path that maps 1,000 possible WordNet to the 
# (1) the unique identifying integers and human readable labels
WORD_IDS = path.sep.join([DEVKIT_PATH,'map_clsloc.txt'])


#define the paths to the training file that maps the (partial)
#image filename to integer class label
TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH,"train"])

#define the paths to the validation filenames along with the file 
#that contains the ground truth validation labels

VAL_LIST = path.sep.join([IMAGE_SETS_PATH,"val"])

VAL_LABELS = path.sep.join([DEVKIT_PATH,'ILSVRC2015_clsloc_validation_ground_truth.txt'])

#define the path to the validation files that are blacklisted 
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH,'ILSVRC2015_clsloc_validation_blacklist.txt'])

# since we do not have access to the testing data we need to to 
# take a number of images from the training data and use it instead

NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

#define the path to the output training, and validation, and testing
#lists 

TRAIN_MX_LIST = path.sep.join([BASE_PATH,"lists/train.lst"])
VAL_MX_LIST = path.sep.join([BASE_PATH,"lists/val.lst"])
TEST_MX_LIST = path.sep.join([BASE_PATH,"lists/test.lst"])

#define the path to the output training, validation, and testing 
#image records

TRAIN_MX_REC = path.sep.join([BASE_PATH,"rec/train.rec"])
VAL_MX_REC = path.sep.join([BASE_PATH,"rec/val.rec"])
TEST_MX_REC = path.sep.join([BASE_PATH,"rec/test.rec"])

#define the path to the dataset mean
DATASET_MEAN = "output/imagenet_mean.json"

#define the batch size and number of devices used for training
BATCH_SIZE = 128
NUM_DEVICES = 1 #my man had 8 

