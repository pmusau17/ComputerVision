# import the necessary packages
import os 

# initialize the base path for the LISA dataset
BASE_PATH = "lisa"

# build the path to the annotations file 
ANNOT_PATH = os.path.sep.join([BASE_PATH,"allAnnotations.csv"])

# build the path the output training and testing record files, 
# along with the class labels file 

TRAIN_RECORD = os.path.sep.join([BASE_PATH,"records/training.record"])
TEST_RECORD  = os.path.sep.join([BASE_PATH,"records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH,"records/classes.pbtxt"])

# initialize the test split size

TEST_SIZE = 0.25

# initialize the class labels dictionary 
# in this particular set of experiments we'll only be interested
# in three classes:
# 1. Pedestrian crossings
# 2. Signal Ahead
# 3. Stop signs

CLASSES ={"pedestrianCrossing":1,"signalAhead":2,"stop":3}

