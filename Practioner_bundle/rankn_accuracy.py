# import the neccessary packages
from utils.ranked import rank_accuracy
import argparse
import pickle
import h5py 

# construct the argument parse and parse the arugments

ap = argparse.ArgumentParser()
ap.add_argument('-d','--db',required=True,help='path HDF5 database')
ap.add_argument('-m','--model',required=True,help='path to the pre-trained model')
args = vars(ap.parse_args())

# load the pre-trained modle 
print("[INFO] loading the pre-trained model...")
model = pickle.loads(open(args['model']).read())

# open the HDF5 dataset for reading then determine the index of
# the training and testing split, provided that this data was 
# already suffled prior to writing it to disk

db = h5py.File(args['db'],"r")
i = int(db['labels'].shape[0] * 0.75)

# make predictions on the testing set and then compute the rank-1
# and rank-5 accuracies

print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank_accuracy(preds,db['labels'][i:],5)

# display the rank-1 and rank-5 accuracies

print("[INFO] rank-1: {:.2f}%".format(rank1*100))
print("[INFO] rank-5: {:.2f}%".format(rank5*100))

db.close()