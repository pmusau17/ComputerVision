#import the neccessary packages

#packages we just defined
import config.imagenet_alexnet_config as config
from utils.imagenethelper import ImageNetHelper

#other incredibly helpful packages
from sklearn.model_selection import train_test_split
import numpy as np  
import progressbar
import json
import cv2

# initialize the ImageNet helper and use it to construct the set 
# of training and testing data
imhelper=ImageNetHelper(config)
train_paths,train_labels=imhelper.buildTrainingSet()
print("[INFO]","Total Image Paths:",len(train_paths),"Total Labels",len(train_labels))
val_paths,val_labels=imhelper.buildValidationSet()
print("[INFO]","Total Validation Paths:",len(val_paths),"Total Labels",len(val_labels))

# perform stratified sampling from the training set to construct a 
# a test set

print("[INFO] constructing splits...")

split = train_test_split(train_paths,train_labels,test_size=config.NUM_TEST_IMAGES,stratify=train_labels,random_state=42)

(trainPaths,testPaths,trainLabels,testLabels)=split

# construct a list pairing the training, validation, and testing 
# image paths along with their corresponding labels and output list files

datasets = [
    ("train",trainPaths,trainLabels,config.TRAIN_MX_LIST),
    ("val",val_paths, val_labels,config.VAL_MX_LIST),
    ("test",testPaths,testLabels,config.TEST_MX_LIST)
]

# initialize the list of Red, Green, and Blue averages
(R,G,B) = ([],[],[])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}".format(outputPath))
    f = open(outputPath, "w")

    #initialize the progress bar
    widgets = ["Building List:", progressbar.Percentage(), " ", progressbar.Bar()," ",progressbar.ETA()]
    pbar= progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    for (i, (path,label)) in enumerate(zip(paths,labels)):
        # write the image index, label, and output path to file
        #the join takes the first value and then put it between the succeeding elements
        # in the iterable
        row = "\t".join([str(i),str(label),path])
        f.write("{}\n".format(row))

        # if we are building the training dataset, then compute the 
        # mean of each channel in the image then update the respective lists

        if dType == "train":
            image = cv2.imread(path)
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        #update the progress bar
        pbar.update(i)
    pbar.finish()
    f.close()

#construct a dictionary of averages then serialize the means to a JSON file
print("[INFO] serializing means...")

D= {"R": np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f = open(config.DATASET_MEAN,'w')
f.write(json.dumps(D))
f.close()