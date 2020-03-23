#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#import the neccessary pacakges
import numpy as np 
import os 
from imutils import paths

class ImageNetHelper:

    def __init__(self, config):
        #store the configuration object that holds all the filepaths to the imagenet datasets
        self.config = config

        #build the label mappings and validation blacklist
        self.labelMappings,self.human_mappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def buildClassLabels(self):
        #load the contents of the file that maps the WordNet IDs
        #to integers, then initialize the label mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split('\n')
        labelMappings={}
        human_mappings={}
        #loop over the labels
        for row in rows:
            #split the row into the WordNet ID, label integer, and human readable label
            (wordID, label, hrLabel)=row.split(" ")

            #update the label mappings dictionary using the word ID
            #as the key and the label as the value, subtracting '1' from the label since 
            # MATLAB is one-indexed while Python is zero-indexed 

            labelMappings[wordID]=int(label)-1
            human_mappings[wordID]=hrLabel

        return labelMappings,human_mappings

    
    def buildBlacklist(self):
        #load the list of blacklisted image IDs and convert them to a set
        rows = open(self.config.VAL_BLACKLIST)
        rows = set(rows.read().strip().split('\n'))
        return rows

    def buildTrainingSet(self):
        im_paths=sorted(list(paths.list_images(self.config.TRAIN_LIST)))
        labels=[]
        #loop over the image paths
        for path in im_paths:
            
            splitPath=path.split(os.path.sep)
            label=splitPath[-2]
            labels.append(self.labelMappings[label])
        
        return (np.array(im_paths),np.array(labels))

    def buildValidationSet(self):
        # Initialize the list of images and class labels
        im_paths=sorted(list(paths.list_images(self.config.VAL_LIST)))
        #Some images are blacklisted so can't use all of the images returned by paths 
        return_paths=[]
        labels=[]

        #load the contents of the file that contains the *actual* 
        #ground-truth
        valLabels= open(self.config.VAL_LABELS).read().strip().split('\n')

        #print(len(valLabels),im_paths[:100])

        #loop over the validation data
        for (img,label) in zip(im_paths, valLabels):
            #split the path and get the image name
            img_split = img.split(os.path.sep)
            #remove the JPEG extension
            im_name=img_split[-1].replace('.JPEG','')
            #the last part before the image extensions is im_num
            im_num=im_name.split('_')[-1]
            #convert the im_num to a int and back to a str
            im_num=str(int(im_num))
            #if it's a blacklisted image skip it
            if im_num in self.valBlacklist:
                continue
            else:
                return_paths.append(img)
                #don't forget we have to zero index our labels
                labels.append(int(label)-1)

        return np.array(return_paths), np.array(labels)





