#import the neccessary pacakges
import numpy as np 
import os 


class ImageNetHelper:

    def __init__(self, config):
        #store the configuration object that holds all the filepaths to the imagenet datasets
        self.config = config

        #build the label mappings and validation blacklist
        self.labelMappings,self.human_mappings = self.buildClassLabels()
        #self.valBlacklist = self.buildBlacklist()

    def buildClassLabels(self):
        #load the contents of the file that maps the WordNet IDs
        #to integers, then initialize the label mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split()
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
