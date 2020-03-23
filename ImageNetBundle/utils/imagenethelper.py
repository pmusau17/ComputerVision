#import the neccessary pacakges
import numpy as np 
import os 


class ImageNetHelper:

    def __init__(self, config):
        #store the configuration object that holds all the filepaths to the imagenet datasets
        self.config = config

        #build the label mappings and validation blacklist
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

        def buildClassLabels(self):
            #load the contents of the file that maps the WordNet IDs
            #to integers, then initialize the label mappings dictionary