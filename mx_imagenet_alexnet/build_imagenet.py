#import the neccessary packages
import config.imagenet_alexnet_config as config
from utils.imagenethelper import ImageNetHelper

if __name__=="__main__":

    imhelper=ImageNetHelper(config)
    #print(imhelper.human_mappings,imhelper.labelMappings)
    #print(imhelper.valBlacklist)
    im_paths,im_labels=imhelper.buildTrainingSet()
    print(len(im_paths),len(im_labels))
    val_paths,val_labels=imhelper.buildValidationSet()
    print(len(val_paths),len(val_labels))
