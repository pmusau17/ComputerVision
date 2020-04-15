# import the neccessary packages
from model import CustomSegmentationModel
from imutils import paths 
import cv2
import numpy as np 

# create the model

model = CustomSegmentationModel.build(360,480,3,50)

model.train( 
train_images = 'dataset1/images_prepped_train/' ,
train_annotations = 'dataset1/annotations_prepped_train', 
checkpoints_path = 'checkpoints/custom_model', epochs = 5 ) 
