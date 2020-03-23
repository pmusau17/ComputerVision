#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the neccessary packages
import cv2
import argparse
import imutils
#import imutils to handle resizing and path importing
from imutils import paths
import numpy as np 
#import augmentation operations from imgaug
from imgaug import augmenters as iaa 
import imgaug as ia 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())

#You can definitely set a seed
#This is massive for my applications

ia.seed(1)

data=[]
count=0
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image=cv2.imread(imagePath)
    data.append(image)
    count+=1
    if(count==11):
        break

seq = iaa.Sequential([

    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),


    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),


    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) 

transformed_data = seq(images=data)
for i in range(len(data)):
    rotated_image=transformed_data[i]
    orig=data[i]
    cv2.imshow("Rotated Image",rotated_image)
    cv2.imshow("Original",orig)
    cv2.waitKey(0)
