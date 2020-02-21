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





#Example of randomly rotating am image by -20 to 20 degrees, you can also fill 
#up any newly created pixels with a random RGB color: 
rotate=iaa.Affine(rotate=(-25,25))


#An Example of doing image augmentation sequentially
seq=iaa.Sequential([
    iaa.Crop(px=(0,16)), #crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),   #horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0,3.0)) #blur images with a sigma of 0 to 3.0
])


count=0
data=[]
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image=cv2.imread(imagePath)
    data.append(image)
    count+=1
    if(count==11):
        break

#apply the sequential transformations 
transformed_data=seq(images=data)

for i in range(len(data)):
    rotated_image=transformed_data[i]
    orig=data[i]
    cv2.imshow("Rotated Image",rotated_image)
    cv2.imshow("Original",orig)
    cv2.waitKey(0)