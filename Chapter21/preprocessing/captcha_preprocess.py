#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import imutils
import cv2

def preprocess(image,height,width):
    #grab the dimensions of the image and then initialize the padding values
    (h,w)=image.shape[:2]

    #if the width is greater than the height then resize along the width
    if w>h:
        image= imutils.resize(image,width=width)
    #otherwise the height is greater than the width so resize along the height
    else: 
        image=imutils.resize(image,height=height)

    #now that we have enlargened or shrunk the image to the size of our choice
    #need to fix the other dimensions

    #Deterime the padding values for the width and the height to obtain the target dimensions
    #One of these is gonna be zero from above.
    padW=int((width-image.shape[1])/2.0)
    padH=int((width-image.shape[0])/2.0)

    #pad the image then apply one more resizing to handle any rounding issues.
    #There will be cases where we are one pixel off
    #the padding order is top, bottom, left,right
    image=cv2.copyMakeBorder(image,padH,padH,padW,padW,cv2.BORDER_REPLICATE)
    image=cv2.resize(image,(width,height))

    #return the image
    return image


""" testing for My own sake
path='../datasets/1/000000.png'
image2=cv2.imread(path)
cv2.imshow('digit',image2)
cv2.waitKey(0)

imagepp=preprocess(image2,224,224)
cv2.imshow('digit',imagepp)
cv2.waitKey(0) """
