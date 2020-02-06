#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# import the necessary packages
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())