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

#Stupide Cudnn nonsense
from tensorflow.compat.v1 import ConfigProto   
from tensorflow.compat.v1 import InteractiveSession
 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,help="path to pre-trained smile detector CNN")
#our script will default to reading framews from a build-in/USB webcame;
#we can specify a video as well.
ap.add_argument("-v", "--video",help="path to the (optional) video file") 
args = vars(ap.parse_args())

#load the face detector cascade and smile detector
detector=cv2.CascadeClassifier(args['cascade'])

model=load_model(args['model'])
print(model.summary())

#if a video path was not supplied, grab the reference to the web cam
if not args.get("video",False):
    camera=cv2.VideoCapture(0)
# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

try: 
    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame, convert it to grayscale, and then clone the
        # original frame so we can draw on it later in the program
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()


        # detect faces in the input frame, then clone the frame so that
        # we can draw on it
        #Here we pass in our grayscale image and indicate that for a given region to be considered a face
        #it must have a minimum width of 30 × 30 pixels. The minNeighbors attribute helps prune false-
        #positives while the scaleFactor controls the number of image pyramid
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

        #The .detectMultiScale method returns a list of 4-tuples that make up the rectangle that
        #bounds the face in the frame. The first two values in this list are the starting (x, y)-coordinates. The
        #second two values in the rects list are the width and height of the bounding box, respectively.
        #We loop over each set of bounding boxes below:

        # loop over the face bounding boxes
        for (fX, fY, fW, fH) in rects:
            # extract the ROI of the face from the grayscale image,
            # resize it to a fixed 28x28 pixels, and then prepare the
            # ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            #need to do this so LeNet can classify it
            roi = np.expand_dims(roi, axis=0)
            
            #you have to get the first dimension because it returns a list of dimensions
            (smiling,not_smiling)=model.predict(roi)[0]
            label = "Smiling" if smiling > not_smiling else "Not Smiling"

            # display the label and bounding box rectangle on the output
            # frame
            #label it above 10 pixels from the starting x,y
            cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            #create the rectangle
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
            # show our detected faces along with smiling/not smiling labels
        
        cv2.imshow("Face", frameClone)
        cv2.waitKey(3)
except KeyboardInterrupt:
    camera.release()
    cv2.destroyAllWindows()



