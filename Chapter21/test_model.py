#import the neccessary packages
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
from preprocessing.captcha_preprocess import preprocess
from imutils import contours,paths
import imutils 
import numpy as np
import argparse
import cv2

#construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path to input dataset")
ap.add_argument("-m",'--model',required=True,help="path to the input model")
args=vars(ap.parse_args())


# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])


# randomly sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,),replace=False)

#loop over the images
for imagePath in imagePaths:
    # load the image and convert it to grayscale, then pad the image
    # to ensure digits caught only the border of the image are
    # retained
    image=cv2.imread(imagePath)

    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #These are the captcha images so we need to pad them to make sure the 
    #numbers are not cutoff on the border
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,cv2.BORDER_REPLICATE)

    # threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #As before we need to get the contours and draw bounding boxes around them
    cnts,hierarcy=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #keep only the four largest contours 
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    #now in this case the order of the conotours is important and we need to sort them 
    #from left to right
    cnts,bounding_boxes=contours.sort_contours(cnts)

    #ok this is only so that we can draw our answer on the image
    #we need to convert it back to RGB
    output = cv2.merge([gray] * 3)
    predictions = []

    #cv2.imshow("gray",gray)
    #cv2.waitKey(0)
    #loop over the contours
    for c in cnts:
        #compute the bounding box for the contour then extract the digit
        (x,y,w,h)= cv2.boundingRect(c)
        
        #expand the bounding box by 5 pixels
        roi= gray[y-5:y+h+5,x-5:x+w+5]

        #preprocess the ROI and classify if then classify it
        roi=preprocess(roi,28,28)
        #Expand the shape of an array. Insert a new axis that will appear at the axis position 
        #in the expanded array shape. The reason we do this is because we predict in batches so
        #we need to convert the image shape from (28,28,1) to (1,28,28,1)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        #now we draw the prediction on the output image
        #bounding box for the digit
        cv2.rectangle(output, (x - 2, y - 2),(x + w + 4, y + h + 4), (255, 0, 0), 1)

        #place the label five units above and to the left of the digit
        cv2.putText(output, str(pred), (x - 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # show the output image
    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", output)
    cv2.waitKey()