#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import the neccessary packages
from imutils import paths
import argparse
import imutils 
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained we purposely pad the input image so it’s not possible for a given digit to
        # touch the border
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.copyMakeBorder(gray, 8, 8, 8, 8,cv2.BORDER_REPLICATE)

        #We binarize the input image via Otsu's thresholding method

        #threshold the image to reveal the digits, now the image is binary. Foreground is white 
        #background is black, it's a standard paradigm in image processing
        thresh = cv2.threshold(gray_img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        #uncomment these lines if you want to see the thresholding
        #cv2.imshow("thresh", thresh)
        #key = cv2.waitKey(0)

        # find contours in the image, keeping only the four largest
        # ones. Explained simply contours are simply a curve joining all the 
        # continuos points (along the boundary) that have the same color or intensity
        #Theoretically, Contours means detecting curves
        cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    

        #Just in case there is noise in the image we sort the contours by their area, keeping only the four largest ones 
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        #Given our contours we can extract each of them by computing the bounding box
        #So we loop over the contours
        for c in cnts:
            #compute the bounding box for the contour then extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            #get the region of interest and increase it by 5 pixels in every direction
            roi = gray_img[y - 5:y + h + 5, x - 5:x + w + 5]
            # display the character, making it larger enough for us
            # to see, then wait for a keypress
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            #This part is going to enable you hand label images by pressing keys on the keyboard
            if key == ord("‘"):
                print("[INFO] ignoring character")
                continue
            # grab the key that was pressed and construct the path
            # the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])

            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labeled character to file
            count = counts.get(key, 0)

            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
            # increment the count for the current key
            counts[key] = count + 1
            print(key,counts)
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
    except Exception as e: 
        print(e)