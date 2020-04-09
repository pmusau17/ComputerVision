# import the neccessary packages
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.croppreprocessor import CropPreprocessor
import cv2 
import argparse
from imutils import paths
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator

""" The purpose of this script is to help you see what some of these preprocessing 
Classes do"""

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the image")
args= vars(ap.parse_args())

PATCH_EXAMPLES = 10 

# load the image
image =cv2.imread(args['image'])
cv2.imshow("Original Image",image)
cv2.waitKey(0)


# resize the image to the krizhevsky pre-size

sp=SimplePreprocessor(256,256)

processed_image = sp.preprocess(image)
cv2.imshow("Krizhevesky 256 x 256",processed_image)
cv2.waitKey(0)

# DISPLAY Examples of patch processing 

pp=PatchPreprocessor(227,227)

for i in range(PATCH_EXAMPLES):

    random_patch=pp.preprocess(processed_image)
    cv2.imshow("Patch: {} 227 x 227".format(i+1),random_patch)
    cv2.waitKey(0)

# Display mean subtraction

mp= MeanPreprocessor(125.14524921569824,116.12856107711792,106.25390909347534)
mean_image=mp.preprocess(processed_image)
cv2.imshow("mean subtraction",mean_image)
cv2.waitKey(0)

# Display results from the crop preprocessor 
cp= CropPreprocessor(227,227)
ims= cp.preprocess(processed_image)

# display the ims
for img in ims:
    cv2.imshow("cropped image",img)
    cv2.waitKey(0)




hg = HDF5DatasetGenerator("data/hdf5/test.hdf5",32,preprocessors=[sp,mp])

gen= hg.generator()
ims,labels = next(gen)
print(ims,labels)

for img in ims:
    cv2.imshow("generated_image",img)
    cv2.waitKey(0)