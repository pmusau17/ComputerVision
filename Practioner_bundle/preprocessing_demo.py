# import the neccessary packages
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.croppreprocessor import CropPreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
import cv2 
import argparse
from imutils import paths
from input_output.hdf5datasetgenerator import HDF5DatasetGenerator
import json

""" The purpose of this script is to help you see what some of these preprocessing 
Classes do"""

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=False,help="path to the image")
args= vars(ap.parse_args())

PATCH_EXAMPLES = 10 

# load the image
if args['image'] is not None:
    image =cv2.imread(args['image'])
    cv2.imshow("Original Image",image)
    cv2.waitKey(0)


# Define preprocessors

sp=SimplePreprocessor(256,256)
pp=PatchPreprocessor(227,227)
cp= CropPreprocessor(227,227)
iap = ImageToArrayPreprocessor()
means = json.loads((open('dogs_vs_cats/output/dogs_vs_cats_mean.json').read()))
mp = MeanPreprocessor(means['R'],means['G'],means['B'])


# resize the image to the krizhevsky pre-size
if args['image'] is not None:
    processed_image = sp.preprocess(image)
    cv2.imshow("Krizhevesky 256 x 256",processed_image)
    cv2.waitKey(0)

# DISPLAY Examples of patch processing 
if args['image'] is not None:
    for i in range(PATCH_EXAMPLES):

        random_patch=pp.preprocess(processed_image)
        cv2.imshow("Patch: {} 227 x 227".format(i+1),random_patch)
        cv2.waitKey(0)


# Display mean subtraction
if args['image'] is not None:
    mean_image=mp.preprocess(processed_image)
    cv2.imshow("mean subtraction",mean_image)
    cv2.waitKey(0)

# Display results from the crop preprocessor 
if args['image'] is not None:
    ims= cp.preprocess(processed_image)
    # display the ims
    for img in ims:
        cv2.imshow("cropped image",img)
        cv2.waitKey(0)




hg = HDF5DatasetGenerator("data/hdf5/train.hdf5",32,preprocessors=[pp,mp,iap])
gen= hg.generator()
ims,labels = next(gen)
print(ims,labels)

for img in ims:
    cv2.imshow("generated_image",img)
    cv2.waitKey(0)


testGen = HDF5DatasetGenerator("data/hdf5/test.hdf5",64,preprocessors=[sp,mp,iap],
                              classes=2)
gen = testGen.generator()
ims,labels = next(gen)
print(ims,labels)

for img in ims:
    cv2.imshow("generated_image",img)
    cv2.waitKey(0)