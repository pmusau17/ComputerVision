from keras_segmentation.predict import predict
from keras.models import load_model
from model import CustomSegmentationModel
import cv2
import numpy as np

#predict( 
#	checkpoints_path="checkpoints/custom_model", 
#	inp="dataset_path/images_prepped_test/0016E5_07959.png", 
#	out_fname="output.png" 
#)


# load input image, reshape for prediction 
image = cv2.imread('dataset1/images_prepped_test/0016E5_07959.png').astype('float')
image_pred= np.expand_dims(image,axis=0)

# load the ground truth image
image_gt=cv2.imread('dataset1/annotations_prepped_test/0016E5_07959.png').astype('float')

# load the custom network model and load the checkpoint weights
model = CustomSegmentationModel.build(360,480,3,50)
model.load_weights('checkpoints/custom_model.4')

# make the prediction
pred=model.predict(image_pred)[0]

# get the highest prediction from the classes
pred=pred.argmax(axis=-1)

# rehspe the prediction to back to the input image oringal shape

pred= pred.reshape((image.shape[0],image.shape[1]))

# calculate the distinct classes
classes= np.unique(pred)

# create a new RGB image so we can visualize it
output_image=np.zeros((image.shape[0],image.shape[1],3))

# generate random rgb values
colors = []
for i in classes:
	colors.append(np.random.rand(3,))

# map each label to a randomly chosen color
for j in range(len(classes)):
	indices= np.where(pred==classes[j])
	output_image[indices]=colors[j]

# display the result
cv2.imshow("original image",image/255.0)
cv2.imshow("segmentation_image",output_image)
cv2.waitKey(0)