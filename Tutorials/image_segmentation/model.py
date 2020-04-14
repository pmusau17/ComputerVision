# import the neccessary packages
from keras.models import Input, Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Dropout
from keras.layers import concatenate

# custom module implemented by divam gupta
# all definitions can be found here: https://github.com/divamgupta/image-segmentation-keras/
from keras_segmentation.models.model_utils import get_segmentation_model

class CustomSegmentationModel:

    @staticmethod
    def build(input_height,input_width,channels,n_classes):

        img_input = Input(shape=(input_height,input_width,channels))

        # first blocks, consist of two convolutional lauers and one max pooling layer
        # conv1 and conv2 contain intermediate information that will be used by the encoder
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        # Let's define the decoder layers. We concatenate the intermediate encoder outputs with 
        # the intermediate decoder outputs which are the skip connections 

        # convoltuional block
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

        # concatentate third block of convoltions with the third block of convolutions
        # I'm assuming they must have the same shape as given by Upsampling2D. The upsampling 
        # layer simply doubles the input size os its the exact opposite of the max pooling operation
        # that occurs at pool2 so the upsampled conv3 has the same dimensions of conv2 
        # we then concatenate the 64 filters with the 128 filters
        up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1) 

        # convolutional block which has the same shape as conv2
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

        # upsample the conv4 and and combine it with conv1
        # conv 5 has the same dimensions as the input image 
        up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

        # now this has a shape of image_height x image_width x 50 
        # This makes sense each pixel is a vector of length 50 
        # we take the softmax of that pixel to get the prediction
        out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)

        model = get_segmentation_model(img_input ,  out )
        return model


if __name__=='__main__':
    model=CustomSegmentationModel.build(224,224,3,50)
    print(model.summary())


        
