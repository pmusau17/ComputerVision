# import the neccessary packages 
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import AveragePooling2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D

from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.models import Model # instead of importing Sequential we import model
# This will allow us to create a network graph with splits and forks like the Inception Module
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras import backend as K  

class MiniGoogLeNet:
   
    """ 
        x: input to the conv module
        K: number of filters in conv layer
        (kX,KY): size of filters
        strid: the stride for the conv layer
        padding: type padding for this layer
        chanDim: The channel dimension, which is derived from either "channels last", or 
        "channels first"
    """
    @staticmethod
    def conv_module(x,K,kX,kY,stride,chanDim,padding='same'):

        #define a CONV => BN => RELU pattern

        x = Conv2D(K,(kX,kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        # return the current block
        return x


    """ We are allowed to concatenate the output of these two output branches
        Because the output size for both convolutions is identical due to padding = 'same' """
    @staticmethod
    def inception_module(x,numK1x1,numK3x3,chanDim):
        # define two CONV modules, then concatenate across the 
        # channel dimension

        conv_1x1 = MiniGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x,numK3x3,3,3,(1,1),chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis=chanDim)

        return x 

    @staticmethod 
    def downsample_module(x,K,chanDim):
        # define the CONV module and POOL, then concatenate
        # across channel dimensions

        conv_3x3 = MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding='valid')
        pool = MaxPooling2D(3,3,strides=(2,2))(x)

        x = concatenate([conv_3x3,pool],axis=chanDim)

        # return the block 
        return x 

    @staticmethod
    def build(width,height,depth, classes):
        #initialize the input shape to be "channels last

        inputShape = (height,width,depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension 

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height, width)
            chanDim = 1

        # Define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs,96,3,3,(1,1),chanDim)

        # Two Inception Modules

        x = MiniGoogLeNet.inception_module(x,32,32,chanDim)
        x = MiniGoogLeNet.inception_module(x,32,48,chanDim)
        x = MiniGoogLeNet.downsample_module(x,80,chanDim)

        # Four Inception Modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x,112,48,chanDim)
        x = MiniGoogLeNet.inception_module(x,96,64,chanDim)
        x = MiniGoogLeNet.inception_module(x,80,80,chanDim)
        x = MiniGoogLeNet.inception_module(x,48,96,chanDim)
        x = MiniGoogLeNet.downsample_module(x,96,chanDim)

        # Two inception modules followed by a global POOL and dropout
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        # create the model 
        model = Model (inputs,x,name="googlenet")

        return model

        


