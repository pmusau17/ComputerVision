#import the necessary packages
from keras import Sequential
from keras.layers.convolutional import Conv2D #this is the Keras implementation of the convolutional layer we have discussed up to this point

#The activation package handles applying an activation to an input
#The Flatten classes takes our multi-dimensional volume and “flattens” it into a 1D array prior to feeding the inputs into the Dense (i.e, fully-connected) layers
from keras.layers.core import Activation, Flatten, Dense 
from keras.backend import backend as K

#implement your neural networks in a class to keep your code organized

class ShallowNet:


    #width: The width of the input images that will be used to train the network
    #height: The height of our input images
    #depth: The number of channels in the input image
    #classes: The total number of classes that our network should learn to predict

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"

        model = Sequential()
        #the image input
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #Every CNN that you implement will have a build method – this function will accept a
        #number of parameters, construct the network architecture, and then return it to the calling function
        #It will accept a number of parameters
        



