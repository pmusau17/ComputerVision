#import the necessary packages
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten    
from tensorflow.python.keras.layers.core import Dense
from keras import backend as K


#We define the build method for oure network which requires four params, width, height, number of channels, class labels
#openCv gives images by height width so that's we will do here
class LeNet:
    @staticmethod
    def build(height, width, depth, classes):
        # initialize the model
        model = Sequential()
        #the shape of our image inputs
        inputShape=(height, width, depth)

        #if we are using "channels first" update the input shape
        if (K.image_data_format()=="channels_first"):
            inputShape=(depth, height, width)

        #first set of CONV=>RELU=> POOL layers
        model.add ( Conv2D(20,(5,5), padding="same",input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
        
        #second set of CONV=>RELU => POOL layers
        #interesting enough here you don't have to specify the size of the previous layer
        model.add(Conv2D(50,(5,5),padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #first (and only) set of FC => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        #return model
        return model

                                        

        
        