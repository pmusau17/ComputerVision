# import the necessary packages
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import Activation 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization

class Cifar_FNN:
    @staticmethod
    def build(input_shape,classes,output_activation='softmax'):
        #initialize the model along with the input shape to be 
        #channel's last and the channels dimension itself
        model = Sequential()

        # define the feed-forward neural network architechture
        model.add(Dense(256,input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        #model.add(BatchNormalization())
        model.add(Dense(512,input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(256,input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(classes))
        model.add(Activation(output_activation))

        #now that we've implemented mini VGGNet Architecthure
        return model