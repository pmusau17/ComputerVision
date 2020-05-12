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
from tensorflow.python.keras.models  import Model 
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.regularizers import l2 
from tensorflow.python.keras import backend as K 

""" Notice above that we are using Model instead of sequential so 
    we can contruct a graph and not a sequential network, the other thing 
    You will notice is that we define modules so that the code doesn't become 
    bloated"""

class DeeperGoogLeNet:
    
    """
    x: input to the block
    K: The number of filters the convolutional layer will learn 
    kX and kY: The filter size for the convolutional layer 
    stride: The stride (in pixels) for the convolution (typically we use 1x1  but if you want to reduce )
            the output volume you can use a large stride
    chanDim: This value controls the dimension (i.e) axis of the iage channel. It is automatically set later in this 
             class based on the Keras Backend
    padding: control the padding of the convolutional layer
    reg: The L2 weight decay strength
    name: Since this network is large it's a good idea to name the layers to help debuh, share/explain the network to others  

    """
    @staticmethod
    def conv_module(x,K,kX,kY,stride,chanDim,padding="same",reg=0.005,name=None):

        # initialize the CONV,BN, aand RELU layer names 
        (convName,bnName,actName) =(None,None,None)

        # if a layer name was supplied, prepend it

        if name is not None:
            convName = name + "_conv"
            bnName   = name + "_bn"
            actName  = name + "_act"

        # define a CONV => BN => RELU pattern
        x = Conv2D(K,(kX,kY),strides=stride,padding=padding,kernel_regularizer=l2(reg),name=convName)(x)
        x = BatchNormalization(axis=chanDim,name=bnName)(x)
        x = Activation("relu",name=actName)(x)

        # return the block
        return x 


    @staticmethod
    def inception_module(x,num1x1,num3x3Reduce,num3x3,num5x5reduce,num5x5,
                        num1x1Proj,chanDim,stage,reg=0.0005):
        
        # define the first branch of the Inception Module which consists of 1x1
        # convolutions

        first = DeeperGoogLeNet.conv_module(x,num1x1,1,1,(1,1),chanDim,reg=reg,name=stage+"_first")


        # define the second branch of the Inception Module which consists of 1x1 and 3x3 convolutions 

        second = DeeperGoogLeNet.conv_module(x,num3x3Reduce,1,1,(1,1),chanDim,reg=reg,name=stage+"_second1")
        second = DeeperGoogLeNet.conv_module(second,num3x3,3,3,(1,1),chanDim,reg=reg,name=stage+"_second2")

        # define the third branch of the inception module which are our 1x1 and 5x5 convolutions 
        third = DeeperGoogLeNet.conv_module(x,num5x5reduce,1,1,(1,1),chanDim,reg=reg,name=stage+"_third1")
        third = DeeperGoogLeNet.conv_module(third,num5x5,5,5,(1,1),chanDim,reg=reg,name=stage+"_third2")

        # define the fourth and final branch of the Inception Module is called the POOL projection 

        fourth = MaxPooling2D((3,3),strides=(1,1),padding="same",name=stage+"_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(fourth,num1x1Proj,1,1,(1,1),chanDim,reg=reg,name=stage+"_fourth")

        # concatenate across the channel dimension 
        x = concatenate([first,second,third,fourth],axis=chanDim,name=stage + "_mixed")

        return x 


    @staticmethod
    def build(width,height,depth,classes,reg=0.005):

        # initialise the input shape to be "channels last and the channel dimension itself"

        inputShape = (height,width,depth)
        chanDim = -1

        # switch if the backend is channels first

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        # define the model input, followed by a sequence of a CONV => 
        # POOL => (CONV * 2) => POOL layers

        print(inputShape)
        inputs = Input(shape=inputShape)

        # 64 filters of shape 5x5
        x = DeeperGoogLeNet.conv_module(inputs,64,5,5,(1,1),chanDim,reg=reg,name="block1")
        # Maxpooling reduce the spatial dimensions by half using 3x3 filter
        x = MaxPooling2D((3,3),strides=(2,2),padding='same',name='pool1')(x)
        
        # 64 filters of shape 1x1
        x = DeeperGoogLeNet.conv_module(x,64,1,1,(1,1),chanDim,reg=reg,name='block2')

        # 192 filters of shape 3x3
        x = DeeperGoogLeNet.conv_module(x,192,3,3,(1,1),chanDim,reg=reg,name='block3')

        # further reduce the spatial dimensions
        x = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool2")(x)


        # apply Two Inception modules followed by POOL

        x = DeeperGoogLeNet.inception_module(x,64,96,128,16,32,32,chanDim,'3a',reg=reg)

        x = DeeperGoogLeNet.inception_module(x,128,128,192,32,96,64,chanDim,'3b',reg=reg)

        x = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool3")(x)

        # apply five Inception modules followed by POOl

        x = DeeperGoogLeNet.inception_module(x,192,96,208,16,48,64,chanDim,'4a',reg=reg)
        
        x = DeeperGoogLeNet.inception_module(x,160,112,224,24,64,64,chanDim,'4b',reg=reg)

        x = DeeperGoogLeNet.inception_module(x,128,128,256,24,64,64,chanDim,'4c',reg=reg)

        x = DeeperGoogLeNet.inception_module(x,112,144,288,32,64,64,chanDim,'4d',reg=reg)

        x = DeeperGoogLeNet.inception_module(x,256,160,320,32,128,128,chanDim,'4e',reg=reg)

        x = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool4")(x)

        # At this point our output volume size is 4x4 classes. To avoid the usage of computationally
        # expensive fully-connected layers (not mention, dramatically increased network size), we apply 
        # average pooling with a 4x4 kernel to reduce the volume size of 1 x 1 x classes:

        x = AveragePooling2D((4,4),name="pool5")(x)
        x = Dropout(0.4,name="do")(x)

        # softmax classifier 
        x = Flatten(name = "flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg),name="labels")(x)
        x = Activation("softmax",name="softmax")(x)

        # create the model 
        model = Model(inputs,x,name="googlenet")

        # return the constructed network architechture
        return model 



     