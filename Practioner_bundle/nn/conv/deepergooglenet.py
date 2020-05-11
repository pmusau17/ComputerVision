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
        x = Conv2D(K,(kX,kY),strides=stride,padding=padding,kernel_regularizer=l2(reg),name=convName)
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

     