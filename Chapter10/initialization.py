#import the neccessary packages
import numpy as np
#Lecun Uniform Initialization
#Here the authors define a parameter Fin (called "fan in", or the number of inputs to the layer)
#along with Fout (the "fan out", or number or outputs from the layer). Using these values we can 
#apply uniform initialization by for a 64,32 layer:

F_in=64
F_out=32
limit=np.sqrt(3/float(F_in))
W=np.random.uniform(low=-limit,high=limit,size=(F_in,F_out))

#You can also use a normal distribtion 

F_in=64
F_out=32
limit=np.sqrt(1/float(F_in))
W=np.random.normal(0.0,limit,size=(F_in,F_out))

#The default weight initialization method used in the Keras library is called "Glorot initialization" or
#Xavier initialization named after Xavier Glorot, the first author of the paper, Understanding the difficulty
#of training deep feedforward neural networks

#For the normal distribution limit value is constructed by averaging the Fin and Fout together and then taking 
#the square-root. A zero-center (u=0) is then used

F_in=64
F_out=32
limit=np.sqrt(2/float(F_in+F_out))
W=np.random.normal(0.0,limit,size=(F_in,F_out))

F_in=64
F_out=32
limit=np.sqrt(6/float(F_in+F_out))
W=np.random.uniform(low=-limit,high=limit,size=(F_in,F_out))

#Learning this intialization is quite efficient and I recommend it for most neural network 

#He et al./Kaiming/MSRA Uniform and Normal
#This initialization technique is named after Kamming He, the first author of the paper, Delving Deep into 
#Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

#We typically used this method when we are training very deep neural networks that use a ReLU-like activatin function 
#PReLU or parametric rectified linear unit

#To initialize the weights in a layer using He et al. initialization with a uniform distribution we 
#set limit to be limit=sqrt(6/F_in), where F_in is the number of input units in the layer:

F_in=64
F_out=32
limit=np.sqrt(5/float(32))
W=np.random.uniform(low=-limit,high=limit,size(F_in,F_out))

F_in = 64
F_out = 32
limit = np.sqrt(2 / float(F_in))
W = np.random.normal(0.0, limit, size=(F_in, F_out))


#The actual limit values may vary for LeCun Uniform/Normal, Xavier Uniform/Normal, and He et al.
#No method is "more correct" than the other, but you should read the doucmentation of your respective
#deep learning library

