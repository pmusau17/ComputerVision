#import the neccessary packages
from LeNet import LeNet
from tensorflow.python.keras.utils import plot_model

#Initialize LeNet and then write the network architechture
#visualization graph to disk
model=LeNet.build(28,28,1,10)

#This is responsible for constructing a graph based on the layers inside 
#The input model and then writing the graph to disk an image
plot_model(model, to_file="lenet.png",show_shapes=True)