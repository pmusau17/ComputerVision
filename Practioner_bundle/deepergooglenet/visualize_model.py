# import sys so we can use packages outside of this folder in
# either python 2 or python 3,
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

from nn.conv.deepergooglenet import DeeperGoogLeNet
from tensorflow.keras.utils import plot_model

model = DeeperGoogLeNet.build(64,64,3,200)
plot_model(model,to_file='googlenet_tiny_imagenet.png',show_shapes=True,show_layer_names=True)