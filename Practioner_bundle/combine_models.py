# import the neccessary packages
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import clone_model

# python glob package is a module that finds all the pathnames matching a 
# specified pattern according to the rules used in the Unix shell, the results are returned in arbitary order.
import glob
import os 
import numpy as np 


"""This class will combine and average a list of models
Inspired by Jason Brownlee
"""

class CombineModels:

    def __init__(self,model_path,search_pattern_extension):
        # this will produce something like this path_name/*.models
        modelPaths = os.path.sep.join([model_path,search_pattern_extension]) 
        self.modelPaths = list(glob.glob(modelPaths))
        self.models = []

        # load the models 
        self.load_models()

    # load all the models using the directory and pathname seach pattern
    def load_models(self):
        for (i,modelPath) in enumerate(self.modelPaths):
            print("[INFO] loading model {}/{}".format(i+1,len(self.modelPaths)))
            self.models.append(load_model(modelPath))

    # average the weights 
    def model_weight_ensemble(self,weights):
        # determine how many layers need to be averaged
        n_layers = len(self.models[0].get_weights())
        #print(weights)

        # create an average model weights
        avg_model_weights = list()

        for layer in range(n_layers):
            # collect this layer from each model
            layer_weights = np.array([model.get_weights()[layer] for model in self.models])
            # weighted average of weights for this layer
            avg_layer_weights=np.average(layer_weights,axis=0,weights=weights)
            # append the weights to the list
            avg_model_weights.append(avg_layer_weights)

        # create a new model with the same structure
        model = self.models[0]
        wb=model.get_weights()[0]
        model.set_weights(avg_model_weights)
        #print(wb-model.get_weights()[0])
        return model
