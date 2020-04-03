# import model averaging class
from combine_models import CombineModels
from sklearn.datasets import make_blobs
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt
import numpy as np 

cm= CombineModels('averaging_study','*.hdf5')

num_models = len(cm.models)

weights = [(1)/float(num_models) for i in range(0,num_models)]

model=cm.model_weight_ensemble(weights)
print(model.summary())

# load the data 

X,y = make_blobs(n_samples=1100, centers=3,n_features=2, cluster_std=2,random_state=2)

for class_value in range(3):
    # select the indices of points with the class label
    row_ix=np.where(y==class_value)
    # scatter plot the points with a different color
    plt.scatter(X[row_ix,0],X[row_ix,1])

# show plot
plt.title("Sklearn Data with two features")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.pause(3.0)

# convert the labels to one hot vectors
y = to_categorical(y)

# split into train and test
n_train =100 
trainX, trainY= X[:n_train], y[:n_train]
testX, testY= X[n_train:],y[n_train:]

_, train_acc = model.evaluate(trainX,trainY)
_, test_acc = model.evaluate(testX,testY)

print(train_acc,test_acc)

for j in range(1,11):
    model=cm.evaluate_n_members(j)
    _, train_acc = model.evaluate(trainX,trainY)
    _, test_acc = model.evaluate(testX,testY)
    print(train_acc,test_acc)


# let's try Linearly and exponentially decreasing weighted average

cm= CombineModels('averaging_study','*.hdf5')

num_models = len(cm.models)
# linearly decreasing weighting 
weights = [(i)/float(num_models) for i in range(num_models,0,-1)]
print(weights)
model=cm.model_weight_ensemble(weights)

# Evaluate the model
_, train_acc = model.evaluate(trainX,trainY)
_, test_acc = model.evaluate(testX,testY)

print("Linear Decreasing:",train_acc,test_acc)

# exponentially weighted average
m= CombineModels('averaging_study','*.hdf5')

num_models = len(cm.models)
# linearly decreasing weighting 
weights = [np.exp(-i/2.0) for i in range(1,num_models+1)]
model=cm.model_weight_ensemble(weights)
print(weights)
# Evaluate the model
_, train_acc = model.evaluate(trainX,trainY)
_, test_acc = model.evaluate(testX,testY)

print("Exponentially Decreasing:",train_acc,test_acc)
