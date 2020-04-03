# import the neccessary packages
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD 
import matplotlib.pyplot as plt
import numpy as np

# generate 2d classification dataset

X,y = make_blobs(n_samples=1100, centers=3,n_features=2, cluster_std=2,random_state=2)

# convert the labels to one hot vectors
y = to_categorical(y)

# split into train and test
n_train =100 
trainX, trainY= X[:n_train], y[:n_train]
testX, testY= X[n_train:],y[n_train:]

# plot the data so we can see what it looks like
plt.style.use('ggplot')
plt.figure()

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

# define our model 

model = Sequential()
model.add(Dense(25,input_dim=2))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

# compile the model
opt=SGD(lr=0.01,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

n_epochs = 500
n_save = 490 
# train for 490 save last 10 models
for i in range(n_epochs):
    # fit model for a single epoch
    model.fit(trainX,trainY, validation_data=(testX,testY), epochs=1,verbose=1)
    if i>=n_save:
        model.save('averaging_study/model_{}.hdf5'.format(i))

_, train_acc = model.evaluate(trainX,trainY)
_, test_acc = model.evaluate(testX,testY)

print(train_acc,test_acc)


