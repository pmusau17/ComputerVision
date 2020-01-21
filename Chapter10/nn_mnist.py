#import the neccessary packages
from neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)

print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")

#perform min/max normalizing by scaling each digit into the range [0, 1]
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))


# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,digits.target, test_size=0.25)

# convert the labels from integers to vectors one hot encoding
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))

# In practice, backpropagation can be not only challenging to implement (due to bugs in com-
# puting the gradient), but also hard to make efficient without special optimization libraries, which
# is why we often use libraries such as Keras, TensorFlow, and mxnet that have already (correctly)
# implemented backpropagation using optimized strategies.