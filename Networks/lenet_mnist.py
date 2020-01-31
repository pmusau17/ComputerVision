#import the necessary packages
from nn.conv.LeNet import LeNet
#import the correct SGD optimizer, the other one gives errors smh
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets 
from tensorflow.python.keras import backend as K 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#path to where we would like to save the network after training is complete
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


#The MNIST dataset has already been preprocessed
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data

#in this data set each image is represented by a 784-d vector so we need to reshape it
if K.image_data_format()=="channels_first":
    data=data.reshape(data.shape[0],1,28,28)

#Otherwise, we are using "channels last" ordering     
else:
    data=data.reshape(data.shape[0],28,28,1)

#scale the input data to the range [0,1] and performa a train/test split

(trainX,testX,trainY,testY)=train_test_split(data/255.0, dataset.target.astype("int"),test_size=0.25,random_state=42)

#convert the lavels from integers to one hot vectors, rather than single 
#integer values
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

#initialize the optimizer and model
print("[INFO] compiling model...")
opt=SGD(lr=0.01)
num_epochs=20
model=LeNet.build(height=28, width=28, depth=1,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#train the network
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=128, epochs=num_epochs,verbose=1)
model.save(args["model"])
#evaluate the network
print("[INFO] evaluating network...")
predictions=model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()