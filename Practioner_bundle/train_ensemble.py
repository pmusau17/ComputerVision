# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the neccessary packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
