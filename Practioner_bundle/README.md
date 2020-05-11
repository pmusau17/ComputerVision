# Adventures in the Practioner Bundle

# Utils
loading ...
# Callbacks 
loading ...

# HDF5 Datawriter 

First, what is HDF5? HDF5 is a binary dat format created by the HDF5 group to store very large datasets on disk that are too large to be stored in memory. With HDF5 you can store huge datasets and manipulate multi-terabyte datasets stored on disk as if they were simply Numpy arrays loaded into memory.

# Creating and HDF5 Dataset 

The following [file](input_output/hdf5datasetwriter.py) is responsible for creting HDF5 datasets. It accepts four inputs two of which are optional:
- **dims**: controls the dimension, or the shape of the data that will be stored in the dataset. As an example, if we have 60,000 images of shape 28 x 28 x 1. Then dims would be (6000,28,28,1) 
- **outputPath** (required): Where to store the hdf5 dataset.
- **datakey** (optional, default="images"): what to call the data you are storing in the hdf5 dataset.
- **bufferSize**: (optional, default=1000): How many images to keep in memory at one time before writng them to file

Key functions: 
- add(row_of_images,row_of_labels): adds a row of images and labels into the hdf5 dataset
- close(): closes the hdf5 writer and saves remaining images to file 
- storeClassLabels(labels): stores the string version of the labels to file