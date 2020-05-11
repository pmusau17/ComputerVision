import h5py
import config.tiny_imagenet_config as config 

filenames = [config.TRAIN_HDF5,config.TEST_HDF5,config.VAL_HDF5]

for filename in filenames:
    db = h5py.File(filename,"r")
    print(db['images'].shape)
    db.close()

