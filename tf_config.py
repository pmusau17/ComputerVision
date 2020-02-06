#This file is for the specific purpose of setting GPU configs for your local machine
#credit to: https://www.tensorflow.org/guide/gpu
import argparse
import tensorflow as tf


#arguments for getting the fraction
ap=argparse.ArgumentParser()
ap.add_argument("-f","--fraction",required=True, help="maximum fraction you want a process to be able to use on the GPU")
args=vars(ap.parse_args())

#this prints the number of physical devices
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#this prints the devices on my current machine
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = float(args['fraction'])
#InteractiveSession(config=config)