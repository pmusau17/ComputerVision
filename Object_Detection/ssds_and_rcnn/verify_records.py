#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from imutils import paths
# import the neccessary packages
from config import lisa_config as config
import tensorflow as tf
tf.enable_eager_execution()
import cv2


"""The first law of building record files: though shalt first verify that your code 
was indeed correct. In God we trust, all others must bring evidence"""

training_record = tf.data.TFRecordDataset(config.TRAIN_RECORD)


# features used in building dataset
data = {
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/filename": tf.io.FixedLenFeature([], tf.string),
            "image/source_id": tf.io.FixedLenFeature([], tf.string),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/class/label": tf.VarLenFeature(tf.int64),
            "image/object/difficult": tf.io.VarLenFeature(tf.int64),
        }

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, data)


paths = list(paths.list_images('lisa'))



parsed_image_dataset = training_record.map(_parse_image_function)

for item in parsed_image_dataset:
    image_raw = item['image/encoded']
    image_filename = item['image/filename'].numpy()
    image_height =  item["image/height"].numpy()
    image_width = item["image/width"].numpy()

    image_xmins = item['image/object/bbox/xmin'].values.numpy()
    image_xmaxs = item['image/object/bbox/xmax'].values.numpy()
    image_ymins = item['image/object/bbox/ymin'].values.numpy()
    image_ymaxs = item['image/object/bbox/ymax'].values.numpy()


    image_filename=image_filename.decode('utf8')
    path = [name for name in paths if image_filename in name][0]


    w = image_width
    h = image_height
    image = cv2.imread(path)
    for i in range(len(image_xmaxs)):

        xMin = image_xmins[i]
        xMax = image_xmaxs[i]
        yMin  = image_ymins[i]
        yMax = image_ymaxs[i]

       
        # load the input image from disk and denormalize the
        # bounding box coordinates
        
        
        startX = int(xMin * w)
        startY = int(yMin * h)
        endX = int(xMax * w)
        endY = int(yMax * h)


        cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)