# import the neccessary packages
from config import lisa_config as config
from utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf 
import os 


def main(_): # at this point i'm assuming that _ means void (don't know though)
    # open the classes output file 
    f = open(config.CLASSES_FILE,'w')

    # loop over the classes
    # The TFOD API expects this file in a certain a JSON/YAML-like format
    # Each class must have an item with two attributes: id and name. The name is the 
    # human readable name for the class label while the id is a unique integer for the class
    # label 

    # Keep in mind thatthe class IDs shoud start counting from 1 rather than 0 as zero is
    # reserved for the background class 
    for (k,v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k +"'\n"
                "}\n")

        f.write(item)

    # close the output classes file 
    f.close()

    # initialize a data dictionary ued to map each data dictionary used to map each image filename 
    # to all bounding boxes associated with the image, then load the contents of the annotations 
    # file 

    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header 
    for row in rows[1:]:
        # break the row into components 
        row = row.split(",")[0].split(";")
        (imagePath,label,startX,startY,endX,endY,_) = row
        (startX,startY) = (float(startX),float(startY)) 
        (endX, endY) = (float(endX),float(endY)) 


        # if we are not interest in the label, ignore it 
        if label not in config.CLASSES:
            continue

        # build the path to the input image, then grab any other 
        # bounding boxes + labels associated with the image path, labels
        # associated with the image path, labels, and bounding box lists
        # respecitively

        p = os.path.sep.join([config.BASE_PATH,imagePath])
        b = D.get(p,[])

        # build a tuple consisting of the label and bounding box
        # then  update the list and store it in the dictionary

        b.append((label,(startX,startY,endX,endY)))

        D[p] = b 
    
    # create training and testing splits from our data dictionary 

    (trainKeys,testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE,random_state=42)

    # initialize the data split files

    datasets = [
        ("train",trainKeys,config.TRAIN_RECORD),
        ("test",testKeys,config.TEST_RECORD),
    ]

    # loop over the datasets 
    for (dtype,keys,outputPath) in datasets:
        # Initialize the Tensorflow writer and intialize the total 
        # number of examples written to file 

        print("[INFO] processing '{}'...".format(dtype))
        writer = tf.python_io.TRecordWriter(outputPath)

        total = 0

        # loop over all the keys in the current set 
        for k in keys:
            # load the input image from disk as a Tensorflow object 
            encoded = tf.gfile.GFile(k,"rb").read()
            encoded  = bytes(encoded)

            # load the image from disk again, this time as a PIL
            # object

            pilImage = Image.open(k)
            (w,h) = pilImage.size[:2]

            # parse the filename and encoding from the input path 
            filename = k.split(os.path)[-1]
            encoding = filename[filename.rfind(".")+1]

            # initialize the annotation object used to store 
            # information regarding the bounding box + labels 

            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            ffAnnot.width = w
            tfAnnot.height = h 

            for (label,(startX,startY,endX,endY)) in D[k]:
                # Tensorflow assumes all bounding boxes are in 
                # range [0,1] so we need to scale them 


                xMin = startX / w
                xMax = endX / w 
                yMin = startY / h
                yMax = endY / h

                # update the bounding boxes + labels lists 

                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)

                tfAnnot.textLabels.append(label.encode('utf8'))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                # increment the total number of examples 
                total += 1
            
            # encode the data point attributes using the Tensorflow
            # helper functions 

            features = tf.train.Features(feature=tfAnnot.build()) 
            example = tf.train.Example(features=features)

            # add the example to the writer

            writer.write(example.SerializeToString())


    # close the writer and print diagnostic information to the user 
    writer.close()
    print("[INFO] {} examples save for '{}'".format(total,dtype))

    # check to see if the main thread should be started




