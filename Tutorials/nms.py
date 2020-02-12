#import numpy
import numpy as np 


#Felzenswalb et al. Method of non-max suppression:
#common values for the threshold are between 0.3 and 0.5

def non_max_suppression_slow(boxes,overlapThresh):
    #If there are no boxes return an empty list
    if len(boxes)==0:
        return []

    #initialize the list of picked indices
    pick=[]
    #grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #compute the area of the bounding boces and sort the bounding boxes by the bottom-right
    #y coordinate of the bounding box
    area=(x2-x1+1)*(y2-y1+1)
    #what argsort returns is a list of array indices that sort a along the specified axis
    #it will return the sorted array regardless of dimensionality. So here we return the indices 
    #that will sort the bounding boxes by the bottom right coordinate. For their method we have
    #to sort along this coordinate
    idxs=np.argsort(y2)

    #keep looping until there are no more indexes to examine
    while len(idxs)>0:
        #grab the last index in the indexes list, add the index value
        #to the list of picked indexes, then initialize the suprresion list (i.e. indexes that will be deleted)
        #using the last index
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        suppress=[last]
        #loop over all indexes in the indexes list
        for pos in range(0,last):
            #grab the current index
            j=idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # # the bounding box and the smallest (x, y) coordinates
            # # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            # of the considerd image j
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        #print(pick)
        idxs = np.delete(idxs, suppress)
        # return only the bounding boxes that were picked
    return boxes[pick]


#This version of non-max suppression is over 100 times faster

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    #initialize the list of picked indices
    pick=[]
    #grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #compute the area of the bounding boces and sort the bounding boxes by the bottom-right
    #y coordinate of the bounding box
    area=(x2-x1+1)*(y2-y1+1)
    #what argsort returns is a list of array indices that sort a along the specified axis
    #it will return the sorted array regardless of dimensionality. So here we return the indices 
    #that will sort the bounding boxes by the bottom right coordinate. For their method we have
    #to sort along this coordinate
    idxs=np.argsort(y2)

    #keep looping until there are no more indexes to examine
    while len(idxs)>0:
        #grab the last index in the indexes list, add the index value
        #to the list of picked indexes, then initialize the suprresion list (i.e. indexes that will be deleted)
        #using the last index
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        print(xx1)
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

    
    #return only the bounding boxes that were picked using and convert back to integers
    return boxes[pick].astype("int")