from matplotlib import pyplot as plt 
import argparse
import cv2


ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args['image'])

#Load the image and show it
cv2.imshow("Original",image)
cv2.waitKey(0)

#split the image into channels
chans=cv2.split(image)

#CV2 stores the numpy arrays in BGR order
colors=("b","g","r")


plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")



#compute the histogram for each image
for (chan,color) in zip(chans,colors):
    #compute the histogram
    #The order: images, channels, mask, histSize, ranges
    hist=cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color=color)
    plt.xlim([0,256])
plt.show()


#Multidimensional histograms
fig=plt.figure()

#add subplots
ax=fig.add_subplot(131)
hist=cv2.calcHist([chans[1],chans[0]],[0,1],None,[32,32],[0,256,0,256])
p=ax.imshow(hist,interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

ax=fig.add_subplot(132)
hist=cv2.calcHist([chans[1],chans[2]],[0,1],None,[32,32],[0,256,0,256])
p=ax.imshow(hist,interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

ax=fig.add_subplot(133)
hist=cv2.calcHist([chans[0],chans[2]],[0,1],None,[32,32],[0,256,0,256])
p=ax.imshow(hist,interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)
plt.show()

print("2D histogram shape: {}, with {} values".format(hist.shape,hist.flatten().shape[0]))


#you can calculate 3D histograms
hist=cv2.calcHist([image],[0,1,2],None [8,8,8],[0,256,0,256,0,256])
print("3D histogram shape: {}, with {} values".format(hist.shape,hist.flatten().shape[0]))


