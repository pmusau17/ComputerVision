#ROS and OpenCV don't play nice in python3
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


#import the neccessary packages
from skimage.exposure import rescale_intensity
import numpy as np 
import argparse
import cv2



#The convolve function requires two parameters: the grayscale image that we want to convolve with the kernel. Given our image and the kernel
#We then determine the spatial dimensions (width and height) of each
def convolve(image,K):
    #grab the spatial dimensions of the image and kernel
    #so imagine an image that is 64x64x3 so we are grabbing the 64 x 64

    (iH,iW)=image.shape[:2]
    (kH,kW)=K.shape[:2]

    #allocate memory for the output image. Taking care to "pad" the borders
    #of the input image so that the spatial size is not reduced. (i.e., width)
    #height are not reduced 

    pad=(kW-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output=np.zeros((iH,iW),dtype='float')


    for y in np.arange(pad,iH+pad):
        for x in np.arange(pad,iW+pad):
            #extract the ROI of the image by extracting the 
            #*center* region of the current (x,y) -coordinates
            #dimensions

            roi=image[y-pad:y+pad+1,x-pad:x+pad+1]

            #perform the actual convolution by taking the element-wise multiplication between the ROI
            #and the kernel, then summing the matrix
            k=(roi*K).sum()

            #store the convolved value in the output (x,y) coordinate of the 
            #output image
            output[y-pad,x-pad]=k

            #rescale the output imate to be in the range [0,255]
            #when working with images we typically deal with pixel values falling in the range [0,255]
            #However when applying convolutions we can easily obtain values that fall outside this range
            #In order to bring our output images back into the range we apply the rescale_intensity function
            #of scikit image
    output=rescale_intensity(output,in_range=(0,255))

    #we also convert our image back into an unsigned 8 bit integer.
    output=(output*255).astype('uint8')
    #return output image
    return output

# construct the argument parser and parse the arguments
# we only require the path to our input image
ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help='path to the input image')
args=vars(ap.parse_args())

#construct average blurring kernesl to smooth an image
smallBlur=np.ones((7,7), dtype='float')*(1.0/(7*7))
largeBlur=np.ones((21,21), dtype='float')*(1.0/(21*21))

#We then have kernel responsible for sharpening an image

sharpen=np.array(([0,-1,0],[-1,5,-1],[0,-1,0]), dtype="int")

#construct the Laplacian kernel used to detect edge like regions of an image
Laplacian=np.array(([0,1,0],[1,-4,1],[0,1,0]), dtype="int")

#The sobel kernels can be used to detect edge like regions along the x and y axis, respectively

sobelX=np.array(([-1,0,1],[-2,0,2],[-1,0,1]), dtype="int")
sobelY=np.array(([-1,-2,-1],[0,0,0],[1,2,1]), dtype="int")

#the emboss kernel
emboss=np.array(([-2,-1,0],[-1,1,1],[0,1,2]), dtype="int")

#Each of these kernesl were manually built to perform a given operation

#construct the kernel bank, a list of kernels we're going to apply 
#using both our custom 'convolve' function and OpenCV's filter2D
#function 

kernelBank=(
    ('small_blur',smallBlur),
    ('large_blur',largeBlur),
    ('sharpen',sharpen),
    ('laplacian',Laplacian),
    ('sobel_x',sobelX),
    ('sobel_y',sobelY),
    ('emboss',emboss))


#load the input image and convert it to grayscale
image=cv2.imread(args['image'])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#loop over the kernels
for (kernelName,K) in kernelBank:
    #apply the kernel to the grayscale image using both our custom 
    #'convolve' function and OpenCV's 'filter2D' function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput=convolve(gray,K)
    opencvOutput=cv2.filter2D(gray,-1,K)

    #show the output images
    cv2.imshow("Original",gray)
    cv2.imshow("{} - convolve".format(kernelName),convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName),opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#As a sanity check we also call cv2.filter2D which also applies our kernel to gray. Its much fater and optimized than ours 

