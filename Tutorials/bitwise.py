import numpy as np 
import cv2 


#create the rectangle
rectangle=np.zeros((300,300), dtype='uint8')
cv2.rectangle(rectangle,(25,25),(275,275),255,-1)
cv2.imshow("Rectangle",rectangle)

#create the circle
circle=np.zeros((300,300),dtype="uint8")
cv2.circle(circle,(150,150),150,255,-1)
cv2.imshow("Circle",circle)

test=cv2.imread('/home/musaup/Documents/Research/ComputerVision/Tutorials/data/pic00013.jpg')
mask=np.zeros(test.shape[:2],dtype='uint8')
cv2.circle(mask,(test.shape[0]//2,test.shape[1]//2),100,255,-1)
cv2.imshow("test",test)
cv2.imshow("mask",mask)
masked=cv2.bitwise_and(test,test,mask=mask)
cv2.imshow('masked image',masked)


#do the bitwise operations 
bitwiseAnd=cv2.bitwise_and(rectangle,circle)
cv2.imshow("AND",bitwiseAnd)

bitwiseOr=cv2.bitwise_or(rectangle,circle)
cv2.imshow("OR",bitwiseOr)

bitwiseXOR=cv2.bitwise_xor(rectangle,circle)
cv2.imshow("XOR",bitwiseXOR)

bitwiseNot=cv2.bitwise_not(circle)
cv2.imshow("NOT",bitwiseNot)
cv2.waitKey(0)