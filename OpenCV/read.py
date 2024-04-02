import cv2 as cv

img = cv.imread('Photos/Car.jpg')
cv.imshow('Cat', img)

cv.waitKey(0)