import cv2
import numpy as np

image = cv2.imread('Photos/MapF1.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


total_space = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  
        total_space += area


print("Total space available for parking (in pixels):", total_space)

object_length_pixels = total_space
scale = 1/30
distance_in_units = object_length_pixels * scale
print("Distance in pixels:", object_length_pixels)
print("Distance in meter:", distance_in_units)

cv2.drawContours(image, contours, -1, (0,255,0), 2)
cv2.imshow('Parking Spaces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
