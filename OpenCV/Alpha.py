import cv2
import numpy as np

# Load satellite image
image = cv2.imread('Photos/Map2.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to segment colors (assuming white parking lines)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Morphological operations
kernel = np.ones((5,5),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate space for parking a car
total_space = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Filter out small contours (adjust threshold as needed)
        total_space += area

# Output the total parking space available
print("Total space available for parking (in pixels):", total_space)

# Visualize detected contours
cv2.drawContours(image, contours, -1, (0,255,0), 2)
cv2.imshow('Parking Spaces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
