import cv2
import numpy as np

def detect_dominant_color(image_path):
    
    image = cv2.imread('Photos/Map.jpg')


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    pixels = image.reshape((-1, 3))


    pixels = np.float32(pixels)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  
    centers = np.uint8(centers)


    dominant_color = tuple(centers[0])

    return dominant_color

image_path = 'Photos/Map.jpg'


dominant_color = detect_dominant_color('Photos/Map.jpg')
print("Dominant color (RGB):", dominant_color)