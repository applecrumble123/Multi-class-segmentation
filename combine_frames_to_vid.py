import cv2
import numpy as np
import os

#folder_path = "/home/johnathon/Desktop/rainy_image_100mm"
folder_path = "/home/johnathon/Desktop/output/day_50m/images/rain/100mm/rainy_image"
 
img_array = []
for i in range(len(os.listdir(folder_path))):
    filename = os.path.join(folder_path, str(i) + '.png')
    #print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
#print(img_array)
 

# 10 is the number of frames per second
# So, 10 images shall be used to create a video of duration one second.
# size --> frame size
"""
out = cv2.VideoWriter('/home/johnathon/Desktop/dark_10m.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
"""
out = cv2.VideoWriter('/home/johnathon/Desktop/day_50m.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
