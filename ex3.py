import cv2 as cv
import numpy as np
# from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
# Read image.
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\oranges.jpg'
img = cv.imread(path)
 
def rescaleImage(Image, scaler_factor):
    height = int(Image.shape[0] * scaler_factor)
    width = int(Image.shape[1] * scaler_factor)
    dimension = (width,height)

    return cv.resize(Image, dimension, interpolation=cv.INTER_AREA)
# img = rescaleImage(img,0.5)

# copy = img
# grayImage = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
# cv.imshow('Black white image', blackAndWhiteImage)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# # apply gaussian blur
# blur = cv.GaussianBlur(gray, (9,9), 0)


# # threshold
# thresh = cv.threshold(blur,128,255,cv.THRESH_BINARY)[1]


# # apply close and open morphology to smooth
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
# thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
# thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

# # draw contours and get centroids
# circles = img.copy()
# contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# for cntr in contours:
#     cv.drawContours(circles, [cntr], -1, (255,255,255), -1)
#     M = cv.moments(cntr)
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
#     x = round(cx)
#     y = round(cy)
    
#     circles[y-2:y+3,x-2:x+3] = (0,255,0)
#     cv.circle(circles, (x, y), 5, (255, 255, 255), -1)
            

# cv.imshow("circles", circles)

# # Printing the number of oranges in the picture
# print("The number of Oranges in the Image is:",len(contours))

# Question 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\orangetree.jpg'
img2 = cv.imread(path)
img2 = rescaleImage(img2,0.1)
copy = img2

def open(image, kernel):
    eroded = cv.erode(image, kernel)
    dilated = cv.dilate(eroded, kernel)
    return dilated


def close(image, kernel):
    dilated = cv.dilate(image, kernel)
    eroded = cv.erode(dilated, kernel)
    return eroded
def findcirlce(image,alpha,beta):
    grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    (_, image) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    opening = open(image,kernel)
    closing = close(opening, kernel)
    label_im = label(closing)
    regions = regionprops(label_im)
    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (num!=0 and (area>10) and (convex_area/area <alpha) #100
        and (convex_area/area >beta)):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    count = len(masks)
    print(count)
findcirlce(img,1.05,0.95)
findcirlce(img2,1.22,0.95)
cv.waitKey(0)
cv.destroyAllWindows()
