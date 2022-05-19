import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.decomposition import PCA
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\kid2.jpg'
img = cv.imread(path)
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\kid_ph1.jpg'
img2 = cv.imread(path)
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\kid_ph2.jpg'
img3 = cv.imread(path)
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\kid_ph3.jpg'
img4 = cv.imread(path)
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\kid_ph4.jpg'
img5 = cv.imread(path)
def rescaleImage(Image, scaler_factor):
    height = int(Image.shape[0] * scaler_factor)
    width = int(Image.shape[1] * scaler_factor)
    dimension = (width,height)

    return cv.resize(Image, dimension, interpolation=cv.INTER_AREA)
img = rescaleImage(img,0.2)
img4= rescaleImage(img4,0.5)
img5 = rescaleImage(img5,0.2)
cv.imshow("Original Image",img4)
faces={}
faces[0]=img
faces[1]=img2
faces[2]=img3
faces[3]=img4
faces[4]=img5
facematrix = []
facelabel = []
for key,val in faces.items():
    if key.startswith("s40/"):
        continue # this is our test set
    if key == "s39/10.pgm":
        continue # this is our test set
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])
 
# Create facematrix as (n_samples,n_pixels) matrix
facematrix = np.array(facematrix)
cv.waitKey(0)   