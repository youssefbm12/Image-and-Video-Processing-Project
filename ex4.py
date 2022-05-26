from re import I
from statistics import covariance
from tokenize import Double
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.decomposition import PCA
from imutils import paths
import os
from statistics import mean
from numpy.linalg import eig
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
images = [img,img2,img3,img4,img5]
def rescaleImage(Image, scaler_factor):
    height = int(Image.shape[0] * scaler_factor)
    width = int(Image.shape[1] * scaler_factor)
    dimension = (width,height)

    return cv.resize(Image, dimension, interpolation=cv.INTER_AREA)
# img = rescaleImage(img,0.2)
# img4= rescaleImage(img4,0.5)
# img5 = rescaleImage(img5,0.2)
# cv.imshow("Original Image",img4)
# faces={}
# faces[0]=img
# faces[1]=img2
# faces[2]=img3
# faces[3]=img4
# faces[4]=img5



# NUM_EIGEN_FACES = 10

# MAX_SLIDER_VALUE = 255

# dirName = "images"
# images = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# sz = images[0].shape

# data = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# print("Calculating PCA ", end="...")

# mean, eigenVectors = cv.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)

# print ("DONE")
# averageFace = mean.reshape(sz)
# eigenFaces = [];
# for eigenVector in eigenVectors:
#     eigenFace = eigenVector.reshape(sz)
#     eigenFaces.append(eigenFace)
# print(eigenFaces)
# cv.namedWindow("Result", cv.WINDOW_AUTOSIZE)
# output = cv.resize(averageFace, (0,0), fx=2, fy=2)
# cv.imshow("Result", output)
# cv.namedWindow("Trackbars", cv.WINDOW_AUTOSIZE)
# sliderValues = []
# def createNewFace(*args):

#   output = averageFace
#   for i in range(0, NUM_EIGEN_FACES):
#     sliderValues[i] = cv.getTrackbarPos("Weight" + str(i), "Trackbars");
#     weight = sliderValues[i] - MAX_SLIDER_VALUE/2
#     output = np.add(output, eigenFaces[i] * weight)
#     output = cv.resize(output, (0,0), fx=2, fy=2)

#   cv.imshow("Result", output)
# # for i in range(0, NUM_EIGEN_FACES):
# #     sliderValues.append(MAX_SLIDER_VALUE/2)
# #     cv.createTrackbar( "Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2, MAX_SLIDER_VALUE, createNewFace)
# #   # You can reset the sliders by clicking on the mean image.
    
# #     print('''Usage:Change the weights using the slidersClick on the result window to reset slidersHit ESC to terminate program.''')


def image_to_vector(image: np.ndarray) -> np.ndarray:
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))
facematrix = []
for i in range(5):
  facematrix.append(images[i].flatten() )
# X1= image_to_vector(img) 
# X2= image_to_vector(img2)
# X3= image_to_vector(img3) 
# X4= image_to_vector(img4) 
# X5= image_to_vector(img5)
# length = max(len(X1),len(X2),len(X3),len(X4),len(X5))
# X = np.zeros(shape=(5,length))
# for i in range(length):
#   X[0][i]=X1[i]
#   X[1][i]=X2[i]
#   X[2][i]=X3[i]
#   X[3][i]=X4[i]
#   X[4][i]=X5[i]

# sum = np.mean(X)
# print(sum)
# X=X-sum
# covariance_x=np.dot(X,np.transpose(X))
# w,v = eig(covariance_x)
# v=w+v
# cv.imshow("Ca",w)
facematrix = np.array(facematrix)
pca = PCA().fit(facematrix)


faceshape = img.shape
n_components = 4  #len(pca)
eigenfaces = pca.components_[:]
print(len(eigenfaces))
# Show the first 16 eigenfaces
fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=(8,10))
for i in range(len(eigenfaces)):
    axes[i%2][i//3].imshow(eigenfaces[i].reshape(faceshape))
    #plt.title("Eigen Faces",i)
plt.show()
cv.waitKey(0) 
cv.destroyAllWindows()  
