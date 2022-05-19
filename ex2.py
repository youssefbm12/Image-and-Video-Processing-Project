import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import imageio as iio
from numpy import pi
from numpy import r_
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\geese.jpg'
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.T, norm='ortho' ).T, norm='ortho' )
def magnitude_spectrume(img):
    ftimage = np.fft.fft2(img)
    ftimage = np.fft.fftshift(ftimage)
    magnitude_spectrum = 20 * np.log(np.abs(ftimage))

    return 
    
dct = dct2(img)
dct =  np.sort(dct)[::-1]
imsize = img.shape
dct = np.zeros(imsize)
img_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:i+8,j:j+8] = dct2( img[i:i+8,j:j+8] )

mag = magnitude_spectrume(dct)
K = 10
alpha = 10

# thresh = 0.012
dct_thresh = dct * (abs(dct) > (K*np.ones(dct)))


for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        img_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )

gaussian = np.random.normal(0, math.sqrt(0.002), K)

###### LAB 6 


imsize = img.shape
dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:i+8,j:j+8] = dct2( img[i:i+8,j:j+8] )
pos = 128



thresh = 0.001
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))


plt.figure()
plt.imshow(dct_thresh,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "Thresholded 8x8 DCTs of the image")

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

print( "Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))
img_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        img_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )
        
        
plt.figure()
plt.imshow( np.hstack( (dct_thresh, img_dct) ) ,cmap='gray')
plt.title("Comparison between original and DCT compressed images" )
plt.show()
cv.waitKey(0)