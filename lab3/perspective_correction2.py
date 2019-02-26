import numpy as np
import sys
import math
from numpy.linalg import inv
from PIL import Image
import bilinear_interp as bi

# choose 20 points, from left to right on two seperate straight lines (10 on each)
# og image size: 1067 x 800

def psuedo_inverse(A):
    return inv(A.T.dot(A)).dot(A.T)

pilim = Image.open("PC_test_2.jpg")
imdat = np.asarray(pilim, np.float)  # put PIL image into numpy array of floats
new_imdat = np.zeros(imdat.shape)

# array of source image points (c, d)
source = np.array([[339, 294], [334, 312], [329, 331], [322, 358], [314, 388], \
                   [304, 426], [291, 475], [275, 537], [250, 625], [214, 754], \
                   [455, 290], [460, 308], [465, 328], [471, 353], [479, 384], \
                   [488, 421], [499, 469], [515, 532], [539, 617], [573, 745] ])

# array of points in new image (a, b)
target = np.array([[300, 90] , [300, 180], [300, 270], [300, 360], [300, 450], \
                   [300, 540], [300, 630], [300, 720], [300, 810], [300, 900], \
                   [500, 90] , [500, 180], [500, 270], [500, 360], [500, 450], \
                   [500, 540], [500, 630], [500, 720], [500, 810], [500, 900] ])
                   
T = target.flatten()
V = np.zeros((40, 8))
""" V is hard coded for 20 points, could change to arbitrary size by
    appending rows instead of filling in place. """
# for each point fill in two rows of V (can this be done cleaner?)
for i in range(source.shape[0]):
    c = source[i,0]
    d = source[i,1]
    a = target[i,0]
    b = target[i,1]
    V[2*i] = [c, d, 1, 0, 0, 0, -1*a*c, -1*a*d]
    V[2*i+1] = [0, 0, 0, c, d, 1, -1*b*c, -1*b*d]

# find H by getting inverse of V, dot with T, append 1 and reshape to square.
h = np.dot(psuedo_inverse(V), T)
h = np.append(h, 1)
H = np.reshape(h, (3,3))

# loop through new image pixels
for a in range(new_imdat.shape[0]):
    for b in range(new_imdat.shape[1]):
        # find the c,d from input image corresponding to a,b output
        v = np.array([a, b, 1])
        cd = np.dot(inv(H), v)
        c = cd[0]/cd[2]
        d = cd[1]/cd[2]
        # use bilinear_interp to assign value of corresponding pixel
        new_imdat[a,b] = bi.bilinear_interp(d, c, imdat)

Image.fromarray(new_imdat.astype('uint8')).save("test2.jpg")
Image.fromarray(new_imdat.astype('uint8')).show()

