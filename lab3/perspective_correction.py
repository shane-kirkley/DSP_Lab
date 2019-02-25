import numpy as np
import sys
import math
from numpy.linalg import inv
from PIL import Image
import bilinear_interp as bi

# choose 20 points, from left to right on two seperate straight lines (10 on each)
# og image size: 1067 x 800

# line 1:
# (294, 339), (312, 334), (331, 329), (358, 322), (388, 314), 
# (426, 304), (475, 291), (537, 275), (625, 250), (754, 214)

# line 2:
# (290, 455), (308, 460), (328, 465), (353, 471), (384, 479), 
# (421, 488), (469, 499), (532, 515), (617, 539), (745, 573)

# specify points on new image to translate original points to

# line 1 translated:
# (90, 200), (180, 200), (270 ,200), (360, 200), (450, 200), 
# (540, 200), (630, 200), (720, 200), (810, 200), (900, 200)

# line 2 translated:
# (90, 600), (180, 600), (270, 600), (360, 600), (450, 600), 
# (540, 600), (630, 600), (720, 600), (810, 600), (900, 600)

def psuedo_inverse(A):
    return inv(A.T.dot(A)).dot(A.T)

pilim = Image.open("PC_test_2.jpg")
imdat = np.asarray(pilim, np.float)  # put PIL image into numpy array of floats
new_imdat = np.zeros(imdat.shape)

# array of selected points from original image (c, d)
source = np.array([[294, 339], [312, 334], [331, 329], [358, 322], [388, 314], \
                   [426, 304], [475, 291], [537, 275], [625, 250], [754, 214], \
                   [290, 455], [308, 460], [328, 465], [353, 471], [384, 479], \
                   [421, 488], [469, 499], [532, 515], [617, 539], [745, 573] ])

# array of points in new image (a, b)
target = np.array([[90, 200] , [180, 200], [270, 200], [360, 200], [450, 200], \
                   [540, 200], [630, 200], [720, 200], [810, 200], [900, 200], \
                   [90, 600] , [180, 600], [270, 600], [360, 600], [450, 600], \
                   [540, 600], [630, 600], [720, 600], [810, 600], [900, 600] ])

T = target.flatten()
V = np.zeros((40, 8))

# for each point fill in two rows of V (can this be done cleaner?)
for i in range(source.shape[0]):
    V[2*i] = [source[i,0], source[i,1], 1, 0, 0, 0, -1*target[i,0]*source[i,0], -1*target[i,0]*source[i,1]]
    V[2*i+1] = [0, 0, 0, source[i,0], source[i,1], 1, -1*target[i,1]*source[i,0], -1*target[i,1]*source[i,1]]

# find H by getting inverse of V, dot with T, append 1 and reshape to square.
h = np.dot(psuedo_inverse(V), T)
h = np.append(h, 1)
H = np.reshape(h, (3,3))

# verify H is correct for some known points
# v = np.dot(H, np.array([800, 300, 1]))
# print(v[0]/v[2])
# print(v[1]/v[2])
# print(v[2])

# loop through new image pixels and remap
for a in range(new_imdat.shape[0]):
    for b in range(new_imdat.shape[1]):
        # find the c,d from input image corresponding to a,b
        # using inverse of H...
        
        # use bilinear_interp to get value of corresponding pixel



Image.fromarray(new_imdat.astype('uint8')).save("test1.jpg")
Image.fromarray(new_imdat.astype('uint8')).show()

