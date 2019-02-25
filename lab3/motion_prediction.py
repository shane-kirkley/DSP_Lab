import numpy as np
import sys
import math
from numpy.linalg import inv
from PIL import Image
import matplotlib.pyplot as plt
import me_method as me

filename_out = "xipredict.jpg"

# N is block size
N = 16

# get frame n data
pilim = Image.open("xi01.jpg")
n_imdat_rgb = np.asarray(pilim, np.float)
n_imdat_bw = np.asarray(pilim.convert('L'), np.float)

# get frame n+1 data
pilim = Image.open("xi02.jpg")
n1_imdat_rgb = np.asarray(pilim, np.float)
n1_imdat_bw = np.asarray(pilim.convert('L'), np.float)

# create prediction image data array
predict_imdat = np.zeros(n1_imdat_rgb.shape)

U = []
V = []
X = []
Y = []

"""
U , V , X, and Y are lists. U and V hold the x and y components
of the vectors to be graphed, and X and Y hold the x and y coordinates of the
locations of the vector tails. So together, U [n], V [n], X[n], and Y [n] specify the n’th
vector’s direction and tail location.
"""

# for each block in frame n+1, find the matched block in frame n
for row in range(math.floor(n1_imdat_bw.shape[0]/N)):
    for col in range(math.floor(n1_imdat_bw.shape[1]/N)):
        # extract the block from frame n+1
        block = n1_imdat_bw[row*N:row*N+N, col*N:col*N+N]
        # get prediction block, mse, and offsets for best match
        pred_block, mse, row_off, col_off = me.motion_match(row*N, col*N, 40, block, n_imdat_bw, n_imdat_rgb)
        # (X,Y) are coords in frame n (tail), (U,V) are coords in frame n+1 (head)
        U = np.append(U, col_off)
        V = np.append(V, -1*row_off)
        X = np.append(X, col)
        Y = np.append(Y, row)
        predict_imdat[row*N:row*N+N, col*N:col*N+N] = pred_block

plt.quiver(X,Y,U,V)
plt.title("Motion Estimation")
plt.xlabel("Horizontal Block Number")
plt.ylabel("Vertical Block Number")
plt.ion()
plt.show()
val = input('hit enter to continue')

Image.fromarray(predict_imdat.astype('uint8')).save(filename_out)
Image.fromarray(predict_imdat.astype('uint8')).show()
