#!/usr/local/bin/python3
import numpy as np
import math
import sys
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
import useful_arrays

"""
Forward DCT for use on 2D arrays. Applies DCT to rows then columns.
"""
def forward_dct(input):
    return dct(dct(input, norm='ortho').T, norm='ortho').T

"""
Inverse forward DCT for use on 2D arrays. Applies idct to columns then rows.
"""
def inverse_forward_dct(input):
    return idct(idct(input.T, norm='ortho').T, norm='ortho')

"""
Helper fcn to return an array of shape (nblocks, rows, cols) 
where nblocks * rows * cols == arr.size
"""
def blockshape(arr, rows, cols):
    h, w = arr.shape
    return arr.reshape(h//rows, rows, -1, cols).swapaxes(1,2).reshape(-1, rows, cols)

"""
Helper fcn to return 1d array of size 64 of zig-zag order elements from 8x8 2d array
"""

def zigzag(input):
    zz = useful_arrays.zz
    output = np.zeros(64)
    for i in range(64):
        idx = zz[i]
        output[i] = input[idx[0],idx[1]]
    return output


"""
Helper fcn to reverse the zigzag function (column to 2d-array)
"""
def zigzag_reverse(input):
    zz = useful_arrays.zz
    output = np.zeros((8,8))
    for i in range(64):
        idx = zz[i]
        output[idx[0], idx[1]] = input[i]
    return output

"""
Lab assignment 1
Input: 2D array of grayscale image data
Output: Coefficient array of size 64 x N, where N is number of 8x8 blocks in image.
"""
def dctmgr(img_data):
    # divide into non-overlapping 8x8 blocks
    rows = img_data.shape[0]
    cols = img_data.shape[1]
    nblocks = int(rows/8) * int(cols/8)
    chunked_data = blockshape(img_data, 8, 8)
    
    coeffs = np.zeros((64, nblocks))
    block_num = 0
    for block in chunked_data:
        # transform each block
        block = forward_dct(block)
        # zig-zag data from blocks to coeffecient columns.
        coeffs[:,block_num] = zigzag(block)
        # replace DC coeffs with differentially encoded values
        if (block_num % cols) != 0:
            coeffs[0, block_num] = coeffs[0, block_num] - coeffs[0, block_num-1]
        block_num = block_num + 1

    return coeffs

"""
Lab assignment 2
Input: Coefficient array of size 64 x N
Output: 2D array of grayscale image data.
"""
def idctmgr(coeff, block_size):
    nblocks = coeff.shape[1]
    img_data = np.zeros((nblocks*block_size, nblocks*block_size))
    
    # for each block b, column coeffs[:,b] is used to reconstruct the block.
    # undo prediction with DC coeffs
    # reverse zig-zag into a block
    # use inverse 2d dct function on block.
    # put data into 2d array of img data.
    
    raise NotImplementedError
