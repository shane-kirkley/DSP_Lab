#!/usr/local/bin/python3
import numpy as np
import math
import sys
from PIL import Image
from scipy.fftpack import dct

"""
Forward DCT for use on 2D arrays. Applies DCT to rows then columns.
"""
def forward_dct(input):
    return dct(dct(input).T).T

"""
Helper fcn to return an array of shape (nblocks, rows, cols) 
where nblocks * rows * cols == arr.size
"""
def blockshape(arr, rows, cols):
    h, w = arr.shape
    return arr.reshape(h//rows, rows, -1, cols).swapaxes(1,2).reshape(-1, rows, cols)

"""
Helper fcn to return 1d array of zig-zag order elements from 2d array
"""
def zigzag(arr):
    np.concatenate([np.diagonal(arr[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-arr.shape[0], arr.shape[0])])

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
    
    # transform each block
    for block in chunked_data:
        block = forward_dct(block)
    
    # zig-zag data from blocks to coefficient columns
    coeffs = np.zeros((64, nblocks))
    for block in range(nblocks):
        # get correct block from chunked_data?
        coeffs[:block] = zigzag(chunked_data[block])
    

"""
Lab assignment 2
Input: Coefficient array of size 64 x N
Output: 2D array of grayscale image data.
"""
def idctmgr(coeff):
    raise NotImplementedError
