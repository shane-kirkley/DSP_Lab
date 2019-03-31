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
Lab assignment 1
Input: 2D array of grayscale image data
Output: Coefficient array of size 64 x N, where N is number of 8x8 blocks in image.
"""
def dctmgr(img_data):
    # divide into non-overlapping 8x8 blocks
    # transform each block
    # Write forward 2d DCT routine
    raise NotImplementedError

"""
Lab assignment 2
Input: Coefficient array of size 64 x N
Output: 2D array of grayscale image data.
"""
def idctmgr(coeff):
    raise NotImplementedError
