#!/usr/local/bin/python3
import numpy as np
import math
import sys
from scipy.fftpack import dct
from scipy.fftpack import idct
import useful_arrays

BLOCK_SIZE = 8

"""
Forward DCT for use on 2D arrays. Applies DCT to rows then columns.
"""
def forward_dct(input):
    return dct(dct(input, norm='ortho').T, norm='ortho').T

"""
Inverse DCT for use on 2D arrays. Applies idct to columns then rows.
"""
def inverse_dct(input):
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
Quantizer function
"""
def quantize(block, loss_factor):
    Q = useful_arrays.Q
    return np.floor(block/(Q * loss_factor) + 0.5)

"""
Inverse quantizer function
"""
def inverse_quantize(block, loss_factor):
    Q = useful_arrays.Q
    return block * Q * loss_factor

"""
Image data DCT function
Input: 2D array of grayscale image data
Output: Variable length encoded array of quantized DCT coefficients for image
"""
def dctmgr(img_data, loss_factor):
    # divide into non-overlapping 8x8 blocks
    rows = img_data.shape[0]
    cols = img_data.shape[1]
    nblocks = int(rows/BLOCK_SIZE) * int(cols/BLOCK_SIZE)
    chunked_data = blockshape(img_data, BLOCK_SIZE, BLOCK_SIZE)
    
    coeffs = np.zeros((64, nblocks))
    encoded = np.zeros(coeffs.shape)
    block_num = 0
    
    for block in chunked_data:
        # transform each block
        block = forward_dct(block)
        # quantize block
        block = quantize(block, loss_factor)
        # zig-zag data from blocks to coeffecient columns.
        coeffs[:,block_num] = zigzag(block)
        encoded[:,block_num] = coeffs[:,block_num] 
        # replace DC coeffs with differentially encoded values
        if (block_num % (cols/BLOCK_SIZE)) != 0: # not at beginning of a row 
            encoded[0, block_num] = coeffs[0, block_num] - coeffs[0, block_num - 1]
        else:
            encoded[0, block_num] = coeffs[0, block_num]
        block_num = block_num + 1
    
    # variable length encoding
    encoded = run_bits_value(encoded)  
    return encoded

"""
Image data Inverse DCT and encoding
Input: - coeffs: variable length encoded array of quantized DCT coefficients
       - Image width and height
       - loss_factor in quantization
Output: 2D array of grayscale image data.
"""
def idctmgr(coeffs, img_width, img_height, loss_factor):
    rows = img_height # for my sanity
    cols = img_width  
    img_data = np.zeros((rows, cols))
    
    # variable length decoding
    coeffs = irun_bits_value(coeffs)
    nblocks = coeffs.shape[1]

    for block_num in range(nblocks):
    # for each block b, column coeffs[:,b] is used to reconstruct the block.
        # undo prediction with DC coeffs
        if (block_num % (cols/BLOCK_SIZE)) != 0:
           coeffs[0, block_num] += coeffs[0, block_num - 1]
        # reverse zig-zag into a block.
        block = zigzag_reverse(coeffs[:, block_num])
        # inverse quantization of block.
        block = inverse_quantize(block, loss_factor)
        # inverse 2d dct function on block.
        block = inverse_dct(block)
        # put block into img data.
        # (img_row, img_col) is coords of top left corner of block
        img_row = int((block_num * BLOCK_SIZE) / cols) * BLOCK_SIZE
        img_col = int((block_num * BLOCK_SIZE) % cols)
        img_data[img_row:img_row+8, img_col:img_col+8] = block

    return img_data

"""
Triplet pre-processor and inverse process functions
"""
def run_bits_value(coeffs):
    # input: DCT coefficient array
    # output: 2d matrix symb, size nx3, each row storing (nZeros, nBits, value)
    DC_BITS = 12
    AC_BITS = 11
    symb = []
    
    for block in coeffs.T:
        nZeros = 0
        symb.append([0, DC_BITS, np.float(block[0])])
        for coeff in block[1:]:
            if coeff == 0:
                nZeros += 1
            else:
                # add new entry to symb
                symb.append([nZeros, AC_BITS, np.float(coeff)])
                nZeros = 0
        # EOB        
        symb.append([0,0,0])
        
    # all blocks processed, convert to np array
    return np.asarray(symb)

def irun_bits_value(symb):
    # input: 2d matrix symb
    # output: DCT coefficient array
    coeffs = np.empty((0,64))
    block = np.empty(0)
    for row in symb:
        if (row[0] == 0) and (row[1] == 0) and (row[2] == 0):
            # add trailing zeros
            num_zeros = 64 - block.shape[0]
            zeros = np.zeros(num_zeros)
            block = np.append(block,zeros)
            coeffs = np.vstack((coeffs, block))
            block = np.empty(0)
            continue
        nZeros, nBits, value = row
        if(nZeros > 0):
            zero = np.zeros(int(nZeros))
            block = np.append(block, zero)
        block = np.append(block,value)
    return coeffs.T

"""
psnr gives peak signal to noise ratio in dB between two images.
"""
def psnr(im_1, im_2):
    if im_1.shape != im_2.shape:
        print(f"ERROR in psnr: images are not same size")
        sys.exit(-1)
    denom = np.sum( (im_1 - im_2)**2 ) / (im_1.shape[0]*im_1.shape[1])
    if denom == 0:
        return 0
    else:
        PSNR = 10 * np.log10( 255**2 / denom )
        return PSNR
