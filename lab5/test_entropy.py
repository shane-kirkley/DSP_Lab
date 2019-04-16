#!/usr/local/bin/python3
import numpy as np
import math
import sys
from scipy.fftpack import dct
from scipy.fftpack import idct
from PIL import Image
import matplotlib.pyplot as plt
import lab5_funcs as lab5

def entropy(p):
    sum = 0
    for i in range(p.shape[0]):
        if p[i] > 0:
            sum += p[i] * np.log2(1/p[i])
    return sum

def nZero_entropy(nZero):
    n_zeros = np.zeros(64)
    for n in nZero:
        n_zeros[int(n)] += 1
    p_zeros = n_zeros / nZero.shape[0]
    return entropy(p_zeros)

def nBit_entropy(nBits):
    n_bits = np.zeros(15)
    for n in nBits:
        n_bits[int(n)] += 1
    p_bits = n_bits / nBits.shape[0]
    return entropy(p_bits)

def value_entropy(value):
    n_values = np.zeros(200)
    for n in value:
        if(n > 0):
            n_values[int(n)] += 1
    p_values = n_values / value.shape[0]
    return entropy(p_values) 

img_data = np.array(Image.open(sys.argv[1]).convert('L'))
img_rows = img_data.shape[0]
img_cols = img_data.shape[1]

compression_ratio = np.zeros(50)
compression_ratio[0] = 1

for loss_factor in range(1,50):
    symb = lab5.dctmgr(img_data, loss_factor)
    nZero_e = nZero_entropy(symb[:,0])
    
    nBit_e = nBit_entropy(symb[:,1])
    value_e = value_entropy(symb[:,2])

    bit_estimate = nZero_e + nBit_e + value_e
    total_bit_estimate = bit_estimate * symb.shape[0]
    
    cr = (8 * img_rows * img_cols) / total_bit_estimate
    compression_ratio[loss_factor] = cr

plt.plot(compression_ratio)
plt.xlim(left=1)
plt.xlabel("Loss Factor")
plt.xticks(np.arange(0,51,5))
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio as function of loss factor for %s" % sys.argv[1])
plt.show()