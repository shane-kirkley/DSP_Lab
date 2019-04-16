#!/usr/local/bin/python3
import numpy as np
import math
import sys
from scipy.fftpack import dct
from scipy.fftpack import idct
from PIL import Image
import lab5_funcs as lab5

img_data = np.array(Image.open(sys.argv[1]).convert('L'))
filename_out = sys.argv[2]
loss_factor = int(sys.argv[3])
img_height, img_width = img_data.shape
coeffs = lab5.dctmgr(img_data, loss_factor)
recovered = lab5.idctmgr(coeffs, img_width, img_height, loss_factor)
Image.fromarray(recovered).convert('L').save(filename_out)
