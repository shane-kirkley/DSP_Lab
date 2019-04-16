#!/usr/local/bin/python3
import numpy as np
import math
import sys
from PIL import Image
import lab5_funcs as lab5



# open image, prepare name
filename = sys.argv[1]
img_data = np.array(Image.open(filename).convert('L'))
filename = filename[:-4]
img_height, img_width = img_data.shape
# save as jpg for comparison
Image.fromarray(img_data).convert('L').save(filename + ".jpg")
img_data = np.array(Image.open(filename + ".jpg").convert('L'))

# generate images with loss-factors of 1, 10, and 20
coeffs = lab5.dctmgr(img_data, 1)
img_loss1 = lab5.idctmgr(coeffs, img_width, img_height, 1)
Image.fromarray(img_loss1).convert('L').save(filename + "_loss1.jpg")

coeffs = lab5.dctmgr(img_data, 10)
img_loss10 = lab5.idctmgr(coeffs, img_width, img_height, 10)
Image.fromarray(img_loss10).convert('L').save(filename + "_loss10.jpg")

coeffs = lab5.dctmgr(img_data, 20)
img_loss20 = lab5.idctmgr(coeffs, img_width, img_height, 20)
Image.fromarray(img_loss20).convert('L').save(filename + "_loss20.jpg")

# get psnr between original images and lossy images
psnr1 = lab5.psnr(img_data, img_loss1)
psnr10 = lab5.psnr(img_data, img_loss10)
psnr20 = lab5.psnr(img_data, img_loss20)

print("PSNR with loss_factor = 1: " + str(psnr1) + "dB")
print("PSNR with loss_factor = 10: " + str(psnr10) + "dB")
print("PSNR with loss_factor = 20: " + str(psnr20) + "dB")