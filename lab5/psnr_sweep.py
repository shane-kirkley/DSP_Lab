#!/usr/local/bin/python3
import numpy as np
import math
import sys
from PIL import Image
import lab5_funcs as lab5
import matplotlib.pyplot as plt

SWEEP_LENGTH = 38
GEN_SWEEP_IMG = True

filename = sys.argv[1]
img_data = np.array(Image.open(filename).convert('L'))
img_height, img_width = img_data.shape
filename = filename[:-4]
img_height, img_width = img_data.shape
# save as jpg for comparison
Image.fromarray(img_data).convert('L').save(filename + ".jpg")
img_data = np.array(Image.open(filename + ".jpg").convert('L'))
psnr = np.zeros(SWEEP_LENGTH)

if(GEN_SWEEP_IMG):
    step = int(img_width/SWEEP_LENGTH)
    img_out = np.zeros(img_data.shape)
    img_out[:,:1*step] = img_data[:,:1*step]

for loss_factor in range(1,SWEEP_LENGTH):
    coeffs = lab5.dctmgr(img_data, loss_factor)
    img_loss = lab5.idctmgr(coeffs, img_width, img_height, loss_factor)
    psnr[loss_factor] = lab5.psnr(img_data, img_loss)
    if(GEN_SWEEP_IMG):
        step_idx = loss_factor*step
        img_out[:,step_idx:step_idx+step] = img_loss[:,step_idx:step_idx+step]

if(GEN_SWEEP_IMG):
    Image.fromarray(img_out).convert('L').save("sweep.jpg")

plt.plot(psnr)
plt.xlim(left=1)
plt.xlabel("loss_factor")
plt.ylabel("PSNR (dB)")
plt.ylim(bottom = 15)
plt.xticks(np.arange(1,SWEEP_LENGTH))
plt.title("PSNR between Original and Lossy '%s" % filename + ".jpg'")
plt.show()
