import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys


pilim = Image.open(sys.argv[1])
pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float) # put PIL image into numpy array of floats

histogram = np.zeros(256)

for intensity in range(0,255):
    histogram[intensity] = (imdat == intensity).sum()

plt.plot(histogram)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Histogram')
plt.show()

S = histogram[1:255].sum()
P = S / 254

remap = np.zeros(256)
remap[0] = histogram[0]
remap[255] = histogram[255]

T = P
curr_sum = 0
out_val = 1
last_idx = 1


# this loop creates remap array and remaps pixels of image
for intensity in range(1,255):
    curr_sum = curr_sum + histogram[intensity]
    remap[intensity] = out_val
    imdat[np.where(imdat == intensity)] = out_val
    if curr_sum > T:
        out_val = round(curr_sum / P)
        T = out_val * P

plt.plot(remap)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Remapped Histogram')
plt.show()

Image.fromarray(imdat).convert('L').save('test_img_contrast1.jpg')
Image.fromarray(imdat).show()