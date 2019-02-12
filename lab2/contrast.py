import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

pilim = Image.open(sys.argv[1])
pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float)  # put PIL image into numpy array of floats

filename_out = sys.argv[2]

histogram = np.zeros(256)

# create histogram
for intensity in range(0, 255):
    histogram[intensity] = (imdat == intensity).sum()

S = histogram[1:255].sum()
P = S / 254
remap = np.zeros(256)
remap[255] = 255
T = P
curr_sum = 0
out_val = 1
last_idx = 1

# this loop creates remap array
for intensity in range(1, 255):
    curr_sum = curr_sum + histogram[intensity]
    remap[intensity] = out_val
    if curr_sum > T:
        out_val = round(curr_sum / P)
        T = out_val * P

# change intensity values in image to remapped values.
# this could be done in the remap array loop instead of separately.
new_imdat = np.zeros(imdat.shape)
for intensity in range(0, 256):
    new_imdat[imdat == intensity] = remap[intensity]

# create new histogram
new_histogram = np.zeros(256)
for intensity in range(0, 256):
    new_histogram[intensity] = (new_imdat == intensity).sum()

# plot histogram comparison
plt.subplot(1, 2, 1)
plt.plot(histogram)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Histogram of ' + sys.argv[1])
plt.subplot(1, 2, 2)
plt.plot(new_histogram)
plt.xlabel('Intensity')
plt.title('Remapped Histogram of ' + sys.argv[1])
plt.show()

Image.fromarray(new_imdat).convert('L').save(filename_out)
Image.fromarray(new_imdat).show()
