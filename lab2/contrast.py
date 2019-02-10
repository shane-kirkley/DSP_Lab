import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys


pilim = Image.open(sys.argv[1])
pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float)  # put PIL image into numpy array of floats

filename_out = sys.argv[2]

histogram = np.zeros(256)

for intensity in range(0, 255):
    histogram[intensity] = (imdat == intensity).sum()

plt.plot(histogram)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Histogram')
plt.show()

S = histogram[1:255].sum()
P = S / 254

remap = np.zeros(256)
T = P
curr_sum = 0
out_val = 1
last_idx = 1

# this loop creates remap array
for intensity in range(1, 255):
    curr_sum = curr_sum + histogram[intensity]
    if curr_sum > T:
        remap[last_idx:intensity] = out_val
        out_val = round(curr_sum / P)
        T = out_val * P
        last_idx = intensity

remap[remap == 0] = 254 # values that are missed when final target isn't hit
remap[0] = 0
remap[255] = 255

# change intensity values in image to remapped values
for intensity in range(0, 256):
    print("setting " + str(intensity) + " to " + str(remap[intensity]))
    imdat[imdat == intensity] = remap[intensity]

# create new histogram
for intensity in range(0, 256):
    histogram[intensity] = (imdat == intensity).sum()

plt.plot(histogram)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Remapped Histogram')
plt.show()

Image.fromarray(imdat).convert('L').save(filename_out)
Image.fromarray(imdat).show()
