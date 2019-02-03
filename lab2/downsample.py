import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# usage: downsample.py image_in.jpg image_out.jpg N

# Assume we have an image with R rows and C columns, where R and C are both divisible by N.
# N is Downsampling Factor
# create a new image of size R/N by C/N.
# Divide original image into non-overlapping blocks of size NxN, average pixel values from each
# block to obtain a pixel value for subsampled image.

pilim = Image.open(sys.argv[1])
pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float) # put PIL image into numpy array of floats

out_filename = sys.argv[2]
N = int(sys.argv[3])

R, C = imdat.shape
print(f"Original Image size: Rows={R}, Cols={C}")

img_down = np.zeros((int(R/N), int(C/N)))
for i in range(int(R/N)):
    for j in range(int(C/N)):
        img_down[i, j] = np.average(imdat[i*N:(i*N)+N, j*N:(j*N)+N])

dR, dC = img_down.shape
print(f"Downsampled size: Rows={dR}, Cols={dC}")

Image.fromarray(img_down).convert('L').save(out_filename)
Image.fromarray(img_down).show()