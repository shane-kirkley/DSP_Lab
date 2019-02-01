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