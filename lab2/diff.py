import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

UPFACTOR = 2    # increase visibility of difference

pilim1 = Image.open(sys.argv[1]).convert('L')
pilim2 = Image.open(sys.argv[2]).convert('L')
filename_out = sys.argv[3]

imdat1 = np.asarray(pilim1, np.float) # put PIL image into numpy array of floats
imdat2 = np.asarray(pilim2, np.float)

if imdat1.shape != imdat2.shape:
    print("Image dimensions not equivalent")

diff = np.clip(np.abs(imdat1 - imdat2) * UPFACTOR, 0, 255)

Image.fromarray(diff).convert('L').save(filename_out)
Image.fromarray(diff).show()
