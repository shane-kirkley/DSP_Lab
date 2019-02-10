import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from scipy import signal

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

pilim = Image.open(sys.argv[1])

if sys.argv[2]:
    out_filename = sys.argv[2]
else:
    out_filename = 'test_img_edge.jpg'

pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float) # put PIL image into numpy array of floats

# it isn't necessary to have a pair of nested loops that process each pixel seperately
grad_x = signal.convolve2d(imdat, kernel_x)
grad_y = signal.convolve2d(imdat, kernel_y)

edge = np.sqrt(grad_x**2 + grad_y**2)

#TODO: Better way to implement threshold?
threshold = 200
edge[edge < threshold] = 0

Image.fromarray(edge).convert('L').save(out_filename)
Image.fromarray(edge).show()
