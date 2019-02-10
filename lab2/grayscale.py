import numpy as np
from PIL import Image
import sys

Y = np.array([.299, .587, .114])

# converts array of rgb image data to grayscale
def grayscale(img):
    return np.clip(np.dot(img[:,:,:3], Y), 0, 255)

pil_image_in = Image.open(sys.argv[1])
image_in = np.asarray(pil_image_in, np.uint8)
filename_out = sys.argv[2]

image_out = grayscale(image_in)

# Output the image to a file
Image.fromarray(image_out.astype(np.uint8)).save(filename_out)

# view the image
Image.fromarray(image_out).show()