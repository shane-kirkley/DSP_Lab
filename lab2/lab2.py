import numpy as np
from PIL import Image

Y = np.array([.299, .587, .114])

# converts array of rgb image data to grayscale
def grayscale(img):
    return np.clip(np.dot(img[:,:,:3], Y), 0, 255)

pil_image_in = Image.open('test01.jpg')
image_in = np.asarray(pil_image_in, np.uint8)

image_out = grayscale(image_in)

# Output the image to a file
# getting error 'cannot write mode F as JPEG', convert('L') fixes it
Image.fromarray(image_out).convert('L').save('test_img_out1.jpg')

# view the image
Image.fromarray(image_out).show()