import numpy as np
from PIL import Image

pil_image_in = Image.open('test01.jpg')
image_in = np.asarray(pil_image_in, np.uint8)
print(f"shape of image = {image_in.shape}")

rows = image_in.shape[0]
cols = image_in.shape[1]

print(f"rows = {rows}; cols = {cols}")

# modify image here as appropriate

# Output the image to a file
Image.fromarray(image_in).save('test_img_out.jpg')

# view the image
Image.fromarray(image_in).show()