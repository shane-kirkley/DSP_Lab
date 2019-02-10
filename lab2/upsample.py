import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import RectBivariateSpline

#######################################
# example bi-linear interpolation
# x = np.asarray([1, 2], np.int)
# y = np.asarray([5, 6], np.int)

# z = np.asarray([[50, 100], [200, 300]], np.float)

# interp_spline = RectBivariateSpline(y, x, z, kx=1, ky=1)

# x1 = np.linspace(1, 2, 3)
# y1 = np.linspace(5, 6, 3)
# ivals = interp_spline(y1, x1)
#######################################

pilim = Image.open(sys.argv[1])
pilim = pilim.convert('L')  # convert to monochrome
imdat = np.asarray(pilim, np.float) # put PIL image into numpy array of floats

out_filename = sys.argv[2]
N = int(sys.argv[3])

R, C = imdat.shape
print(f"Original Image size: Rows={R}, Cols={C}")

img_up = np.zeros((int(R*N), int(C*N)))

# nested for loops over the smaller img
#TODO: this is very slow, make it better
for i in range(R - 1):
    for j in range(C - 1):
        x = np.array([i, i+1])
        y = np.array([j, j+1])
        z = imdat[i:i+2, j:j+2]

        interp_spline = RectBivariateSpline(y, x, z, kx=1, ky=1)
        x1 = np.linspace(i, i+1, N+1)
        y1 = np.linspace(j, j+1, N+1)
        ivals = interp_spline(y1, x1)
        img_up[i*N:i*N+(N+1), j*N:j*N+(N+1)] = ivals

dR, dC = img_up.shape
print(f"Upsampled size: Rows={dR}, Cols={dC}")

Image.fromarray(img_up).convert('L').save(out_filename)
Image.fromarray(img_up).show()