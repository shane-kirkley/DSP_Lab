import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import RectBivariateSpline

#######################################
# example bi-linear interpolation
x = np.asarray([1, 2], np.int)
y = np.asarray([5, 6], np.int)

z = np.asarray([[50, 100], [200, 300]], np.float)

interp_spline = RectBivariateSpline(y, x, z, kx=1, ky=1)

x1 = np.linspace(1, 2, 3)
y1 = np.linspace(5, 6, 3)
ivals = interp_spline(y1, x1)
#######################################